from __future__ import annotations

import io
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from flask import Flask, Response, abort, g, jsonify, render_template, request, send_file
from itsdangerous import BadSignature, URLSafeSerializer
from PIL import Image, UnidentifiedImageError

from .index import FileVectorIndex
from .jobs import ChatJobManager
from .models import AttachmentMetadata
from .rendering import SafeMarkdownRenderer
from .serve_chat import AttachmentRef, ServeChatEngine, SessionStore
from .session_log import SessionJsonlLogger


COOKIE_NAME = 'local_rag_session'
SUPPORTED_IMAGES = {
    'JPEG': ('image/jpeg', '.jpg'),
    'PNG': ('image/png', '.png'),
    'WEBP': ('image/webp', '.webp'),
}
SUGGESTED_QUESTIONS = [
    'CMP 機台開機前需要確認哪些項目？',
    '研磨作業的標準操作流程為何？',
    '發生異常警報時應如何處理？',
    '機台清潔與保養有哪些注意事項？',
]


class UploadValidationError(ValueError):
    pass


def create_app(*, config, embedding_backend, chat_backend,
               build_root: Path, output_root: Path) -> Flask:
    if not config.session_secret:
        raise ValueError('LOCAL_RAG_SESSION_SECRET is required for serve')
    app = Flask(__name__, template_folder=str(config.base_dir / 'templates'),
                static_folder=str(config.base_dir / 'static'))
    signer = URLSafeSerializer(config.session_secret, salt='local-rag-session-v1')
    index = FileVectorIndex.load(build_root, output_root=output_root)
    logger = SessionJsonlLogger(config.session_log_root,
                               enabled=config.session_log_enabled)
    sessions = SessionStore(
        config.session_ttl_seconds,
        on_expire=lambda session_id: logger.end(session_id, 'ttl_expired'),
    )
    engine = ServeChatEngine(
        config=config, index=index, embedding_backend=embedding_backend,
        chat_backend=chat_backend, sessions=sessions, session_logger=logger,
        renderer=SafeMarkdownRenderer(enabled=config.markdown_enabled),
    )
    jobs = ChatJobManager(engine, retention_seconds=config.job_retention_seconds)
    upload_root = (config.log_root or config.base_dir / 'runtime' / 'logs') / 'chat_uploads'
    _cleanup_stale_uploads(upload_root, config.job_retention_seconds)
    artifact_records = {record.record_id: record for record in index.records
                        if record.source.artifact_path}
    app.extensions['chat_job_manager'] = jobs

    @app.before_request
    def identify_session():
        token = request.cookies.get(COOKIE_NAME)
        try:
            session_id = signer.loads(token) if token else None
        except BadSignature:
            session_id = None
        if not isinstance(session_id, str) or len(session_id) != 32:
            session_id = uuid4().hex
            g.issue_session_cookie = True
        g.session_id = session_id

    @app.after_request
    def issue_session_cookie(response):
        if getattr(g, 'issue_session_cookie', False):
            response.set_cookie(
                COOKIE_NAME, signer.dumps(g.session_id), httponly=True,
                samesite='Strict', secure=request.is_secure,
                max_age=config.session_ttl_seconds,
            )
        return response

    def rotate_session() -> None:
        g.session_id = uuid4().hex
        g.issue_session_cookie = True

    @app.get('/')
    def home():
        return render_template('index.html', upload_max_files=config.upload_max_files,
                               upload_max_file_mb=config.upload_max_file_bytes // (1024 * 1024),
                               upload_max_total_mb=config.upload_max_total_bytes // (1024 * 1024),
                               suggested_questions=SUGGESTED_QUESTIONS)

    @app.get('/api/health')
    def health():
        return jsonify(configuration_ready=True, snapshot_ready=True,
                       embedding_ready=True, llm_ready=True)

    @app.post('/api/chat/jobs')
    def create_job():
        _require_same_origin()
        question = (request.form.get('question') or '').strip()
        if not question:
            return jsonify(error_code='invalid_question', message='問題不可空白'), 400
        job_id = uuid4().hex
        try:
            attachments = _save_attachments(
                request.files.getlist('attachments'), upload_root / g.session_id / job_id,
                config,
            )
        except UploadValidationError as error:
            return jsonify(error_code='invalid_attachment', message=str(error)), 400
        job = jobs.submit(g.session_id, question, attachments, job_id=job_id)
        return jsonify(job_id=job.job_id, status=job.status), 202

    @app.get('/api/chat/jobs/active')
    def active_jobs():
        return jsonify(jobs=jobs.active(g.session_id))

    @app.get('/api/chat/jobs/<job_id>/events')
    def job_events(job_id: str):
        job = jobs.get(job_id, g.session_id)
        if job is None:
            return jsonify(error_code='job_not_found', message='工作不存在'), 404
        try:
            after = int(request.headers.get('Last-Event-ID', '0'))
        except ValueError:
            after = 0

        def stream():
            for event in jobs.events(job, after=after):
                if event is None:
                    yield ': heartbeat\n\n'
                    continue
                payload = json.dumps(event, ensure_ascii=False)
                yield 'id: {}\nevent: {}\ndata: {}\n\n'.format(
                    event['id'], event['event'], payload
                )

        return Response(stream(), mimetype='text/event-stream', headers={
            'Cache-Control': 'no-cache, no-transform',
            'X-Accel-Buffering': 'no',
        })

    @app.get('/api/history')
    def history():
        return jsonify(messages=sessions.history(g.session_id))

    @app.post('/api/chat/suggestions')
    def select_suggestion():
        _require_same_origin()
        payload = request.get_json(silent=True) or {}
        question = payload.get('question')
        if question not in SUGGESTED_QUESTIONS:
            return jsonify(error_code='invalid_suggestion', message='提示問題不存在'), 400
        sessions.set_pending_suggestion(g.session_id, question)
        return ('', 204)

    @app.post('/api/chat/qa/<qa_id>/actions')
    def qa_action(qa_id: str):
        _require_same_origin()
        payload = request.get_json(silent=True) or {}
        action = payload.get('action')
        feedback = payload.get('feedback')
        if action == 'copy':
            feedback = None
        elif action == 'feedback':
            if isinstance(feedback, bool) or feedback not in {-1, 0, 1}:
                return jsonify(error_code='invalid_feedback', message='Feedback 必須為 -1、0 或 1'), 400
        else:
            return jsonify(error_code='invalid_action', message='不支援的操作'), 400
        updated = sessions.update_qa_action(g.session_id, qa_id, action, feedback)
        if updated is None:
            return jsonify(error_code='qa_not_found', message='回答不存在'), 404
        logger.append(g.session_id, 'qa_action', qa_id=qa_id,
                      action=action, feedback=feedback)
        return jsonify(qa_id=qa_id, feedback=updated['feedback'],
                       actions=updated['actions'])

    @app.post('/api/history/report')
    def download_report():
        _require_same_origin()
        if jobs.active(g.session_id):
            return jsonify(error_code='job_active', message='回答進行中，暫時無法輸出'), 409
        snapshot = sessions.snapshot(g.session_id)
        qa_blocks = _qa_blocks(snapshot['messages'])
        if not qa_blocks:
            return jsonify(error_code='empty_history', message='目前沒有可輸出的對話'), 409
        logger.append(g.session_id, 'session_action', action='export_report')
        exported_at = datetime.now(timezone.utc)
        report = _markdown_report(
            session_id=g.session_id,
            created_at=str(snapshot['created_at_utc']),
            exported_at=exported_at.isoformat(),
            model=str(getattr(chat_backend, 'model', chat_backend.__class__.__name__)),
            document=f'{output_root.name}/{build_root.name}',
            qa_blocks=qa_blocks,
        )
        filename = f'cmp_chat_{exported_at.strftime("%Y%m%d_%H%M%S")}.md'
        return Response(report, mimetype='text/markdown; charset=utf-8', headers={
            'Content-Disposition': f'attachment; filename="{filename}"',
        })

    @app.delete('/api/history')
    def clear_history():
        _require_same_origin()
        jobs.cancel_session(g.session_id)
        logger.append(g.session_id, 'session_action', action='clear_chat')
        sessions.clear(g.session_id)
        logger.end(g.session_id, 'clear_chat')
        rotate_session()
        return ('', 204)

    @app.post('/api/session/end')
    def end_session():
        _require_same_origin()
        jobs.cancel_session(g.session_id)
        logger.append(g.session_id, 'session_action', action='end_chat')
        sessions.clear(g.session_id)
        logger.end(g.session_id, 'user_ended')
        rotate_session()
        return ('', 204)

    @app.get('/api/crops/<record_id>')
    def crop(record_id: str):
        record = artifact_records.get(record_id)
        if record is None or record.source.artifact_path is None:
            return jsonify(error_code='crop_not_found', message='圖片不存在'), 404
        candidate = (index.output_root / record.source.artifact_path).resolve()
        crops_root = (index.output_root / 'crops').resolve()
        if crops_root not in candidate.parents or not candidate.is_file():
            return jsonify(error_code='crop_not_found', message='圖片不存在'), 404
        return send_file(candidate, conditional=True)

    @app.errorhandler(UploadValidationError)
    def invalid_upload(error):
        return jsonify(error_code='invalid_attachment', message=str(error)), 400

    return app


def _qa_blocks(messages) -> list[dict[str, object]]:
    blocks = []
    attachments = []
    for message in messages:
        if message.get('role') == 'user':
            attachments = message.get('attachments', [])
            continue
        if message.get('role') != 'assistant':
            continue
        for item in message.get('items', []):
            blocks.append({**item, 'attachments': attachments})
    return blocks


def _markdown_report(*, session_id: str, created_at: str, exported_at: str,
                     model: str, document: str, qa_blocks) -> str:
    feedback_counts = {
        'like': sum(item.get('feedback') == 1 for item in qa_blocks),
        'dislike': sum(item.get('feedback') == -1 for item in qa_blocks),
        'none': sum(item.get('feedback', 0) == 0 for item in qa_blocks),
    }
    lines = [
        '---',
        'report_version: 1',
        f'session_id: {json.dumps(session_id, ensure_ascii=False)}',
        f'created_at: {json.dumps(created_at, ensure_ascii=False)}',
        f'exported_at: {json.dumps(exported_at, ensure_ascii=False)}',
        f'model: {json.dumps(model, ensure_ascii=False)}',
        f'document: {json.dumps(document, ensure_ascii=False)}',
        f'question_count: {len(qa_blocks)}',
        'feedback_summary:',
        f'  like: {feedback_counts["like"]}',
        f'  dislike: {feedback_counts["dislike"]}',
        f'  none: {feedback_counts["none"]}',
        '---',
        '',
        '# CMP 化學機械研磨機操作規範－對話報告',
        '',
    ]
    route_names = {
        'security_request': '安全限制',
        'out_of_scope': '超出文件範圍',
        'document_question': '正常查詢',
    }
    feedback_names = {-1: '讚', 0: '無回應', 1: '差'}
    for ordinal, item in enumerate(qa_blocks, start=1):
        route = item.get('route', 'document_question')
        category = '文件資訊不足' if item.get('insufficient_context') else route_names.get(route, route)
        lines.extend([
            f'## 對話 {ordinal}',
            '',
            '### 使用者問題',
            '',
            str(item.get('question', '')),
            '',
            '### 系統回答',
            '',
            str(item.get('answer', '')),
            '',
            f'- 回答類型：{category}',
            f'- Feedback：{feedback_names.get(item.get("feedback", 0), "無回應")}',
        ])
        attachments = item.get('attachments', [])
        if attachments:
            summary = '、'.join(
                f'{entry.get("name")}（{entry.get("media_type")}，'
                f'{entry.get("width")}x{entry.get("height")}）'
                for entry in attachments
            )
            lines.append(f'- 附件：{summary}')
        citations = item.get('citations', [])
        if citations:
            lines.append('- 參考資料：')
            for citation in citations:
                if citation.get('source_type') == 'attachment':
                    lines.append(f'  - 附件：{citation.get("name")}')
                else:
                    start = citation.get('page_start')
                    end = citation.get('page_end')
                    pages = f'第 {start} 頁' if start == end else f'第 {start}~{end} 頁'
                    lines.append(f'  - {citation.get("document_name")}，{pages}')
        lines.append('')
    return '\n'.join(lines)


def _require_same_origin() -> None:
    origin = request.headers.get('Origin')
    if origin and origin.rstrip('/') != request.host_url.rstrip('/'):
        abort(403)


def _save_attachments(files, destination: Path, config) -> list[AttachmentRef]:
    files = [item for item in files if item and item.filename]
    if len(files) > config.upload_max_files:
        raise UploadValidationError(f'附件最多 {config.upload_max_files} 張')
    if not files:
        return []
    destination.mkdir(parents=True, exist_ok=False)
    total = 0
    output = []
    try:
        for uploaded in files:
            data = uploaded.stream.read(config.upload_max_file_bytes + 1)
            if not data or len(data) > config.upload_max_file_bytes:
                raise UploadValidationError('單張附件大小超過限制')
            total += len(data)
            if total > config.upload_max_total_bytes:
                raise UploadValidationError('附件總大小超過限制')
            try:
                with Image.open(io.BytesIO(data)) as image:
                    image_format = str(image.format).upper()
                    width, height = image.size
                    if image_format not in SUPPORTED_IMAGES:
                        raise UploadValidationError('僅接受 JPEG、PNG 或 WebP')
                    if width * height > config.upload_max_pixels:
                        raise UploadValidationError('附件像素尺寸超過限制')
                    image.verify()
            except (UnidentifiedImageError, OSError) as error:
                raise UploadValidationError('附件不是有效圖片') from error
            media_type, suffix = SUPPORTED_IMAGES[image_format]
            path = destination / f'{uuid4().hex}{suffix}'
            path.write_bytes(data)
            metadata = AttachmentMetadata(
                name=Path(uploaded.filename).name,
                media_type=media_type,
                size_bytes=len(data), width=width, height=height,
            )
            output.append(AttachmentRef(metadata=metadata, path=path))
        return output
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def _cleanup_stale_uploads(root: Path, retention_seconds: int) -> None:
    if not root.is_dir():
        return
    cutoff = time.time() - retention_seconds
    resolved_root = root.resolve()
    for session_dir in root.iterdir():
        if not session_dir.is_dir():
            continue
        for job_dir in session_dir.iterdir():
            resolved = job_dir.resolve()
            if (job_dir.is_dir() and resolved_root in resolved.parents
                    and job_dir.stat().st_mtime < cutoff):
                shutil.rmtree(job_dir, ignore_errors=True)
        try:
            session_dir.rmdir()
        except OSError:
            pass
