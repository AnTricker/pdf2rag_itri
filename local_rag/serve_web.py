from __future__ import annotations

import io
import hmac
import json
import shutil
import threading
import time
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from uuid import uuid4

from flask import Flask, Response, abort, g, jsonify, redirect, render_template, request, send_file, url_for
from itsdangerous import BadSignature, SignatureExpired, URLSafeSerializer, URLSafeTimedSerializer
from PIL import Image, UnidentifiedImageError

from .index import FileVectorIndex
from .jobs import ChatJobManager
from .models import AttachmentMetadata
from .monitoring import MonitoringStore
from .rendering import SafeMarkdownRenderer
from .serve_chat import AttachmentRef, ServeChatEngine, SessionStore
from .session_log import SessionJsonlLogger


COOKIE_NAME = 'local_rag_session'
ADMIN_COOKIE_NAME = 'local_rag_admin'
ADMIN_SESSION_SECONDS = 15 * 60
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
    if not config.admin_password:
        raise ValueError('LOCAL_RAG_ADMIN_PASSWORD is required for serve')
    if config.monitoring_db_path is None or config.access_log_root is None:
        raise ValueError('monitoring paths are required for serve')
    app = Flask(__name__, template_folder=str(config.base_dir / 'templates'),
                static_folder=str(config.base_dir / 'static'))
    access_signer = URLSafeSerializer(config.session_secret, salt='local-rag-access-v1')
    chat_signer = URLSafeSerializer(config.session_secret, salt='local-rag-chat-v1')
    stream_signer = URLSafeTimedSerializer(config.session_secret, salt='local-rag-stream-v1')
    admin_signer = URLSafeTimedSerializer(config.session_secret, salt='local-rag-admin-v1')
    monitoring = MonitoringStore(config.monitoring_db_path, config.access_log_root)
    index = FileVectorIndex.load(build_root, output_root=output_root)
    logger = SessionJsonlLogger(config.session_log_root,
                               enabled=config.session_log_enabled)
    sessions = SessionStore(
        config.session_ttl_seconds,
        on_expire=lambda session_id: finish_chat(session_id, 'ttl_expired'),
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
    app.extensions['monitoring_store'] = monitoring
    active_chats: dict[str, str] = {}
    materialized_chats: set[str] = set()
    active_chats_lock = threading.RLock()

    def finish_chat(chat_session_id: str, reason: str,
                    *, action: str | None = None) -> None:
        jobs.cancel_session(chat_session_id)
        with active_chats_lock:
            has_chat = chat_session_id in materialized_chats
        if action and has_chat:
            logger.append(chat_session_id, 'session_action', action=action)
        sessions.clear(chat_session_id)
        if has_chat:
            logger.end(chat_session_id, reason)
        try:
            monitoring.end_chat(chat_session_id, reason)
        except Exception:
            app.logger.exception('failed to finish monitoring chat session')
        with active_chats_lock:
            active_chats.pop(chat_session_id, None)
            materialized_chats.discard(chat_session_id)

    def finish_access(access_session_id: str, reason: str) -> None:
        with active_chats_lock:
            chat_ids = [chat_id for chat_id, access_id in active_chats.items()
                        if access_id == access_session_id]
        for chat_id in chat_ids:
            finish_chat(chat_id, reason)
        try:
            monitoring.end_access(access_session_id, reason)
        except Exception:
            app.logger.exception('failed to finish monitoring access session')

    def new_chat_session() -> tuple[str, str]:
        chat_session_id = uuid4().hex
        with active_chats_lock:
            active_chats[chat_session_id] = g.access_session_id
        try:
            monitoring.add_session(
                g.access_session_id, chat_session_id, g.ip_addr
            )
        except Exception:
            app.logger.exception('failed to create monitoring chat session')
        token = chat_signer.dumps({
            'access_session_id': g.access_session_id,
            'chat_session_id': chat_session_id,
        })
        return chat_session_id, token

    def materialize_chat(chat_session_id: str) -> None:
        with active_chats_lock:
            if chat_session_id in materialized_chats:
                return
            materialized_chats.add(chat_session_id)
        pretty_path = logger.reserve(chat_session_id)
        try:
            monitoring.materialize_chat(chat_session_id, pretty_path)
        except Exception:
            app.logger.exception('failed to materialize monitoring chat session')

    def decode_chat_token(token: str):
        try:
            payload = chat_signer.loads(token)
        except BadSignature:
            return None
        if not isinstance(payload, dict):
            return None
        access_session_id = payload.get('access_session_id')
        chat_session_id = payload.get('chat_session_id')
        if not isinstance(access_session_id, str) or not isinstance(chat_session_id, str):
            return None
        return access_session_id, chat_session_id

    def chat_required(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            decoded = decode_chat_token(request.headers.get('X-Chat-Session', ''))
            if decoded is None:
                return jsonify(error_code='invalid_chat_session',
                               message='無效的對話 session'), 401
            access_session_id, chat_session_id = decoded
            if access_session_id != g.access_session_id:
                return jsonify(error_code='access_session_changed',
                               message='連線環境已改變，請建立新對話'), 409
            with active_chats_lock:
                active_access_id = active_chats.get(chat_session_id)
            if active_access_id != g.access_session_id:
                return jsonify(error_code='chat_session_ended',
                               message='對話已結束，請建立新對話'), 409
            g.chat_session_id = chat_session_id
            return function(*args, **kwargs)
        return wrapped

    @app.before_request
    def identify_access_session():
        g.request_started = time.perf_counter()
        g.request_id = uuid4().hex
        g.ip_addr = request.remote_addr or ''
        token = request.cookies.get(COOKIE_NAME)
        try:
            access_session_id = access_signer.loads(token) if token else None
        except BadSignature:
            access_session_id = None
        if not isinstance(access_session_id, str) or len(access_session_id) != 32:
            access_session_id = None
        row = None
        if access_session_id:
            try:
                row = monitoring.access_row(access_session_id)
            except Exception:
                app.logger.exception('failed to read monitoring access session')
        if row and (row.get('ended_at_utc') or row.get('ip_addr') != g.ip_addr):
            if not row.get('ended_at_utc'):
                finish_access(access_session_id, 'network_changed')
            access_session_id = None
            g.access_session_changed = True
        if access_session_id is None:
            access_session_id = uuid4().hex
            g.issue_session_cookie = True
        g.access_session_id = access_session_id
        try:
            if monitoring.access_row(access_session_id) is None:
                monitoring.create_access(access_session_id, g.ip_addr)
        except Exception:
            app.logger.exception('failed to create monitoring access session')

    @app.after_request
    def record_access(response):
        if getattr(g, 'issue_session_cookie', False):
            response.set_cookie(
                COOKIE_NAME, access_signer.dumps(g.access_session_id), httponly=True,
                samesite='Strict', secure=request.is_secure,
                max_age=config.session_ttl_seconds,
            )
        response.headers['X-Request-ID'] = g.request_id
        event = {
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'access_session_id': g.access_session_id,
            'chat_session_id': getattr(g, 'chat_session_id', None),
            'ip_addr': g.ip_addr,
            'method': request.method,
            'path': request.path,
            'status_code': response.status_code,
            'duration_ms': round((time.perf_counter() - g.request_started) * 1000, 3),
            'user_agent': request.user_agent.string,
            'request_id': g.request_id,
        }
        try:
            monitoring.append_access(g.access_session_id, event)
        except Exception:
            app.logger.exception('failed to append access log')
        return response

    @app.get('/')
    def home():
        return render_template('index.html', upload_max_files=config.upload_max_files,
                               upload_max_file_mb=config.upload_max_file_bytes // (1024 * 1024),
                               upload_max_total_mb=config.upload_max_total_bytes // (1024 * 1024),
                               suggested_questions=SUGGESTED_QUESTIONS)

    @app.get('/api/health')
    def health():
        llm_service_ready, llm_loaded = chat_backend.model_status()
        return jsonify(configuration_ready=True, snapshot_ready=True,
                       embedding_ready=True, llm_ready=llm_service_ready,
                       llm_service_ready=llm_service_ready,
                       llm_loaded=llm_loaded)

    @app.post('/api/chat/session/start')
    def start_chat_session():
        _require_same_origin()
        payload = request.get_json(silent=True) or {}
        previous = payload.get('previous_chat_token')
        decoded = decode_chat_token(previous) if isinstance(previous, str) else None
        if decoded is not None:
            previous_access_id, previous_chat_id = decoded
            with active_chats_lock:
                is_active = active_chats.get(previous_chat_id) == previous_access_id
            if is_active:
                finish_chat(previous_chat_id, 'page_reloaded')
            else:
                try:
                    monitoring.end_chat(previous_chat_id, 'page_reloaded')
                except Exception:
                    app.logger.exception('failed to close previous monitoring chat')
        chat_session_id, chat_token = new_chat_session()
        g.chat_session_id = chat_session_id
        return jsonify(chat_session_id=chat_session_id, chat_token=chat_token)

    @app.post('/api/chat/jobs')
    @chat_required
    def create_job():
        _require_same_origin()
        question = (request.form.get('question') or '').strip()
        if not question:
            return jsonify(error_code='invalid_question', message='問題不可空白'), 400
        job_id = uuid4().hex
        try:
            attachments = _save_attachments(
                request.files.getlist('attachments'), upload_root / g.chat_session_id / job_id,
                config,
            )
        except UploadValidationError as error:
            return jsonify(error_code='invalid_attachment', message=str(error)), 400
        materialize_chat(g.chat_session_id)
        job = jobs.submit(g.chat_session_id, question, attachments, job_id=job_id)
        stream_token = stream_signer.dumps({
            'access_session_id': g.access_session_id,
            'chat_session_id': g.chat_session_id,
            'job_id': job.job_id,
        })
        return jsonify(job_id=job.job_id, status=job.status,
                       stream_token=stream_token), 202

    @app.get('/api/chat/jobs/active')
    @chat_required
    def active_jobs():
        return jsonify(jobs=jobs.active(g.chat_session_id))

    @app.get('/api/chat/jobs/<job_id>/events')
    def job_events(job_id: str):
        try:
            stream_payload = stream_signer.loads(
                request.args.get('token', ''), max_age=config.job_retention_seconds
            )
        except (BadSignature, SignatureExpired):
            return jsonify(error_code='invalid_stream_token', message='無效的進度 token'), 401
        if not isinstance(stream_payload, dict) or stream_payload.get('job_id') != job_id:
            return jsonify(error_code='invalid_stream_token', message='無效的進度 token'), 401
        if stream_payload.get('access_session_id') != g.access_session_id:
            return jsonify(error_code='access_session_changed',
                           message='連線環境已改變，請建立新對話'), 409
        chat_session_id = stream_payload.get('chat_session_id')
        if not isinstance(chat_session_id, str):
            return jsonify(error_code='invalid_stream_token', message='無效的進度 token'), 401
        g.chat_session_id = chat_session_id
        job = jobs.get(job_id, chat_session_id)
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
    @chat_required
    def history():
        return jsonify(messages=sessions.history(g.chat_session_id))

    @app.post('/api/chat/suggestions')
    @chat_required
    def select_suggestion():
        _require_same_origin()
        payload = request.get_json(silent=True) or {}
        question = payload.get('question')
        if question not in SUGGESTED_QUESTIONS:
            return jsonify(error_code='invalid_suggestion', message='提示問題不存在'), 400
        sessions.set_pending_suggestion(g.chat_session_id, question)
        return ('', 204)

    @app.post('/api/chat/qa/<qa_id>/actions')
    @chat_required
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
        updated = sessions.update_qa_action(g.chat_session_id, qa_id, action, feedback)
        if updated is None:
            return jsonify(error_code='qa_not_found', message='回答不存在'), 404
        logger.append(g.chat_session_id, 'qa_action', qa_id=qa_id,
                      action=action, feedback=feedback)
        return jsonify(qa_id=qa_id, feedback=updated['feedback'],
                       actions=updated['actions'])

    @app.post('/api/history/report')
    @chat_required
    def download_report():
        _require_same_origin()
        if jobs.active(g.chat_session_id):
            return jsonify(error_code='job_active', message='回答進行中，暫時無法輸出'), 409
        snapshot = sessions.snapshot(g.chat_session_id)
        qa_blocks = _qa_blocks(snapshot['messages'])
        if not qa_blocks:
            return jsonify(error_code='empty_history', message='目前沒有可輸出的對話'), 409
        logger.append(g.chat_session_id, 'session_action', action='export_report')
        exported_at = datetime.now(timezone.utc)
        report = _markdown_report(
            session_id=g.chat_session_id,
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
    @chat_required
    def clear_history():
        _require_same_origin()
        finish_chat(g.chat_session_id, 'clear_chat', action='clear_chat')
        chat_session_id, chat_token = new_chat_session()
        g.chat_session_id = chat_session_id
        return jsonify(chat_session_id=chat_session_id, chat_token=chat_token)

    @app.post('/api/session/end')
    @chat_required
    def end_session():
        _require_same_origin()
        finish_chat(g.chat_session_id, 'user_ended', action='end_chat')
        return ('', 204)

    def local_admin_only() -> None:
        if request.remote_addr not in {'127.0.0.1', '::1'}:
            abort(403)

    def admin_authenticated() -> bool:
        token = request.cookies.get(ADMIN_COOKIE_NAME, '')
        try:
            return admin_signer.loads(token, max_age=ADMIN_SESSION_SECONDS) == 'admin'
        except (BadSignature, SignatureExpired):
            return False

    def admin_required(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            local_admin_only()
            if not admin_authenticated():
                return redirect(url_for('admin_login'))
            return function(*args, **kwargs)
        return wrapped

    @app.route('/admin/login', methods=['GET', 'POST'])
    def admin_login():
        local_admin_only()
        error = None
        if request.method == 'POST':
            _require_same_origin()
            candidate = request.form.get('password', '')
            if hmac.compare_digest(candidate, config.admin_password):
                response = redirect(url_for('admin_monitoring'))
                response.set_cookie(
                    ADMIN_COOKIE_NAME, admin_signer.dumps('admin'), httponly=True,
                    samesite='Strict', secure=request.is_secure,
                    max_age=ADMIN_SESSION_SECONDS,
                )
                return response
            error = '密碼錯誤'
        return render_template('admin_login.html', error=error)

    @app.post('/admin/logout')
    @admin_required
    def admin_logout():
        response = redirect(url_for('admin_login'))
        response.delete_cookie(ADMIN_COOKIE_NAME)
        return response

    @app.get('/admin/monitoring')
    @admin_required
    def admin_monitoring():
        filters = {
            'ip_addr': request.args.get('ip_addr', '').strip(),
            'access_session_id': request.args.get('access_session_id', '').strip(),
            'chat_session_id': request.args.get('chat_session_id', '').strip(),
            'from_utc': request.args.get('from_utc', '').strip(),
            'to_utc': request.args.get('to_utc', '').strip(),
        }
        rows = monitoring.query(**filters)
        return render_template('admin_monitoring.html', rows=rows, filters=filters,
                               events=None, history=None, detail_title=None)

    @app.get('/admin/monitoring/access/<access_session_id>')
    @admin_required
    def admin_access_detail(access_session_id: str):
        page = max(1, request.args.get('page', 1, type=int))
        events = monitoring.access_events(access_session_id, page=page)
        show_noise = request.args.get('show_noise') == '1'
        if not show_noise:
            events = [event for event in events
                      if not event.get('path', '').startswith('/static/')
                      and event.get('path') != '/api/health']
        return render_template(
            'admin_monitoring.html', rows=None, filters={}, events=events,
            history=None, detail_title=f'Access session: {access_session_id}',
            show_noise=show_noise, page=page,
        )

    @app.get('/admin/monitoring/chat/<chat_session_id>')
    @admin_required
    def admin_chat_detail(chat_session_id: str):
        history = monitoring.chat_history(chat_session_id)
        return render_template(
            'admin_monitoring.html', rows=None, filters={}, events=None,
            history=history, detail_title=f'Chat session: {chat_session_id}',
        )

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
