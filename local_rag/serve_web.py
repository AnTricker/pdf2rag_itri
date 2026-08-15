from __future__ import annotations

import io
import json
import shutil
import time
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
                               upload_max_total_mb=config.upload_max_total_bytes // (1024 * 1024))

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

    @app.delete('/api/history')
    def clear_history():
        _require_same_origin()
        jobs.cancel_session(g.session_id)
        sessions.clear(g.session_id)
        logger.end(g.session_id, 'clear_chat')
        rotate_session()
        return ('', 204)

    @app.post('/api/session/end')
    def end_session():
        _require_same_origin()
        jobs.cancel_session(g.session_id)
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
