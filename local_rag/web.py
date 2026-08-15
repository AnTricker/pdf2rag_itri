from __future__ import annotations

import re
from pathlib import Path

import requests
from flask import Flask, jsonify, render_template, request, send_file

from .chat import ChatEngine, ModelProcessingError, PreprocessingError, SessionStore
from .config import AppConfig
from .index import FileVectorIndex
from .rendering import SafeMarkdownRenderer
from .session_log import SessionJsonlLogger


SESSION_PATTERN = re.compile(r"^[A-Za-z0-9_-]{8,128}$")


def create_app(
    *,
    config: AppConfig,
    embedding_backend,
    chat_backend,
    build_root: Path,
    output_root: Path,
) -> Flask:
    app = Flask(
        __name__,
        template_folder=str(config.base_dir / "templates"),
        static_folder=str(config.base_dir / "static"),
    )
    index = FileVectorIndex.load(build_root, output_root=output_root)
    session_logger = SessionJsonlLogger(
        config.session_log_root,
        enabled=config.session_log_enabled,
    )
    sessions = SessionStore(
        config.session_ttl_seconds,
        on_expire=lambda session_id: session_logger.end(
            session_id, "ttl_expired"
        ),
    )
    engine = ChatEngine(
        config=config,
        index=index,
        embedding_backend=embedding_backend,
        chat_backend=chat_backend,
        sessions=sessions,
        session_logger=session_logger,
        renderer=SafeMarkdownRenderer(enabled=config.markdown_enabled),
    )
    artifact_records = {
        record.record_id: record
        for record in index.records
        if record.source.artifact_path
    }

    @app.get("/")
    def home():
        return render_template("index.html")

    @app.get("/api/health")
    def health():
        llm_ready = True
        health_method = getattr(chat_backend, "health", None)
        if callable(health_method):
            try:
                llm_ready = bool(health_method())
            except Exception:
                llm_ready = False
        status = 200 if llm_ready else 503
        return jsonify(
            configuration_ready=True,
            snapshot_ready=True,
            embedding_ready=True,
            llm_ready=llm_ready,
        ), status

    @app.post("/api/chat")
    def chat():
        body = request.get_json(silent=True) or {}
        session_id = body.get("session_id")
        question = body.get("question")
        if not isinstance(session_id, str) or not SESSION_PATTERN.fullmatch(session_id):
            return jsonify(error_code="invalid_session", message="無效的 session identifier"), 400
        if not isinstance(question, str) or not question.strip():
            return jsonify(error_code="invalid_question", message="問題不可空白"), 400
        return jsonify(engine.ask(session_id, question.strip()))

    @app.get("/api/history/<session_id>")
    def history(session_id: str):
        if not SESSION_PATTERN.fullmatch(session_id):
            return jsonify(error_code="invalid_session", message="無效的 session identifier"), 400
        return jsonify(messages=sessions.get(session_id).messages)

    @app.delete("/api/history/<session_id>")
    def clear_history(session_id: str):
        if not SESSION_PATTERN.fullmatch(session_id):
            return jsonify(error_code="invalid_session", message="無效的 session identifier"), 400
        sessions.clear(session_id)
        session_logger.end(session_id, "clear_chat")
        return ("", 204)

    @app.post("/api/session/<session_id>/end")
    def end_session(session_id: str):
        if not SESSION_PATTERN.fullmatch(session_id):
            return jsonify(error_code="invalid_session", message="無效的 session identifier"), 400
        sessions.clear(session_id)
        session_logger.end(session_id, "user_ended")
        return ("", 204)


    @app.get("/api/crops/<record_id>")
    def crop(record_id: str):
        record = artifact_records.get(record_id)
        if record is None or record.source.artifact_path is None:
            return jsonify(error_code="crop_not_found", message="圖片不存在"), 404
        candidate = (index.output_root / record.source.artifact_path).resolve()
        crops_root = (index.output_root / "crops").resolve()
        if crops_root not in candidate.parents or not candidate.is_file():
            return jsonify(error_code="crop_not_found", message="圖片不存在"), 404
        return send_file(candidate, conditional=True)

    @app.errorhandler(requests.Timeout)
    def model_timeout(_error):
        return jsonify(error_code="model_timeout", message="模型回應逾時，請稍後重試"), 504

    @app.errorhandler(requests.RequestException)
    def model_unavailable(_error):
        return jsonify(error_code="model_unavailable", message="本機模型服務目前無法使用"), 503

    @app.errorhandler(PreprocessingError)
    def preprocessing_failed(_error):
        return jsonify(
            error_code="admission_failed",
            message="無法完成問題安全分類，請稍後重試。",
        ), 422

    @app.errorhandler(ModelProcessingError)
    def answer_processing_failed(_error):
        return jsonify(
            error_code="answer_processing_failed",
            message="模型處理答案失敗，請稍後重試。",
        ), 502

    @app.errorhandler(500)
    def internal_error(_error):
        return jsonify(error_code="internal_error", message="處理失敗，請稍後重試"), 500

    return app


# Public compatibility import: serve now uses the job/SSE implementation.
from .serve_web import create_app as create_app
