from __future__ import annotations

import queue
import shutil
import threading
import time
from dataclasses import dataclass, field
from typing import Optional
from uuid import uuid4

import requests

from .serve_chat import AttachmentRef, JobCancelled


TERMINAL_STATES = {'completed', 'failed', 'cancelled'}


@dataclass
class ChatJob:
    job_id: str
    session_id: str
    question: str
    attachments: list[AttachmentRef]
    created_at: float = field(default_factory=time.monotonic)
    updated_at: float = field(default_factory=time.monotonic)
    status: str = 'queued'
    cancelled: bool = False
    events: list[dict[str, object]] = field(default_factory=list)
    condition: threading.Condition = field(default_factory=threading.Condition)


class ChatJobManager:
    def __init__(self, engine, *, retention_seconds: int) -> None:
        self.engine = engine
        self.retention_seconds = retention_seconds
        self._jobs: dict[str, ChatJob] = {}
        self._lock = threading.RLock()
        self._queue: queue.Queue[Optional[str]] = queue.Queue()
        self._worker = threading.Thread(target=self._run, name='local-rag-gpu-worker', daemon=True)
        self._worker.start()

    def submit(self, session_id: str, question: str,
               attachments: list[AttachmentRef], *, job_id: Optional[str] = None) -> ChatJob:
        self._cleanup_expired()
        job = ChatJob(job_id=job_id or uuid4().hex, session_id=session_id,
                      question=question, attachments=attachments)
        with self._lock:
            self._jobs[job.job_id] = job
            position = self._queue.qsize() + 1
        self._publish(job, 'queued', message='等待模型', queue_position=position)
        self._queue.put(job.job_id)
        return job

    def get(self, job_id: str, session_id: str) -> Optional[ChatJob]:
        with self._lock:
            job = self._jobs.get(job_id)
            return job if job is not None and job.session_id == session_id else None

    def active(self, session_id: str) -> list[dict[str, object]]:
        with self._lock:
            return [
                {'job_id': job.job_id, 'status': job.status}
                for job in self._jobs.values()
                if job.session_id == session_id and job.status not in TERMINAL_STATES
            ]

    def events(self, job: ChatJob, after: int = 0):
        cursor = max(0, after)
        while True:
            with job.condition:
                if cursor >= len(job.events) and job.status not in TERMINAL_STATES:
                    job.condition.wait(timeout=15)
                pending = job.events[cursor:]
                terminal = job.status in TERMINAL_STATES
            if not pending:
                yield None
            for event in pending:
                cursor += 1
                yield event
            if terminal and cursor >= len(job.events):
                return

    def cancel_session(self, session_id: str) -> None:
        with self._lock:
            jobs = [job for job in self._jobs.values()
                    if job.session_id == session_id and job.status not in TERMINAL_STATES]
        for job in jobs:
            job.cancelled = True
            if job.status == 'queued':
                self._cleanup_attachments(job)
                job.status = 'cancelled'
                self._publish(job, 'cancelled', message='工作已取消')

    def shutdown(self) -> None:
        self._queue.put(None)
        self._worker.join(timeout=5)

    def _publish(self, job: ChatJob, event: str, **data) -> None:
        with job.condition:
            job.updated_at = time.monotonic()
            job.events.append({'id': len(job.events) + 1, 'event': event, **data})
            job.condition.notify_all()
        self.engine.session_logger.append(job.session_id, f'job_{event}', **data)

    def _run(self) -> None:
        while True:
            job_id = self._queue.get()
            if job_id is None:
                return
            with self._lock:
                job = self._jobs.get(job_id)
            if job is None:
                continue
            if job.cancelled:
                self._cleanup_attachments(job)
                continue
            job.status = 'running'
            self._publish(job, 'started', message='模型開始處理')
            try:
                result = self.engine.ask(
                    job.session_id,
                    job.question,
                    job.attachments,
                    progress=lambda event, **data: self._publish(job, event, **data),
                    cancelled=lambda: job.cancelled,
                )
                if job.cancelled:
                    raise JobCancelled('job was cancelled')
                self._cleanup_attachments(job)
                job.status = 'completed'
                self._publish(job, 'completed', message='回答完成', result=result)
            except JobCancelled:
                self._cleanup_attachments(job)
                job.status = 'cancelled'
                self._publish(job, 'cancelled', message='工作已取消')
            except requests.Timeout:
                self._cleanup_attachments(job)
                job.status = 'failed'
                self._publish(job, 'failed', error_code='model_timeout',
                              message='模型回應逾時，請稍後重試')
            except requests.RequestException:
                self._cleanup_attachments(job)
                job.status = 'failed'
                self._publish(job, 'failed', error_code='model_unavailable',
                              message='本機模型服務目前無法使用')
            except Exception as error:
                self._cleanup_attachments(job)
                job.status = 'failed'
                self._publish(job, 'failed', error_code='internal_error',
                              message='處理失敗，請稍後重試', detail=type(error).__name__)
            finally:
                self._cleanup_attachments(job)

    @staticmethod
    def _cleanup_attachments(job: ChatJob) -> None:
        roots = {item.path.parent for item in job.attachments}
        for root in roots:
            if root.name == job.job_id and root.is_dir():
                shutil.rmtree(root, ignore_errors=True)

    def _cleanup_expired(self) -> None:
        cutoff = time.monotonic() - self.retention_seconds
        with self._lock:
            expired = [job_id for job_id, job in self._jobs.items()
                       if job.status in TERMINAL_STATES and job.updated_at < cutoff]
            for job_id in expired:
                self._jobs.pop(job_id, None)
