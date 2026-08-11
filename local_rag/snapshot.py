from __future__ import annotations

import json
import hashlib
import shutil
from pathlib import Path
from typing import Optional, Union
from uuid import uuid4

import numpy as np

from .models import BuildReport, CorpusProfile, KnowledgeRecord, SnapshotManifest


class SnapshotWriter:
    def __init__(self, index_root: Path) -> None:
        self.index_root = index_root

    def write(
        self,
        *,
        source_pdf: Path,
        records: list[KnowledgeRecord],
        embeddings: np.ndarray,
        manifest: SnapshotManifest,
        report: BuildReport,
        crop_artifacts: Optional[list[Path]] = None,
        corpus_profile: CorpusProfile,
    ) -> Path:
        self.index_root.mkdir(parents=True, exist_ok=True)
        stage = self.index_root / f".stage-{uuid4().hex}"
        stage.mkdir()
        try:
            document_dir = stage / "document"
            document_dir.mkdir()
            shutil.copy2(source_pdf, document_dir / source_pdf.name)
            if crop_artifacts:
                crop_dir = stage / "crops"
                crop_dir.mkdir()
                for crop in crop_artifacts:
                    shutil.copy2(crop, crop_dir / crop.name)
            self._write_jsonl(stage / "records.jsonl", records)
            self._write_pretty(stage / "records.pretty.json", records)
            self._write_json(stage / "corpus_profile.json", corpus_profile)
            np.save(stage / "embeddings.npy", embeddings, allow_pickle=False)
            records_sha256 = self._sha256(stage / "records.jsonl")
            embeddings_sha256 = self._sha256(stage / "embeddings.npy")
            artifact_sha256 = {
                path.relative_to(stage).as_posix(): self._sha256(path)
                for path in sorted(stage.rglob("*"))
                if path.is_file()
                and (
                    path.parent.name in {"document", "crops"}
                    or path.name in {"records.pretty.json", "corpus_profile.json"}
                )
            }
            snapshot_id = hashlib.sha256(
                f"{manifest.document_sha256}:{records_sha256}:{embeddings_sha256}".encode("ascii")
            ).hexdigest()[:24]
            completed_manifest = manifest.model_copy(
                update={
                    "snapshot_id": snapshot_id,
                    "records_sha256": records_sha256,
                    "embeddings_sha256": embeddings_sha256,
                    "artifact_sha256": artifact_sha256,
                }
            )
            completed_report = report.model_copy(update={"snapshot_id": snapshot_id})
            self._write_json(stage / "manifest.json", completed_manifest)
            self._write_json(stage / "build_report.json", completed_report)
            self._validate(stage)
            return self._promote(stage)
        except Exception:
            shutil.rmtree(stage, ignore_errors=True)
            raise

    @staticmethod
    def _write_jsonl(path: Path, records: list[KnowledgeRecord]) -> None:
        with path.open("w", encoding="utf-8", newline="\n") as output:
            for record in records:
                output.write(json.dumps(record.model_dump(), ensure_ascii=False))
                output.write("\n")

    @staticmethod
    def _write_json(
        path: Path, model: Union[SnapshotManifest, BuildReport, CorpusProfile]
    ) -> None:
        path.write_text(
            json.dumps(model.model_dump(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _write_pretty(path: Path, records: list[KnowledgeRecord]) -> None:
        output = []
        for record in records:
            source = record.source.model_dump(exclude={"document_sha256"})
            processing = record.processing.model_dump(exclude={"content_sha256"})
            output.append(
                {
                    "modality": record.modality,
                    "language": record.language,
                    "content": record.content,
                    "source": source,
                    "processing": processing,
                    "image": record.image.model_dump() if record.image else None,
                }
            )
        path.write_text(
            json.dumps(output, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _validate(stage: Path) -> None:
        record_lines = (stage / "records.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        vectors = np.load(stage / "embeddings.npy", allow_pickle=False)
        manifest = SnapshotManifest.model_validate_json(
            (stage / "manifest.json").read_text(encoding="utf-8")
        )
        corpus_profile = CorpusProfile.model_validate_json(
            (stage / "corpus_profile.json").read_text(encoding="utf-8")
        )
        if corpus_profile.document_id != manifest.document_sha256:
            raise ValueError("corpus profile document identifier mismatch")
        if vectors.dtype != np.float32:
            raise ValueError("embeddings.npy must contain float32 vectors")
        if vectors.ndim != 2:
            raise ValueError("embeddings.npy must be a two-dimensional matrix")
        expected = (manifest.record_count, manifest.vector_dimension)
        if vectors.shape != expected or len(record_lines) != manifest.record_count:
            raise ValueError("snapshot record/vector counts do not match manifest")
        if not np.isfinite(vectors).all():
            raise ValueError("embeddings.npy contains non-finite values")
        if SnapshotWriter._sha256(stage / "records.jsonl") != manifest.records_sha256:
            raise ValueError("records checksum mismatch")
        if SnapshotWriter._sha256(stage / "embeddings.npy") != manifest.embeddings_sha256:
            raise ValueError("embeddings checksum mismatch")
        documents = list((stage / "document").iterdir())
        if len(documents) != 1 or SnapshotWriter._sha256(documents[0]) != manifest.document_sha256:
            raise ValueError("document checksum mismatch")
        if corpus_profile.document_name != documents[0].name:
            raise ValueError("corpus profile document name mismatch")
        for relative_path, expected_hash in manifest.artifact_sha256.items():
            artifact = (stage / relative_path).resolve()
            if stage.resolve() not in artifact.parents or not artifact.is_file():
                raise ValueError("manifest references an invalid artifact path")
            if SnapshotWriter._sha256(artifact) != expected_hash:
                raise ValueError("artifact checksum mismatch")
        for line in record_lines:
            record = KnowledgeRecord.model_validate_json(line)
            if record.source.artifact_path:
                artifact = (stage / record.source.artifact_path).resolve()
                if stage.resolve() not in artifact.parents or not artifact.is_file():
                    raise ValueError("record references an invalid artifact path")

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _promote(self, stage: Path) -> Path:
        current = self.index_root / "current"
        backup = self.index_root / f".previous-{uuid4().hex}"
        if current.exists():
            current.replace(backup)
        try:
            stage.replace(current)
        except Exception:
            if backup.exists():
                backup.replace(current)
            raise
        shutil.rmtree(backup, ignore_errors=True)
        return current
