from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Union
from uuid import uuid4

import numpy as np

from .models import BuildReport, CorpusProfile, KnowledgeRecord, SnapshotManifest


class BuildWriter:
    def __init__(self, output_root: Path, review_version: int) -> None:
        self.output_root = output_root.resolve()
        self.review_version = review_version

    def write(
        self,
        *,
        records: list[KnowledgeRecord],
        embeddings: np.ndarray,
        manifest: SnapshotManifest,
        report: BuildReport,
        corpus_profile: CorpusProfile,
    ) -> Path:
        builds_root = self.output_root / "builds"
        builds_root.mkdir(parents=True, exist_ok=True)
        destination = builds_root / f"{self.review_version:03d}"
        if destination.exists():
            raise FileExistsError(f"build already exists: {destination}")

        stage = builds_root / f".stage-{uuid4().hex}"
        stage.mkdir()
        try:
            self._write_jsonl(stage / "records.jsonl", records)
            self._write_json(stage / "corpus_profile.json", corpus_profile)
            np.save(stage / "embeddings.npy", embeddings, allow_pickle=False)
            records_sha256 = self._sha256(stage / "records.jsonl")
            embeddings_sha256 = self._sha256(stage / "embeddings.npy")
            artifact_sha256 = self._referenced_crop_hashes(records)
            snapshot_id = hashlib.sha256(
                f"{manifest.document_sha256}:{records_sha256}:{embeddings_sha256}".encode(
                    "ascii"
                )
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
            self._validate(stage, self.output_root)
            stage.replace(destination)
            return destination
        except Exception:
            shutil.rmtree(stage, ignore_errors=True)
            raise

    def _referenced_crop_hashes(
        self, records: list[KnowledgeRecord]
    ) -> dict[str, str]:
        hashes: dict[str, str] = {}
        for record in records:
            relative_path = record.source.artifact_path
            if not relative_path:
                continue
            artifact = self._resolve_crop(self.output_root, relative_path)
            hashes[relative_path] = self._sha256(artifact)
        return dict(sorted(hashes.items()))

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

    @classmethod
    def _validate(cls, build_root: Path, output_root: Path) -> None:
        record_lines = (build_root / "records.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        vectors = np.load(build_root / "embeddings.npy", allow_pickle=False)
        manifest = SnapshotManifest.model_validate_json(
            (build_root / "manifest.json").read_text(encoding="utf-8")
        )
        corpus_profile = CorpusProfile.model_validate_json(
            (build_root / "corpus_profile.json").read_text(encoding="utf-8")
        )
        if corpus_profile.document_id != manifest.document_sha256:
            raise ValueError("corpus profile document identifier mismatch")
        if vectors.dtype != np.float32 or vectors.ndim != 2:
            raise ValueError("invalid embedding matrix")
        expected = (manifest.record_count, manifest.vector_dimension)
        if vectors.shape != expected or len(record_lines) != manifest.record_count:
            raise ValueError("build record/vector counts do not match manifest")
        if not np.isfinite(vectors).all():
            raise ValueError("embeddings.npy contains non-finite values")
        if cls._sha256(build_root / "records.jsonl") != manifest.records_sha256:
            raise ValueError("records checksum mismatch")
        if cls._sha256(build_root / "embeddings.npy") != manifest.embeddings_sha256:
            raise ValueError("embeddings checksum mismatch")
        for relative_path, expected_hash in manifest.artifact_sha256.items():
            artifact = cls._resolve_crop(output_root, relative_path)
            if cls._sha256(artifact) != expected_hash:
                raise ValueError("artifact checksum mismatch")
        for line in record_lines:
            record = KnowledgeRecord.model_validate_json(line)
            if record.document_id != manifest.document_sha256:
                raise ValueError("record document identifier mismatch")
            if record.source.artifact_path:
                cls._resolve_crop(output_root, record.source.artifact_path)

    @staticmethod
    def _resolve_crop(output_root: Path, relative_path: str) -> Path:
        normalized = Path(relative_path)
        if normalized.is_absolute() or normalized.parts[:1] != ("crops",):
            raise ValueError("crop artifact path must be relative to crops/")
        candidate = (output_root / normalized).resolve()
        crops_root = (output_root / "crops").resolve()
        if crops_root not in candidate.parents or not candidate.is_file():
            raise ValueError("record references an invalid crop artifact")
        return candidate

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()