from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Optional

import numpy as np

from .models import CorpusProfile, KnowledgeRecord, SnapshotManifest


@dataclass(frozen=True)
class SearchHit:
    record: KnowledgeRecord
    score: float
    selection_reason: str = "semantic"


class FileVectorIndex:
    def __init__(
        self,
        *,
        snapshot_root: Path,
        output_root: Path,
        records: list[KnowledgeRecord],
        vectors: np.ndarray,
        manifest: SnapshotManifest,
        corpus_profile: CorpusProfile,
    ) -> None:
        self.snapshot_root = snapshot_root
        self.output_root = output_root
        self.records = records
        self.vectors = vectors
        self.manifest = manifest
        self.corpus_profile = corpus_profile
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self.normalized_vectors = np.divide(
            vectors,
            norms,
            out=np.zeros_like(vectors),
            where=norms != 0,
        )

    @classmethod
    def load(
        cls, snapshot_root: Path, *, output_root: Optional[Path] = None
    ) -> "FileVectorIndex":
        root = snapshot_root.resolve(strict=True)
        output = (
            output_root.resolve(strict=True)
            if output_root is not None
            else root.parents[1]
        )
        manifest = SnapshotManifest.model_validate_json(
            (root / "manifest.json").read_text(encoding="utf-8")
        )
        corpus_profile = CorpusProfile.model_validate_json(
            (root / "corpus_profile.json").read_text(encoding="utf-8")
        )
        records = [
            KnowledgeRecord.model_validate_json(line)
            for line in (root / "records.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        vectors = np.load(root / "embeddings.npy", allow_pickle=False)
        if cls._sha256(root / "records.jsonl") != manifest.records_sha256:
            raise ValueError("records checksum mismatch")
        if cls._sha256(root / "embeddings.npy") != manifest.embeddings_sha256:
            raise ValueError("embeddings checksum mismatch")

        crops_root = (output / "crops").resolve()
        for relative_path, expected_hash in manifest.artifact_sha256.items():
            artifact = (output / relative_path).resolve()
            if crops_root not in artifact.parents or not artifact.is_file():
                raise ValueError("manifest crop path is invalid")
            if cls._sha256(artifact) != expected_hash:
                raise ValueError("artifact checksum mismatch")
        if vectors.dtype != np.float32 or vectors.ndim != 2:
            raise ValueError("invalid embedding matrix")
        if not np.isfinite(vectors).all():
            raise ValueError("embedding matrix contains non-finite values")
        expected = (manifest.record_count, manifest.vector_dimension)
        if len(records) != manifest.record_count or vectors.shape != expected:
            raise ValueError("snapshot counts or dimensions do not match")
        return cls(
            snapshot_root=root,
            output_root=output,
            records=records,
            vectors=vectors,
            manifest=manifest,
            corpus_profile=corpus_profile,
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def search(
        self,
        query_vector: np.ndarray,
        *,
        top_k: int,
        modalities: Optional[set[str]] = None,
    ) -> list[SearchHit]:
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if query.shape != (self.manifest.vector_dimension,):
            raise ValueError("query vector dimension does not match snapshot")
        norm = float(np.linalg.norm(query))
        if not np.isfinite(query).all() or norm == 0:
            return []
        scores = self.normalized_vectors @ (query / norm)
        candidates = [
            index
            for index, record in enumerate(self.records)
            if modalities is None or record.modality in modalities
        ]
        order = sorted(candidates, key=lambda index: -float(scores[index]))[:top_k]
        return [
            SearchHit(record=self.records[index], score=float(scores[index]))
            for index in order
        ]

    def representative(
        self,
        query_vector: np.ndarray,
        *,
        top_k: int,
        modalities: Optional[set[str]] = None,
    ) -> list[SearchHit]:
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if query.shape != (self.manifest.vector_dimension,):
            raise ValueError("query vector dimension does not match snapshot")
        norm = float(np.linalg.norm(query))
        if not np.isfinite(query).all() or norm == 0:
            return []
        scores = self.normalized_vectors @ (query / norm)
        representatives: list[int] = []
        seen_groups: set[tuple[object, ...]] = set()
        for index, record in enumerate(self.records):
            if modalities is not None and record.modality not in modalities:
                continue
            group = (
                ("section", *record.source.section_path)
                if record.source.section_path
                else ("page", record.source.page_start)
            )
            if group not in seen_groups:
                seen_groups.add(group)
                representatives.append(index)
        if len(representatives) > top_k:
            if top_k == 1:
                selected = [representatives[0]]
            else:
                selected = [
                    representatives[
                        round(position * (len(representatives) - 1) / (top_k - 1))
                    ]
                    for position in range(top_k)
                ]
        else:
            selected = representatives
        return [
            SearchHit(
                record=self.records[index],
                score=float(scores[index]),
                selection_reason="representative",
            )
            for index in selected
        ]
