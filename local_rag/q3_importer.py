from __future__ import annotations

from collections.abc import Sequence
import hashlib
import hmac
import json
from pathlib import Path
import re
import shutil
from typing import Any

import numpy as np

from .config import AppConfig
from .models import (
    BuildReport,
    CorpusProfile,
    EmbeddingMetadata,
    KnowledgeRecord,
    ProcessingMetadata,
    SnapshotManifest,
    SourceMetadata,
)
from .snapshot import BuildWriter


_OUTPUT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class Q3Importer:
    """Import precomputed Qwen3-VL knowledge bases into a serveable build."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def import_collections(
        self,
        sources: Sequence[Path],
        *,
        output_id: str,
        collection_name: str,
    ) -> Path:
        if not sources:
            raise ValueError("at least one Qwen3-VL knowledge base is required")
        if not _OUTPUT_ID.fullmatch(output_id):
            raise ValueError("output id may contain only letters, numbers, '.', '_' and '-'")
        if not collection_name.strip():
            raise ValueError("collection name cannot be blank")

        output_root = self.config.index_root / output_id
        if output_root.exists():
            raise FileExistsError(f"output already exists: {output_root}")

        knowledge_bases = [self._knowledge_base(path) for path in sources]
        loaded = [self._load_source(path) for path in knowledge_bases]
        model = self._shared_model(loaded)
        document_id = self._collection_id(loaded)

        output_root.mkdir(parents=True)
        try:
            records: list[KnowledgeRecord] = []
            vectors: list[np.ndarray] = []
            ignored_count = 0
            record_ids: set[str] = set()
            all_types: set[str] = set()
            all_headings: set[str] = set()

            for source_number, source in enumerate(loaded, start=1):
                source_tag = f"{source_number:03d}-{source['manifest_sha256'][:12]}"
                matrices = source["matrices"]
                for raw in source["records"]:
                    metadata = self._mapping(raw.get("metadata"), "record metadata")
                    all_types.update(self._strings(metadata.get("types")))
                    all_headings.update(self._strings(metadata.get("heading_path")))
                    vector_ref = raw.get("vector_ref")
                    if vector_ref is None:
                        ignored_count += 1
                        continue
                    vector_ref = self._mapping(vector_ref, "vector_ref")
                    kind = vector_ref.get("kind")
                    row = vector_ref.get("row")
                    if kind not in matrices or not isinstance(row, int):
                        raise ValueError(f"invalid vector reference for record {raw.get('id')}")
                    matrix = matrices[kind]
                    if row < 0 or row >= len(matrix):
                        raise ValueError(f"vector row is out of range for record {raw.get('id')}")

                    record_id = str(raw.get("id", "")).strip()
                    if not record_id or record_id in record_ids:
                        raise ValueError(f"missing or duplicate record id: {record_id!r}")
                    record_ids.add(record_id)
                    modality = "image" if kind == "image_vector" else "text"
                    artifact_path = None
                    if modality == "image":
                        artifact_path = self._copy_crop(
                            source["root"], metadata, output_root, source_tag
                        )
                    content = self._content(raw)
                    page_start, page_end = self._pages(metadata)
                    records.append(
                        KnowledgeRecord(
                            record_id=record_id,
                            document_id=document_id,
                            modality=modality,
                            content=content,
                            language=self._language(content),
                            source=SourceMetadata(
                                document_name=collection_name.strip(),
                                document_sha256=document_id,
                                page_start=page_start,
                                page_end=page_end,
                                artifact_path=artifact_path,
                                section_path=self._strings(metadata.get("heading_path")),
                            ),
                            processing=ProcessingMetadata(
                                extractor="qwen3vl-import-v1",
                                token_count=0,
                                content_sha256=self._sha256_bytes(content.encode("utf-8")),
                            ),
                        )
                    )
                    vectors.append(np.asarray(matrix[row], dtype=np.float32))

            if not records:
                raise ValueError("Qwen3-VL sources contain no searchable records")
            embeddings = np.stack(vectors).astype(np.float32, copy=False)
            dimension = int(model["dimension"])
            if embeddings.shape != (len(records), dimension):
                raise ValueError("imported record/vector dimensions do not match")
            if not np.isfinite(embeddings).all():
                raise ValueError("imported vectors contain non-finite values")

            profile = self._profile(
                collection_name.strip(),
                document_id,
                len(loaded),
                records,
                all_types,
                all_headings,
            )
            manifest = SnapshotManifest(
                document_sha256=document_id,
                record_count=len(records),
                vector_count=len(records),
                vector_dimension=dimension,
                embedding=EmbeddingMetadata(
                    backend="sentence_transformers_qwen3vl_import",
                    model=str(model["model_id"]),
                ),
                chunk_max_tokens=self.config.chunk_max_tokens,
                chunk_overlap_tokens=self.config.chunk_overlap_tokens,
                prompt_versions={"importer": "qwen3vl-import-v1"},
            )
            image_count = sum(record.modality == "image" for record in records)
            report = BuildReport(
                source_document=collection_name.strip(),
                document_id=document_id,
                text_record_count=len(records) - image_count,
                image_record_count=image_count,
                source_image_count=image_count,
                image_chunk_count=image_count,
                embedded_count=len(records),
                warnings=[f"ignored {ignored_count} provenance-only records"],
                stage_statuses={
                    "source_checksum_validation": "complete",
                    "record_conversion": "complete",
                    "vector_import": "complete",
                    "build_validation": "complete",
                },
                artifact_locations={
                    "records": "records.jsonl",
                    "corpus_profile": "corpus_profile.json",
                    "embeddings": "embeddings.npy",
                },
            )
            return BuildWriter(output_root, 1).write(
                records=records,
                embeddings=embeddings,
                manifest=manifest,
                report=report,
                corpus_profile=profile,
            )
        except Exception:
            shutil.rmtree(output_root, ignore_errors=True)
            raise

    @staticmethod
    def _knowledge_base(source: Path) -> Path:
        root = source.resolve(strict=True)
        candidate = root / "knowledge_base" if (root / "knowledge_base").is_dir() else root
        required = [
            candidate / "manifest.json",
            candidate / "records.jsonl",
            candidate / "vectors" / "text.npy",
            candidate / "vectors" / "image.npy",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"incomplete Qwen3-VL knowledge base: {missing}")
        return candidate.resolve()

    def _load_source(self, root: Path) -> dict[str, Any]:
        manifest_path = root / "manifest.json"
        manifest = self._mapping(
            json.loads(manifest_path.read_text(encoding="utf-8")), "manifest"
        )
        if manifest.get("schema_version") != "1.0" or manifest.get("mode") != "qwen3vl":
            raise ValueError(f"unsupported Qwen3-VL manifest: {manifest_path}")
        self._verify_source_files(root, manifest)
        model = self._mapping(manifest.get("model"), "manifest model")
        matrices = {
            "text_vector": np.load(root / "vectors" / "text.npy", allow_pickle=False),
            "image_vector": np.load(root / "vectors" / "image.npy", allow_pickle=False),
        }
        dimension = model.get("dimension")
        for kind, matrix in matrices.items():
            if matrix.ndim != 2 or matrix.shape[1] != dimension:
                raise ValueError(f"invalid {kind} matrix shape in {root}")
            if matrix.dtype not in {np.dtype(np.float16), np.dtype(np.float32)}:
                raise ValueError(f"unsupported {kind} dtype in {root}: {matrix.dtype}")
        records = []
        for line_number, line in enumerate(
            (root / "records.jsonl").read_text(encoding="utf-8").splitlines(), start=1
        ):
            if line.strip():
                value = json.loads(line)
                records.append(self._mapping(value, f"record at line {line_number}"))
        self._validate_source_alignment(root, manifest, records, matrices)
        return {
            "root": root,
            "manifest": manifest,
            "manifest_sha256": self._sha256(manifest_path),
            "model": model,
            "matrices": matrices,
            "records": records,
        }

    def _validate_source_alignment(
        self,
        root: Path,
        manifest: dict[str, Any],
        records: list[dict[str, Any]],
        matrices: dict[str, np.ndarray],
    ) -> None:
        counts = self._mapping(manifest.get("counts"), "manifest counts")
        expected_counts = {
            "records": len(records),
            "text_vectors": len(matrices["text_vector"]),
            "image_vectors": len(matrices["image_vector"]),
            "provenance_only": sum(raw.get("vector_ref") is None for raw in records),
        }
        for name, actual in expected_counts.items():
            if counts.get(name) != actual:
                raise ValueError(
                    f"manifest count mismatch for {name} in {root}: "
                    f"expected {counts.get(name)!r}, found {actual}"
                )

        rows: dict[str, set[int]] = {kind: set() for kind in matrices}
        for raw in records:
            reference = raw.get("vector_ref")
            if reference is None:
                continue
            reference = self._mapping(reference, "vector_ref")
            kind = reference.get("kind")
            row = reference.get("row")
            if kind not in rows or not isinstance(row, int) or row < 0:
                raise ValueError(f"invalid vector reference in {root}: {reference!r}")
            if row in rows[kind]:
                raise ValueError(f"duplicate {kind} row {row} in {root}")
            rows[kind].add(row)
        for kind, matrix in matrices.items():
            expected = set(range(len(matrix)))
            if rows[kind] != expected:
                missing = sorted(expected - rows[kind])[:8]
                extra = sorted(rows[kind] - expected)[:8]
                raise ValueError(
                    f"{kind} rows do not cover its matrix in {root}; "
                    f"missing={missing}, extra={extra}"
                )

    def _verify_source_files(self, root: Path, manifest: dict[str, Any]) -> None:
        files = self._mapping(manifest.get("files"), "manifest files")
        for relative, expected in files.items():
            if not isinstance(relative, str) or not isinstance(expected, str):
                raise ValueError("manifest file checksums must be strings")
            path = self._contained_file(root, relative)
            if not hmac.compare_digest(self._sha256(path), expected.lower()):
                raise ValueError(f"source checksum mismatch: {relative}")

    @staticmethod
    def _shared_model(sources: list[dict[str, Any]]) -> dict[str, Any]:
        first = sources[0]["model"]
        identity = (
            first.get("provider"),
            first.get("model_id"),
            first.get("dimension"),
            first.get("normalize_embeddings"),
        )
        if not isinstance(first.get("dimension"), int) or first["dimension"] <= 0:
            raise ValueError("embedding dimension must be a positive integer")
        if first.get("normalize_embeddings") is not True:
            raise ValueError("import currently requires normalized Qwen3-VL embeddings")
        for source in sources[1:]:
            model = source["model"]
            candidate = (
                model.get("provider"),
                model.get("model_id"),
                model.get("dimension"),
                model.get("normalize_embeddings"),
            )
            if candidate != identity:
                raise ValueError("all imported knowledge bases must use the same model")
        return first

    @staticmethod
    def _collection_id(sources: list[dict[str, Any]]) -> str:
        digest = hashlib.sha256()
        for value in sorted(source["manifest_sha256"] for source in sources):
            digest.update(value.encode("ascii"))
        return digest.hexdigest()

    def _copy_crop(
        self,
        source_root: Path,
        metadata: dict[str, Any],
        output_root: Path,
        source_tag: str,
    ) -> str:
        crop = metadata.get("crop")
        if not isinstance(crop, str) or not crop.startswith("crops/"):
            raise ValueError("image vector record does not reference a crop")
        source = self._contained_file(source_root, crop)
        relative = Path(crop)
        destination_relative = Path("crops") / source_tag / Path(*relative.parts[1:])
        destination = output_root / destination_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return destination_relative.as_posix()

    @staticmethod
    def _content(raw: dict[str, Any]) -> str:
        for key in ("plain_text", "embedding_text"):
            value = raw.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        raise ValueError(f"record {raw.get('id')} has no usable content")

    @staticmethod
    def _pages(metadata: dict[str, Any]) -> tuple[int, int]:
        page_index = metadata.get("page_index")
        if isinstance(page_index, int) and page_index >= 0:
            return page_index + 1, page_index + 1
        page_indexes = [
            value
            for value in metadata.get("page_indexes", [])
            if isinstance(value, int) and value >= 0
        ]
        if page_indexes:
            return min(page_indexes) + 1, max(page_indexes) + 1
        return 1, 1

    @staticmethod
    def _profile(
        name: str,
        document_id: str,
        source_count: int,
        records: list[KnowledgeRecord],
        types: set[str],
        headings: set[str],
    ) -> CorpusProfile:
        topics = list(dict.fromkeys([*sorted(headings), *sorted(types)]))[:20]
        if not topics:
            topics = ["document content"]
        return CorpusProfile(
            document_id=document_id,
            document_name=name,
            document_type="Qwen3-VL imported knowledge collection",
            summary=(
                f"Imported collection containing {source_count} knowledge bases and "
                f"{len(records)} searchable records."
            ),
            in_scope_topics=topics,
            out_of_scope_examples=[],
            key_sections=sorted(headings)[:40],
            key_entities=[],
            profile_source="deterministic-qwen3vl-import",
            generation_model="none",
            prompt_version="qwen3vl-import-v1",
        )

    @staticmethod
    def _mapping(value: Any, label: str) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError(f"{label} must be a JSON object")
        return value

    @staticmethod
    def _strings(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    @staticmethod
    def _language(text: str) -> str:
        has_cjk = bool(re.search(r"[\u3400-\u9fff]", text))
        has_latin = bool(re.search(r"[A-Za-z]", text))
        if has_cjk and has_latin:
            return "mixed"
        if has_cjk:
            return "zh"
        if has_latin:
            return "en"
        return "und"

    @staticmethod
    def _contained_file(root: Path, relative: str) -> Path:
        value = Path(relative)
        if value.is_absolute():
            raise ValueError(f"manifest path must be relative: {relative}")
        candidate = (root / value).resolve()
        if root not in candidate.parents or not candidate.is_file():
            raise ValueError(f"manifest path is invalid: {relative}")
        return candidate

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _sha256_bytes(value: bytes) -> str:
        return hashlib.sha256(value).hexdigest()
