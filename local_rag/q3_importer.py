from __future__ import annotations

from collections import Counter
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
    CorpusProfileDraft,
    EmbeddingMetadata,
    KnowledgeRecord,
    ProcessingMetadata,
    SnapshotManifest,
    SourceMetadata,
)
from .snapshot import BuildWriter


_OUTPUT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_PROFILE_NOISE_TYPES = {"pageheader", "pagefooter"}


class Q3Importer:
    """Import official Qwen3-VL knowledge bases into one serveable build."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def import_collections(
        self,
        sources: Sequence[Path],
        *,
        output_id: str,
        collection_name: str,
        profile_path: Path | None = None,
    ) -> Path:
        if not sources:
            raise ValueError("at least one Qwen3-VL knowledge base is required")
        if not _OUTPUT_ID.fullmatch(output_id):
            raise ValueError("output id may contain only letters, numbers, '.', '_' and '-'")
        name = collection_name.strip()
        if not name:
            raise ValueError("collection name cannot be blank")

        output_root = self.config.index_root / output_id
        if output_root.exists():
            raise FileExistsError(f"output already exists: {output_root}")

        knowledge_bases = [self._knowledge_base(path) for path in sources]
        loaded = sorted(
            [self._load_source(path) for path in knowledge_bases],
            key=lambda source: source["manifest_sha256"],
        )
        source_hashes = [source["manifest_sha256"] for source in loaded]
        if len(source_hashes) != len(set(source_hashes)):
            raise ValueError("duplicate Qwen3-VL knowledge base input")
        model = self._shared_model(loaded)
        preprocessing_sha256 = self._shared_preprocessing(loaded)
        document_id = self._collection_id(loaded)
        profile_override = self._profile_override(profile_path)

        output_root.mkdir(parents=True)
        try:
            records: list[KnowledgeRecord] = []
            vectors: list[np.ndarray] = []
            record_ids: set[str] = set()
            ignored_count = 0
            heading_counts: Counter[str] = Counter()
            heading_order: dict[str, int] = {}
            type_counts: Counter[str] = Counter()

            for source_number, source in enumerate(loaded, start=1):
                source_tag = f"{source_number:03d}-{source['manifest_sha256'][:12]}"
                matrices = source["matrices"]
                for raw in source["records"]:
                    metadata = self._mapping(raw.get("metadata"), "record metadata")
                    headings = self._strings(metadata.get("heading_path"))
                    content_types = self._strings(metadata.get("types"))
                    vector_ref = raw.get("vector_ref")
                    if vector_ref is None:
                        ignored_count += 1
                        continue
                    self._count_values(headings, heading_counts, heading_order)
                    type_counts.update(content_types)
                    vector_ref = self._mapping(vector_ref, "vector_ref")
                    kind = vector_ref.get("kind")
                    row = vector_ref.get("row")
                    if kind not in matrices or type(row) is not int:
                        raise ValueError(f"invalid vector reference for record {raw.get('id')}")
                    matrix = matrices[kind]
                    if row < 0 or row >= len(matrix):
                        raise ValueError(f"vector row is out of range for record {raw.get('id')}")

                    raw_record_id = raw.get("id")
                    source_record_id = (
                        raw_record_id.strip() if isinstance(raw_record_id, str) else ""
                    )
                    if not source_record_id:
                        raise ValueError("Qwen3-VL record id cannot be blank")
                    record_id = self._record_id(source["manifest_sha256"], source_record_id)
                    if record_id in record_ids:
                        raise ValueError(f"duplicate imported record id: {record_id}")
                    record_ids.add(record_id)
                    modality = "image" if kind == "image_vector" else "text"
                    source_image_id = None
                    artifact_path = None
                    if modality == "image":
                        source_image_id, artifact_path = self._copy_image_crop(
                            source, metadata, row, output_root, source_tag
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
                                document_name=name,
                                document_sha256=document_id,
                                page_start=page_start,
                                page_end=page_end,
                                artifact_path=artifact_path,
                                section_path=headings,
                                source_record_id=source_record_id,
                                source_kb_id=source_tag,
                                source_image_id=source_image_id,
                                region_ids=self._strings(metadata.get("region_ids")),
                                source_indexes=self._strings(metadata.get("source_indexes")),
                                content_types=content_types,
                            ),
                            processing=ProcessingMetadata(
                                extractor="qwen3vl-import-v2",
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
                name=name,
                document_id=document_id,
                source_count=len(loaded),
                records=records,
                heading_counts=heading_counts,
                heading_order=heading_order,
                type_counts=type_counts,
                override=profile_override,
            )
            manifest = SnapshotManifest(
                document_sha256=document_id,
                record_count=len(records),
                vector_count=len(records),
                vector_dimension=dimension,
                embedding=EmbeddingMetadata(
                    backend="sentence_transformers_qwen3vl_import",
                    model=str(model["model_id"]),
                    provider=str(model["provider"]),
                    revision=self._optional_string(model.get("revision")),
                    normalized=True,
                    preprocessing_sha256=preprocessing_sha256,
                ),
                chunk_max_tokens=self.config.chunk_max_tokens,
                chunk_overlap_tokens=self.config.chunk_overlap_tokens,
                prompt_versions={"importer": "qwen3vl-import-v2"},
            )
            image_count = sum(record.modality == "image" for record in records)
            report = BuildReport(
                source_document=name,
                document_id=document_id,
                text_record_count=len(records) - image_count,
                image_record_count=image_count,
                source_image_count=image_count,
                image_chunk_count=image_count,
                provenance_only_count=ignored_count,
                embedded_count=len(records),
                warnings=[f"ignored {ignored_count} provenance-only records"],
                stage_statuses={
                    "source_checksum_validation": "complete",
                    "source_compatibility_validation": "complete",
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
        checksummed_files = self._verify_source_files(root, manifest)
        required_checksums = {
            "records.jsonl",
            "vectors/text.npy",
            "vectors/image.npy",
        }
        if not required_checksums.issubset(checksummed_files):
            raise ValueError(f"core artifacts are not covered by manifest checksums: {root}")
        model = self._mapping(manifest.get("model"), "manifest model")
        matrices = {
            "text_vector": np.load(root / "vectors" / "text.npy", allow_pickle=False),
            "image_vector": np.load(root / "vectors" / "image.npy", allow_pickle=False),
        }
        dimension = model.get("dimension")
        declared_dtype = model.get("dtype")
        expected_dtype = {
            "float16": np.dtype(np.float16),
            "float32": np.dtype(np.float32),
        }.get(declared_dtype)
        if expected_dtype is None:
            raise ValueError(f"unsupported manifest vector dtype in {root}: {declared_dtype!r}")
        for kind, matrix in matrices.items():
            if matrix.ndim != 2 or matrix.shape[1] != dimension:
                raise ValueError(f"invalid {kind} matrix shape in {root}")
            if matrix.dtype != expected_dtype:
                raise ValueError(
                    f"{kind} dtype does not match manifest in {root}: {matrix.dtype}"
                )

        records = []
        for line_number, line in enumerate(
            (root / "records.jsonl").read_text(encoding="utf-8").splitlines(), start=1
        ):
            if line.strip():
                records.append(
                    self._mapping(json.loads(line), f"record at line {line_number}")
                )
        self._validate_source_alignment(root, manifest, records, matrices)
        image_catalog = self._image_catalog(
            root,
            len(matrices["image_vector"]),
            checksummed_files,
        )
        return {
            "root": root,
            "manifest_sha256": self._sha256(manifest_path),
            "model": model,
            "preprocessing": self._preprocessing(root, manifest),
            "matrices": matrices,
            "records": records,
            "image_catalog": image_catalog,
        }

    def _image_catalog(
        self,
        root: Path,
        image_count: int,
        checksummed_files: set[str],
    ) -> dict[str, dict[str, Any]]:
        if image_count == 0:
            return {}
        metadata_path = root / "embedding_inputs" / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"official Qwen3-VL image metadata is required: {metadata_path}"
            )
        metadata = self._mapping(
            json.loads(metadata_path.read_text(encoding="utf-8")),
            "embedding input metadata",
        )
        items = metadata.get("items")
        if not isinstance(items, list):
            raise ValueError("embedding input metadata items must be a JSON array")
        catalog: dict[str, dict[str, Any]] = {}
        for item in items:
            item = self._mapping(item, "embedding input item")
            raw_source_image_id = item.get("source_image_id")
            source_image_id = (
                raw_source_image_id.strip()
                if isinstance(raw_source_image_id, str)
                else ""
            )
            if not source_image_id or source_image_id in catalog:
                raise ValueError(f"missing or duplicate source_image_id: {source_image_id!r}")
            source_crop = item.get("source_crop")
            crop_path = source_crop.get("path") if isinstance(source_crop, dict) else source_crop
            if (
                not isinstance(crop_path, str)
                or Path(crop_path).parts[:1] != ("crops",)
            ):
                raise ValueError(f"source crop is missing for image {source_image_id}")
            self._contained_file(root, crop_path)
            if crop_path not in checksummed_files:
                raise ValueError(
                    f"source crop is not covered by manifest checksum: {crop_path}"
                )
            item["_source_crop_path"] = crop_path
            catalog[source_image_id] = item
        if len(catalog) != image_count:
            raise ValueError("embedding input metadata count does not match image vector count")
        return catalog

    def _copy_image_crop(
        self,
        source: dict[str, Any],
        metadata: dict[str, Any],
        vector_row: int,
        output_root: Path,
        source_tag: str,
    ) -> tuple[str, str]:
        raw_source_image_id = metadata.get("source_image_id")
        source_image_id = (
            raw_source_image_id.strip()
            if isinstance(raw_source_image_id, str)
            else ""
        )
        if not source_image_id:
            raise ValueError("image vector record is missing source_image_id")
        if not self._valid_image_metadata_ref(
            metadata.get("image_metadata_ref"), source_image_id
        ):
            raise ValueError(f"image record {source_image_id} is missing image_metadata_ref")
        item = source["image_catalog"].get(source_image_id)
        if item is None:
            raise ValueError(f"image metadata not found for {source_image_id}")
        metadata_row = self._image_vector_row(item)
        if metadata_row is not None and metadata_row != vector_row:
            raise ValueError(f"image vector row mismatch for {source_image_id}")

        crop = item["_source_crop_path"]
        crop_source = self._contained_file(source["root"], crop)
        crop_relative = Path(crop)
        destination_relative = Path("crops") / source_tag / Path(*crop_relative.parts[1:])
        destination = output_root / destination_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(crop_source, destination)
        return source_image_id, destination_relative.as_posix()

    @staticmethod
    def _valid_image_metadata_ref(value: Any, source_image_id: str) -> bool:
        if isinstance(value, str):
            if value == source_image_id:
                return True
            path = value.split("#", 1)[0]
        elif isinstance(value, dict):
            path = value.get("path")
            referenced_id = value.get("source_image_id") or value.get("id")
            if referenced_id is not None and referenced_id != source_image_id:
                return False
        else:
            return False
        return path == "embedding_inputs/metadata.json"

    @staticmethod
    def _image_vector_row(item: dict[str, Any]) -> int | None:
        for key in ("image_vector_row", "vector_row", "row"):
            value = item.get(key)
            if type(value) is int:
                return value
        reference = item.get("vector_ref")
        if isinstance(reference, dict) and type(reference.get("row")) is int:
            return reference["row"]
        return None

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
            declared = counts.get(name)
            if type(declared) is not int or declared != actual:
                raise ValueError(
                    f"manifest count mismatch for {name} in {root}: "
                    f"expected {declared!r}, found {actual}"
                )

        rows: dict[str, set[int]] = {kind: set() for kind in matrices}
        for raw in records:
            reference = raw.get("vector_ref")
            if reference is None:
                continue
            reference = self._mapping(reference, "vector_ref")
            kind = reference.get("kind")
            row = reference.get("row")
            if kind not in rows or type(row) is not int or row < 0:
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

    def _verify_source_files(
        self, root: Path, manifest: dict[str, Any]
    ) -> set[str]:
        files = self._mapping(manifest.get("files"), "manifest files")
        for relative, expected in files.items():
            if not isinstance(relative, str):
                continue
            critical = (
                relative in {
                    "records.jsonl",
                    "vectors/text.npy",
                    "vectors/image.npy",
                }
                or Path(relative).parts[:1] == ("crops",)
            )
            if not critical:
                continue
            if not isinstance(expected, str):
                raise ValueError(f"checksum must be a string for index artifact: {relative}")
            path = self._contained_file(root, relative)
            actual = self._sha256(path)
            if not hmac.compare_digest(actual, expected.lower()):
                raise ValueError(
                    f"source checksum mismatch for {relative}: "
                    f"manifest={expected.lower()}, actual={actual}"
                )
        return {relative for relative in files if isinstance(relative, str)}

    @staticmethod
    def _preprocessing(root: Path, manifest: dict[str, Any]) -> Any:
        if "preprocessing" in manifest:
            preprocessing = manifest["preprocessing"]
            if isinstance(preprocessing, dict):
                return preprocessing
        config_path = root.parent / "resolved_config.json"
        if config_path.is_file():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            preprocessing = config.get("embedding_preprocess") if isinstance(config, dict) else None
            if isinstance(preprocessing, dict):
                return preprocessing
        raise ValueError(f"Qwen3-VL preprocessing metadata is missing for {root}")

    @classmethod
    def _shared_model(cls, sources: list[dict[str, Any]]) -> dict[str, Any]:
        first = sources[0]["model"]
        identity = cls._model_identity(first)
        if not isinstance(first.get("provider"), str) or not first["provider"].strip():
            raise ValueError("embedding provider and model id are required")
        if not isinstance(first.get("model_id"), str) or not first["model_id"].strip():
            raise ValueError("embedding provider and model id are required")
        if type(first.get("dimension")) is not int or first["dimension"] <= 0:
            raise ValueError("embedding dimension must be a positive integer")
        if first.get("normalize_embeddings") is not True:
            raise ValueError("import requires normalized Qwen3-VL embeddings")
        for source in sources[1:]:
            if cls._model_identity(source["model"]) != identity:
                raise ValueError("all imported knowledge bases must use the same model")
        return first

    @staticmethod
    def _model_identity(model: dict[str, Any]) -> tuple[Any, ...]:
        return (
            model.get("provider"),
            model.get("model_id"),
            model.get("revision"),
            model.get("dimension"),
            model.get("normalize_embeddings"),
        )

    @classmethod
    def _shared_preprocessing(cls, sources: list[dict[str, Any]]) -> str:
        fingerprints = {
            cls._sha256_bytes(cls._canonical_json(source["preprocessing"]))
            for source in sources
        }
        if len(fingerprints) != 1:
            raise ValueError("all imported knowledge bases must use the same preprocessing")
        return fingerprints.pop()

    @staticmethod
    def _collection_id(sources: list[dict[str, Any]]) -> str:
        digest = hashlib.sha256()
        for value in sorted(source["manifest_sha256"] for source in sources):
            digest.update(value.encode("ascii"))
        return digest.hexdigest()

    @staticmethod
    def _record_id(manifest_sha256: str, source_record_id: str) -> str:
        digest = hashlib.sha256(
            f"{manifest_sha256}:{source_record_id}".encode("utf-8")
        ).hexdigest()[:24]
        return f"q3-{digest}"

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
        if type(page_index) is int and page_index >= 0:
            return page_index + 1, page_index + 1
        page_indexes = [
            value
            for value in metadata.get("page_indexes", [])
            if type(value) is int and value >= 0
        ]
        if page_indexes:
            return min(page_indexes) + 1, max(page_indexes) + 1
        return 1, 1

    @classmethod
    def _profile(
        cls,
        *,
        name: str,
        document_id: str,
        source_count: int,
        records: list[KnowledgeRecord],
        heading_counts: Counter[str],
        heading_order: dict[str, int],
        type_counts: Counter[str],
        override: CorpusProfileDraft | None,
    ) -> CorpusProfile:
        if override is None:
            headings = sorted(
                heading_counts,
                key=lambda value: (-heading_counts[value], heading_order[value], value),
            )
            useful_types = [
                value
                for value, _count in type_counts.most_common()
                if value.lower() not in _PROFILE_NOISE_TYPES
            ]
            topics = list(dict.fromkeys([*headings, *useful_types]))[:20]
            if not topics:
                topics = ["文件內容"]
            text_count = sum(record.modality == "text" for record in records)
            image_count = len(records) - text_count
            page_count = cls._source_page_count(records)
            section_text = "、".join(headings[:5]) or "未標示章節"
            type_text = "、".join(useful_types[:5]) or "未標示內容類型"
            draft = CorpusProfileDraft(
                document_type="Qwen3-VL imported knowledge collection",
                summary=(
                    f"{name} 整合 {source_count} 個知識庫，包含 {text_count} 筆文字與 "
                    f"{image_count} 筆圖片紀錄，涵蓋 {page_count} 個來源頁碼位置；"
                    f"主要內容類型：{type_text}；主要章節：{section_text}。"
                ),
                in_scope_topics=topics,
                out_of_scope_examples=[],
                key_sections=headings[:40],
                key_entities=[],
            )
            profile_source = "deterministic-qwen3vl-import"
        else:
            draft = override
            profile_source = "user-qwen3vl-import-profile"
        return CorpusProfile(
            **draft.model_dump(),
            document_id=document_id,
            document_name=name,
            profile_source=profile_source,
            generation_model="none",
            prompt_version="qwen3vl-import-v2",
        )

    @staticmethod
    def _source_page_count(records: list[KnowledgeRecord]) -> int:
        intervals: dict[str | None, list[tuple[int, int]]] = {}
        for record in records:
            intervals.setdefault(record.source.source_kb_id, []).append(
                (record.source.page_start, record.source.page_end)
            )
        count = 0
        for source_intervals in intervals.values():
            merged: list[list[int]] = []
            for start, end in sorted(source_intervals):
                if not merged or start > merged[-1][1] + 1:
                    merged.append([start, end])
                else:
                    merged[-1][1] = max(merged[-1][1], end)
            count += sum(end - start + 1 for start, end in merged)
        return count

    @staticmethod
    def _profile_override(path: Path | None) -> CorpusProfileDraft | None:
        if path is None:
            return None
        return CorpusProfileDraft.model_validate_json(
            path.resolve(strict=True).read_text(encoding="utf-8")
        )

    @staticmethod
    def _count_values(
        values: list[str], counts: Counter[str], order: dict[str, int]
    ) -> None:
        for value in values:
            if value not in order:
                order[value] = len(order)
            counts[value] += 1

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
    def _optional_string(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

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
    def _canonical_json(value: Any) -> bytes:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")

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
