from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from local_rag.config import AppConfig
from local_rag.index import FileVectorIndex
from local_rag.q3_importer import Q3Importer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def rewrite_records(run: Path, records: list[dict]) -> None:
    kb = run / "knowledge_base"
    records_path = kb / "records.jsonl"
    records_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8",
    )
    manifest_path = kb / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["records.jsonl"] = sha256(records_path)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")


def write_q3_run(
    root: Path,
    name: str,
    *,
    model_id: str = "Qwen/Qwen3-VL-Embedding-8B",
    preprocessing_marker: str = "same",
) -> Path:
    run = root / name
    kb = run / "knowledge_base"
    (kb / "vectors").mkdir(parents=True)
    (kb / "crops").mkdir()
    (kb / "embedding_inputs").mkdir()
    (kb / "crops" / "image-1.png").write_bytes(b"fixture image")
    (kb / "embedding_inputs" / "image-1-overview.png").write_bytes(
        b"fixture overview"
    )
    np.save(
        kb / "vectors" / "text.npy",
        np.asarray([[1.0, 0.0]], dtype=np.float16),
        allow_pickle=False,
    )
    np.save(
        kb / "vectors" / "image.npy",
        np.asarray([[0.0, 1.0]], dtype=np.float16),
        allow_pickle=False,
    )
    records = [
        {
            "id": "text-1",
            "embedding_text": "[section=Reset]\n重設程序",
            "plain_text": "重設程序",
            "raw_html": "<p>重設程序</p>",
            "metadata": {
                "scope": "page",
                "route": "text_vector",
                "region_ids": ["region-text"],
                "types": ["text"],
                "page_index": 4,
                "heading_path": ["Reset"],
                "source_indexes": ["source/assets/index.json"],
            },
            "vector_ref": {"kind": "text_vector", "row": 0},
        },
        {
            "id": "image-1",
            "embedding_text": "[type=diagram]",
            "plain_text": "",
            "raw_html": "<img/>",
            "metadata": {
                "scope": "region",
                "route": "image_vector",
                "region_ids": ["region-image"],
                "types": ["diagram"],
                "page_index": 5,
                "heading_path": ["Reset"],
                "source_indexes": ["source/assets/index.json"],
                "source_image_id": "image-1",
                "image_metadata_ref": "embedding_inputs/metadata.json#image-1",
            },
            "vector_ref": {"kind": "image_vector", "row": 0},
        },
        {
            "id": "provenance-1",
            "embedding_text": "[type=pagefooter]",
            "plain_text": "",
            "raw_html": "",
            "metadata": {
                "scope": "region",
                "route": "provenance_only",
                "region_ids": ["region-footer"],
                "types": ["pagefooter"],
                "page_index": 5,
                "heading_path": [],
                "source_indexes": ["source/assets/index.json"],
            },
            "vector_ref": None,
        },
    ]
    (kb / "records.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8",
    )
    image_metadata = {
        "items": [
            {
                "source_image_id": "image-1",
                "source_crop": {"path": "crops/image-1.png"},
                "overview": {"path": "embedding_inputs/image-1-overview.png"},
                "image_vector_row": 0,
            }
        ]
    }
    (kb / "embedding_inputs" / "metadata.json").write_text(
        json.dumps(image_metadata, ensure_ascii=False), encoding="utf-8"
    )
    (run / "resolved_config.json").write_text(
        json.dumps(
            {"embedding_preprocess": {"fixture": preprocessing_marker}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    artifact_names = [
        "records.jsonl",
        "vectors/text.npy",
        "vectors/image.npy",
        "crops/image-1.png",
        "embedding_inputs/metadata.json",
        "embedding_inputs/image-1-overview.png",
    ]
    manifest = {
        "schema_version": "1.0",
        "mode": "qwen3vl",
        "model": {
            "provider": "sentence_transformers",
            "model_id": model_id,
            "revision": "fixture-revision",
            "dimension": 2,
            "dtype": "float16",
            "normalize_embeddings": True,
        },
        "counts": {
            "records": 3,
            "text_vectors": 1,
            "image_vectors": 1,
            "provenance_only": 1,
        },
        "sources": [
            {
                "index": f"{name}/assets/index.json",
                "sha256": hashlib.sha256(name.encode("utf-8")).hexdigest(),
            }
        ],
        "files": {value: sha256(kb / value) for value in artifact_names},
    }
    (kb / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    return run


class Q3ImporterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = AppConfig.for_test(index_root=self.root / "outputs")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_imports_official_text_image_and_provenance_routes(self) -> None:
        source = write_q3_run(self.root, "source-a")
        build = Q3Importer(self.config).import_collections(
            [source], output_id="combined", collection_name="Fixture Collection"
        )
        index = FileVectorIndex.load(build, output_root=build.parents[1])
        self.assertEqual(index.vectors.shape, (2, 2))
        self.assertEqual(index.vectors.dtype, np.float32)
        self.assertEqual(index.manifest.embedding.provider, "sentence_transformers")
        self.assertEqual([record.modality for record in index.records], ["text", "image"])
        self.assertEqual(index.records[0].source.page_start, 5)
        self.assertEqual(index.records[0].source.source_record_id, "text-1")
        self.assertEqual(index.records[1].source.source_image_id, "image-1")
        report = json.loads((build / "build_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["provenance_only_count"], 1)
        self.assertTrue(
            (build.parents[1] / index.records[1].source.artifact_path).is_file()
        )
        self.assertEqual(index.corpus_profile.in_scope_topics[0], "Reset")

    def test_merges_sources_with_namespaced_ids_and_profile_override(self) -> None:
        first = write_q3_run(self.root, "source-a")
        second = write_q3_run(self.root, "source-b")
        profile = self.root / "profile.json"
        profile.write_text(
            json.dumps(
                {
                    "document_type": "CMP manuals",
                    "summary": "人工確認摘要",
                    "in_scope_topics": ["重設"],
                    "out_of_scope_examples": ["天氣"],
                    "key_sections": ["Reset"],
                    "key_entities": ["CMP"],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        build = Q3Importer(self.config).import_collections(
            [first, second],
            output_id="combined",
            collection_name="Fixture Collection",
            profile_path=profile,
        )
        index = FileVectorIndex.load(build, output_root=build.parents[1])
        self.assertEqual(len(index.records), 4)
        self.assertEqual(len({record.record_id for record in index.records}), 4)
        self.assertEqual(index.corpus_profile.summary, "人工確認摘要")
        self.assertEqual(index.corpus_profile.profile_source, "user-qwen3vl-import-profile")

    def test_rejects_old_image_format_without_embedding_input_metadata(self) -> None:
        source = write_q3_run(self.root, "source-a")
        kb = source / "knowledge_base"
        records = [
            json.loads(line)
            for line in (kb / "records.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        records[1]["metadata"].pop("source_image_id")
        records[1]["metadata"].pop("image_metadata_ref")
        records[1]["metadata"]["crop"] = "crops/image-1.png"
        rewrite_records(source, records)
        (kb / "embedding_inputs" / "metadata.json").unlink()
        (kb / "embedding_inputs" / "image-1-overview.png").unlink()
        manifest_path = kb / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"].pop("embedding_inputs/metadata.json")
        manifest["files"].pop("embedding_inputs/image-1-overview.png")
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
        with self.assertRaisesRegex((FileNotFoundError, ValueError), "metadata"):
            Q3Importer(self.config).import_collections(
                [source], output_id="combined", collection_name="Fixture"
            )

    def test_rejects_checksum_vector_and_source_compatibility_errors(self) -> None:
        checksum_source = write_q3_run(self.root, "checksum")
        (checksum_source / "knowledge_base" / "records.jsonl").write_text(
            "{}\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "checksum"):
            Q3Importer(self.config).import_collections(
                [checksum_source], output_id="checksum-output", collection_name="Fixture"
            )

        first = write_q3_run(self.root, "source-a")
        other_model = write_q3_run(self.root, "source-b", model_id="different-model")
        with self.assertRaisesRegex(ValueError, "same model"):
            Q3Importer(self.config).import_collections(
                [first, other_model], output_id="model-output", collection_name="Fixture"
            )

        other_preprocessing = write_q3_run(
            self.root, "source-c", preprocessing_marker="different"
        )
        with self.assertRaisesRegex(ValueError, "same preprocessing"):
            Q3Importer(self.config).import_collections(
                [first, other_preprocessing],
                output_id="preprocessing-output",
                collection_name="Fixture",
            )

        wrong_dtype = write_q3_run(self.root, "wrong-dtype")
        matrix_path = wrong_dtype / "knowledge_base" / "vectors" / "text.npy"
        np.save(matrix_path, np.asarray([[1.0, 0.0]], dtype=np.float32), allow_pickle=False)
        manifest_path = wrong_dtype / "knowledge_base" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"]["vectors/text.npy"] = sha256(matrix_path)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "dtype does not match"):
            Q3Importer(self.config).import_collections(
                [wrong_dtype], output_id="dtype-output", collection_name="Fixture"
            )

    def test_cleans_incomplete_output_after_conversion_failure(self) -> None:
        source = write_q3_run(self.root, "bad-reference")
        records = [
            json.loads(line)
            for line in (source / "knowledge_base" / "records.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        records[1]["metadata"]["image_metadata_ref"] = "wrong.json#image-1"
        rewrite_records(source, records)
        with self.assertRaisesRegex(ValueError, "image_metadata_ref"):
            Q3Importer(self.config).import_collections(
                [source], output_id="failed-output", collection_name="Fixture"
            )
        self.assertFalse((self.config.index_root / "failed-output").exists())

    def test_warns_but_imports_when_non_index_metadata_checksum_is_stale(self) -> None:
        source = write_q3_run(self.root, "stale-metadata-checksum")
        metadata_path = source / "knowledge_base" / "embedding_inputs" / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["checkpoint_note"] = "written after manifest publication"
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
        with self.assertWarnsRegex(RuntimeWarning, "metadata.json"):
            build = Q3Importer(self.config).import_collections(
                [source], output_id="warning-output", collection_name="Fixture"
            )
        report = json.loads((build / "build_report.json").read_text(encoding="utf-8"))
        self.assertTrue(any("metadata.json" in item for item in report["warnings"]))

    def test_accepts_numeric_schema_version(self) -> None:
        source = write_q3_run(self.root, "numeric-schema")
        manifest_path = source / "knowledge_base" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["schema_version"] = 1.0
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
        build = Q3Importer(self.config).import_collections(
            [source], output_id="numeric-schema-output", collection_name="Fixture"
        )
        self.assertEqual(build.name, "001")

    def test_uses_record_vector_row_when_image_metadata_does_not_repeat_it(self) -> None:
        source = write_q3_run(self.root, "metadata-without-row")
        kb = source / "knowledge_base"
        metadata_path = kb / "embedding_inputs" / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["items"][0].pop("image_vector_row")
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
        manifest_path = kb / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"]["embedding_inputs/metadata.json"] = sha256(metadata_path)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
        build = Q3Importer(self.config).import_collections(
            [source], output_id="metadata-without-row-output", collection_name="Fixture"
        )
        index = FileVectorIndex.load(build, output_root=build.parents[1])
        self.assertEqual(index.records[1].source.source_image_id, "image-1")

    def test_rejects_existing_output(self) -> None:
        source = write_q3_run(self.root, "source-a")
        (self.config.index_root / "combined").mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            Q3Importer(self.config).import_collections(
                [source], output_id="combined", collection_name="Fixture"
            )

    def test_rejects_duplicate_and_missing_vector_rows(self) -> None:
        duplicate = write_q3_run(self.root, "duplicate-row")
        duplicate_records = [
            json.loads(line)
            for line in (duplicate / "knowledge_base" / "records.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        duplicate_records[1]["vector_ref"] = {"kind": "text_vector", "row": 0}
        rewrite_records(duplicate, duplicate_records)
        with self.assertRaisesRegex(ValueError, "duplicate text_vector row"):
            Q3Importer(self.config).import_collections(
                [duplicate], output_id="duplicate-output", collection_name="Fixture"
            )

        missing = write_q3_run(self.root, "missing-row")
        missing_records = [
            json.loads(line)
            for line in (missing / "knowledge_base" / "records.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        missing_records[1]["vector_ref"] = None
        rewrite_records(missing, missing_records)
        manifest_path = missing / "knowledge_base" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["counts"]["provenance_only"] = 2
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "do not cover"):
            Q3Importer(self.config).import_collections(
                [missing], output_id="missing-output", collection_name="Fixture"
            )


if __name__ == "__main__":
    unittest.main()
