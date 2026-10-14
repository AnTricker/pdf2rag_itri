"""Read the current SuryaQ3 knowledge base without importing or rewriting it."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image

from .index import SearchHit
from .models import CorpusProfile

_LOG = logging.getLogger(__name__)


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


@dataclass(frozen=True)
class Q3RecordView:
    """A runtime view; the original record remains available as raw."""
    raw: dict[str, Any]
    source: SimpleNamespace

    @property
    def record_id(self) -> str:
        return self.raw['id']

    @property
    def modality(self) -> str:
        return 'image' if self.raw['vector_ref']['kind'] == 'image_vector' else 'text'

    @property
    def content(self) -> str:
        # Preserve structured table/list text for answer evidence.
        return str(self.raw.get('embedding_text') or self.raw.get('plain_text') or '')


class Q3KnowledgeIndex:
    source_mode = 'suryaq3'
    direct_retrieval = True

    def __init__(self, root: Path, kb_name: str):
        self.output_root = root
        self.kb_name = kb_name
        self.raw_manifest = _object(root / 'manifest.json')
        if self.raw_manifest.get('schema_version') != '1.2':
            raise ValueError('serveq3 supports current SuryaQ3 schema 1.2 only')
        self.manifest_hash = hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest()
        model = self.raw_manifest.get('model')
        if not isinstance(model, dict) or model.get('provider') != 'sentence_transformers':
            raise ValueError('Invalid SuryaQ3 model metadata')
        dimension = model.get('dimension')
        if type(dimension) is not int or dimension <= 0 or not model.get('model_id'):
            raise ValueError('Missing Q3 model identity or dimension')
        self.manifest = SimpleNamespace(vector_dimension=dimension)
        self.matrices = {}
        self.normalized_matrices = {}
        for kind, filename in [('text_vector', 'text.npy'), ('image_vector', 'image.npy')]:
            matrix = np.load(root / 'vectors' / filename, allow_pickle=False)
            if matrix.ndim != 2 or matrix.shape[1] != dimension or matrix.dtype.kind != 'f':
                raise ValueError(f'Invalid Q3 matrix: {filename}')
            if not np.isfinite(matrix).all():
                raise ValueError(f'Non-finite Q3 vectors: {filename}')
            self.matrices[kind] = matrix
            work = matrix.astype(np.float32)
            norms = np.linalg.norm(work, axis=1, keepdims=True)
            self.normalized_matrices[kind] = np.divide(
                work, norms, out=np.zeros_like(work), where=norms != 0,
            )
        self.image_catalog = {}
        metadata = _object(root / 'embedding_inputs' / 'metadata.json')
        items = metadata.get('items')
        if not isinstance(items, list):
            raise ValueError('Invalid Q3 image metadata items')
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get('source_image_id'), str):
                raise ValueError('Invalid Q3 image metadata item')
            image_id = item['source_image_id']
            if image_id in self.image_catalog:
                raise ValueError(f'Duplicate Q3 image ID: {image_id}')
            self.image_catalog[image_id] = item
        self.raw_records = []
        self.records = []
        self.record_by_id = {}
        self.ignored_count = 0
        self.invalid_count = 0
        for number, line in enumerate((root / 'records.jsonl').read_text(encoding='utf-8').splitlines(), 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if (not isinstance(raw, dict) or not isinstance(raw.get('metadata'), dict)
                    or not isinstance(raw.get('id'), str) or not raw['id']):
                raise ValueError(f'Invalid Q3 record structure at line {number}')
            self.raw_records.append(raw)
            ref = raw.get('vector_ref')
            if ref is None:
                self.ignored_count += 1
                continue
            if (not isinstance(ref, dict) or not isinstance(ref.get('kind'), str)
                    or ref['kind'] not in self.matrices
                    or type(ref.get('row')) is not int
                    or not 0 <= ref['row'] < len(self.matrices[ref['kind']])):
                self.invalid_count += 1
                _LOG.error('Q3 record %s excluded: invalid vector_ref %r', raw['id'], ref)
                continue
            if raw['id'] in self.record_by_id:
                raise ValueError(f'Duplicate Q3 record ID: {raw["id"]}')
            md = raw['metadata']
            for field in ('heading_path', 'types'):
                values = md.get(field, [])
                if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
                    raise ValueError(f'Invalid Q3 {field}: {raw["id"]}')
            page = md.get('page_index')
            indexes = [page] if type(page) is int else md.get('page_indexes', [])
            if not isinstance(indexes, list) or any(type(p) is not int or p < 0 for p in indexes):
                raise ValueError(f'Invalid Q3 page metadata: {raw["id"]}')
            pages = sorted({p + 1 for p in indexes})
            image_id = md.get("source_image_id")
            image_item = self.image_catalog.get(image_id) if isinstance(image_id, str) else None
            crop_info = image_item.get("source_crop") if image_item else None
            crop_path = crop_info.get("path") if isinstance(crop_info, dict) else None
            source = SimpleNamespace(
                document_name=kb_name, page_start=pages[0] if pages else None,
                page_end=pages[-1] if pages else None, page_numbers=pages,
                section_path=md.get('heading_path', []), content_types=md.get('types', []),
                source_kb_id=kb_name, source_record_id=raw['id'],
                source_image_id=md.get('source_image_id'),
                artifact_path=crop_path if isinstance(crop_path, str) and ref['kind'] == 'image_vector' else None,
                vector_ref=dict(ref),
            )
            view = Q3RecordView(raw, source)
            self.records.append(view)
            self.record_by_id[view.record_id] = view
        if not self.records:
            raise ValueError('Q3 KB contains no searchable records')
        headings = list(dict.fromkeys(
            str(h) for r in self.records for h in r.source.section_path if h
        ))[:20]
        types = list(dict.fromkeys(
            str(t) for r in self.records for t in r.source.content_types if t
        ))[:20]
        self.suggested_questions = [f'{h} 的內容重點是什麼？' for h in headings[:4]]
        if not self.suggested_questions:
            self.suggested_questions = [f'請說明文件中 {t} 類型的內容。' for t in types[:4]]
        self.corpus_profile = CorpusProfile(
            document_id=self.manifest_hash, document_name=kb_name,
            document_type='SuryaQ3 knowledge base',
            summary=f'{kb_name}: {len(self.records)} searchable records. Profile is a query hint, not an admission rule.',
            in_scope_topics=headings or types or [kb_name], key_sections=headings,
            profile_source='q3_runtime', generation_model='none', prompt_version='q3-runtime-v1',
        )
        _LOG.info('Q3 loaded: KB=%s searchable=%s provenance_only=%s invalid=%s model=%s',
                  kb_name, len(self.records), self.ignored_count, self.invalid_count, model)

    @classmethod
    def load(cls, q3_root: Path, kb_name: str) -> 'Q3KnowledgeIndex':
        if not kb_name or kb_name in {'.', '..'} or '/' in kb_name or '\\' in kb_name:
            raise ValueError('serveq3 requires a single KB name')
        base = q3_root.resolve()
        root = (base / kb_name / 'knowledge_base').resolve(strict=True)
        if base not in root.parents:
            raise ValueError('Q3 KB is outside LOCAL_RAG_Q3_ROOT')
        return cls(root, kb_name)

    @property
    def log_context(self) -> dict[str, Any]:
        return dict(source_mode=self.source_mode, kb_name=self.kb_name,
                    manifest_sha256=self.manifest_hash, kb_schema='1.2',
                    embedding_model=self.raw_manifest['model'])

    def validate_query_model(self, backend) -> None:
        if backend.dimension != self.manifest.vector_dimension:
            raise ValueError('Query embedding dimension does not match Q3 KB')
        # Model paths may move; compare source names against local model metadata.
        expected = str(self.raw_manifest['model']['model_id']).rstrip('/').split('/')[-1]
        candidates = {Path(backend.model_name).name}
        config_path = Path(backend.model_name) / 'config.json'
        config = _object(config_path) if config_path.is_file() else {}
        model_path = Path(backend.model_name).resolve()
        if model_path.parent.name == 'snapshots' and model_path.parent.parent.name.startswith('models--'):
            candidates.add(model_path.parent.parent.name.split('--')[-1])
        original = config.get('_name_or_path')
        if original:
            candidates.add(str(original).rstrip('/').split('/')[-1])
        if expected not in candidates:
            raise ValueError(f'Query model identity does not match Q3 model {expected}; local identities: {sorted(candidates)}')
        revision = self.raw_manifest['model'].get('revision')
        if revision:
            actual = config.get('_commit_hash')
            model_path = Path(backend.model_name).resolve()
            expected_revision = str(revision)
            if model_path.parent.name == 'snapshots':
                actual = model_path.name
                refs = model_path.parent.parent / 'refs'
                revision_file = (refs / expected_revision).resolve()
                if refs.resolve() in revision_file.parents and revision_file.is_file():
                    expected_revision = revision_file.read_text().strip()
            if actual != expected_revision:
                raise ValueError('Cannot verify matching Q3 model revision from local model metadata')

    def search(self, query_vector, *, top_k: int, modalities=None) -> list[SearchHit]:
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if query.shape != (self.manifest.vector_dimension,):
            raise ValueError('Query vector dimension does not match Q3 KB')
        norm = np.linalg.norm(query)
        if not np.isfinite(query).all() or not np.isfinite(norm) or norm == 0:
            return []
        scores = {kind: matrix @ (query / norm) for kind, matrix in self.normalized_matrices.items()}
        hits = [SearchHit(record=r, score=float(scores[r.raw['vector_ref']['kind']][r.raw['vector_ref']['row']]))
                for r in self.records if modalities is None or r.modality in modalities]
        return sorted(hits, key=lambda hit: (-hit.score, hit.record.record_id))[:top_k]

    def resolve_crop(self, record) -> Path | None:
        if record.modality != 'image':
            return None
        try:
            item = self.image_catalog.get(record.source.source_image_id)
            if item is None:
                raise ValueError('image metadata entry missing')
            inputs = item.get('embedding_inputs', [])
            if not isinstance(inputs, list) or len(inputs) != 1:
                raise ValueError('invalid image metadata inputs')
            if inputs[0].get('vector_ref') != record.raw['vector_ref']:
                raise ValueError('image metadata vector reference mismatch')
            ref = record.raw['metadata'].get('image_metadata_ref')
            expected = f'embedding_inputs/metadata.json#{record.source.source_image_id}'
            if ref != expected:
                raise ValueError('image_metadata_ref mismatch')
            crop = item.get('source_crop')
            relative = crop.get('path') if isinstance(crop, dict) else None
            if not isinstance(relative, str):
                raise ValueError('source crop path missing')
            candidate = (self.output_root / relative).resolve()
            crop_root = (self.output_root / 'crops').resolve()
            if crop_root not in candidate.parents or not candidate.is_file():
                raise ValueError('crop missing or outside KB crops')
            with Image.open(candidate) as image:
                image.verify()
            return candidate
        except (OSError, ValueError, TypeError, AttributeError, KeyError, Image.DecompressionBombError) as error:
            _LOG.error('Q3 crop unavailable: KB=%s record=%s image=%s: %s',
                       self.kb_name, record.record_id, record.source.source_image_id, error)
            return None
