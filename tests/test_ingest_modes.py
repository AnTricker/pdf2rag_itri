from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from local_rag.application import PdfRagApplication
from local_rag.config import AppConfig
from local_rag.models import ExtractedPage, FigureExtraction, ImageCandidate, ImageClassification


class RecordingTextExtractor:
    name = 'fixture-text'

    def __init__(self):
        self.calls = 0

    def extract(self, pdf_path):
        self.calls += 1
        return [ExtractedPage(page_number=1, text='Reset procedure')]


class RecordingImagePipeline:
    name = 'fixture-image'

    def __init__(self):
        self.calls = 0

    def extract(self, pdf_path, artifact_root):
        self.calls += 1
        crop_dir = artifact_root / 'crops'
        crop_dir.mkdir(parents=True, exist_ok=True)
        crop = crop_dir / 'page_0001_crop_0001.jpg'
        crop.write_bytes(b'crop')
        return [ImageCandidate(
            page_number=1, crop_path=crop, artifact_name=crop.name,
            detector_label='diagram', confidence=0.9,
            bbox_normalized=(0.1, 0.1, 0.9, 0.9),
        )]


class ReviewBackend:
    name = 'fixture-review'
    model_name = 'fixture-model'
    prompt_versions = {'figure_extractor': 'figure-v1'}

    def classify(self, crop_path, detector_label):
        return ImageClassification(content_type='figure', reason='fixture')

    def extract_figure(self, crop_path):
        return FigureExtraction(visual_description='Reset diagram', text_blocks=[])


class EmbeddingBackend:
    name = 'fixture-embedding'
    model_name = 'fixture-2d'
    dimension = 2
    max_input_tokens = 64

    def count(self, text):
        return len(text.split())

    def embed(self, texts):
        return np.ones((len(texts), 2), dtype=np.float32)


class IngestModeTests(unittest.TestCase):
    def test_modes_run_only_selected_review_pipelines(self):
        expectations = {
            'text': ({'text'}, 1, 0),
            'image': ({'image'}, 0, 1),
            'multi': ({'text', 'image'}, 1, 1),
        }
        for mode, (modalities, text_calls, image_calls) in expectations.items():
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                pdf = root / 'manual.pdf'
                pdf.write_bytes(b'fixture')
                text = RecordingTextExtractor()
                image = RecordingImagePipeline()
                app = PdfRagApplication(
                    config=AppConfig.for_test(index_root=root / 'index'),
                    text_extractor=text,
                    embedding_backend=EmbeddingBackend(),
                    image_pipeline=image,
                    image_review_backend=ReviewBackend(),
                )
                output = app.ingest(pdf, mode=mode)
                envelope = json.loads((output / 'pending' / '001_records.pretty.json').read_text(encoding='utf-8'))
                self.assertEqual({record['modality'] for record in envelope['records']}, modalities)
                self.assertEqual(text.calls, text_calls)
                self.assertEqual(image.calls, image_calls)


if __name__ == '__main__':
    unittest.main()
