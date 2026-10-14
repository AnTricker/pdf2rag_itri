# Digital PDF Toolkit

本專案提供 HEIC 轉 PNG、PNG 合併 PDF、Digital PDF native extraction、Surya OCR 與 Marker 文件分析。

---
## Setup

```bash
./scripts/linux/setup.sh
```

```cmd
scripts\windows\setup.cmd
```

建立／更新 `digital-pdf-core`、`digital-pdf-surya` 與 `digital-pdf-marker`。

ROCm embedding 環境需明確指定與主機相容的 PyTorch wheel index，避免誤裝其他 build：

```bash
PYTORCH_ROCM_INDEX_URL=https://download.pytorch.org/whl/rocm6.4 \
  ./scripts/linux/setup-embedding.sh
```

## Commands

### HEIC to PNG

```bash
./scripts/linux/heic-to-png.sh phone-photos
./scripts/linux/heic-to-png.sh phone-photos --output converted-pages
```

預設輸出至 `<input>/scan_source_pages/`。只處理 input root 的 `.heic/.heif`，依 natural filename order 產生 `p0001.png...`。

---
### PNG to PDF

```bash
./scripts/linux/png-to-pdf.sh png-folder
./scripts/linux/png-to-pdf.sh png-folder --output document.pdf
```

只處理 input root 的 `.png`，依 natural filename order 合併為單一 PDF。預設輸出為 `<input>/<folder-name>.pdf`，不覆寫既有檔案。

---
### Extract

```bash
./scripts/linux/extract.sh document.pdf
./scripts/linux/extract.sh document.pdf --config config/custom.yml
```

輸出 native text、geometry、embedded images 與 page renders：

```text
output/MMDDHHmm_extract/
├─ document.json
├─ pages/
├─ page_renders/
├─ embedded_images/
├─ command.json
├─ environment.json
├─ status.json
├─ run.log
└─ events.jsonl
```

---
### Surya2

```bash
./scripts/linux/surya2.sh png-folder
./scripts/linux/surya2.sh document.pdf
./scripts/linux/surya2.sh png-folder --config config/custom.yml
```

PNG 不超過 10 張時，結果直接位於 `output/MMDDHHmm_surya2/`。超過 10 張時，input PNG 會依 natural order 移入 `1/、2/…`，每批 10 張，並循序執行：

```text
output/MMDDHHmm_surya2/
├─ 1/
├─ 2/
└─ 3/
```

PDF 會先 render：

```text
output/MMDDHHmm_surya2/
├─ rendered_png/
└─ result/                 # 不超過 10 頁
```

超過 10 頁時，`rendered_png/` 與 result root 都使用相同數字 batch folders。每個 result folder 包含：

```text
results.json
assets/crops/
assets/overlays/
assets/index.json
metadata/summary.json
metadata/samples.csv
command.json
environment.json
status.json
stdout.log
stderr.log
run.log
events.jsonl
```

既有 batch folder 必須從 `1/` 連續編號；每個 folder 可包含 1–10 張。重新執行會全部重跑並建立新的 timestamp output。自動分批仍以每批 10 張切分，最後一批可不足 10 張。

---
### Marker

```bash
./scripts/linux/marker.sh document.pdf
./scripts/linux/marker.sh png-folder
./scripts/linux/marker.sh document.pdf --config config/custom.yml
```

Marker 接受單一 PDF 或僅含第一層 PNG 的 folder。PNG folder 會依 natural order 轉成保留於 output 的 `input.image-only.pdf`，再交由 Marker 分析。除了官方 JSON、Markdown 與原生擷取圖片外，`result/` 保存 Marker 兩個原始 Document 階段；`block_provenance.json` 保存兩階段的 `text_extraction_method`，並將 `surya` 標示為 `ocr`：

```text
output/MMDDHHmm_marker/
├─ result.json
├─ result.md
├─ result_meta.json
├─ result/
│  ├─ DocumentBuilder.json
│  └─ PdfConverter.build_document.json
├─ block_provenance.json
├─ input.image-only.pdf       # 僅 PNG folder 輸入
├─ input_manifest.json        # 僅 PNG folder 輸入
├─ <Marker 原生圖片>
├─ metadata/
├─ command.json
├─ environment.json
├─ status.json
├─ stdout.log
├─ stderr.log
├─ run.log
└─ events.jsonl
```

Marker 預設固定使用 `balanced` mode，並透過 `llama.cpp` 的 `llama-server` 執行 Surya VLM；不啟用額外 LLM、強制 OCR、額外 crop 或 embedding normalization。

---
### Qwen3VL embedding

#### 建立與接續知識庫

先建立主機設定，填入本機模型路徑與 runtime：

```bash
cp config/local.example.yml config/local.yml
```

`config/local.yml` 已被 Git 忽略。`model_id` 可使用 Hugging Face ID，或已下載模型的絕對路徑：

```yaml
embedding:
  modes:
    qwen3vl:
      model_id: /absolute/path/to/Qwen3-VL-Embedding-8B
      runtime:
        device: cuda
        dtype: float16
        attention: sdpa
        batch_size: 1
```

輸入可為單一 Surya `assets/index.json`、單批 run，或包含數字 batch subfolder 的完整 Surya run：

```bash
./scripts/linux/qwen3vl.sh output/MMDDHHmm_surya2
./scripts/linux/qwen3vl.sh output/MMDDHHmm_surya2/1/assets/index.json --config config/local.yml
```

若 image embedding 中途失敗，使用原 Surya input 與既有 qwen3vl run 接續：

```bash
./scripts/linux/qwen3vl.sh output/MMDDHHmm_surya2 --resume-from output/MMDDHHmm_qwen3vl
```

Resume 強制沿用舊 run 的 `resolved_config.json`，不可同時使用 `--config`。已成功的 vectors、crop 與 embedding inputs 不重算；從上一張失敗圖片繼續。

#### Routing 規則

每個 Surya region 依序判斷：

```text
error
→ provenance only，不建立 vector

plain_text 非空
→ text_vector，不複製 crop

plain_text 為空且 crop 可讀
→ image_vector，保存代表 crop 與實際模型輸入圖

plain_text 為空且 crop 不可用
→ provenance only＋warning
```

文字以頁面為主要 chunk；table/form 等獨立文字 region 不會再重複放入頁面正文。Heading 透過 `heading_path` 跨頁延續；重複 header/footer 聚合為 `document_root` record。重複純圖片先經 pHash 去重，再對代表圖建立一個 image vector。

#### Output 結構

成功時：

```text
output/MMDDHHmm_qwen3vl/
├─ resolved_config.json
├─ command.json
├─ status.json
├─ stdout.log / stderr.log
└─ knowledge_base/
   ├─ manifest.json
   ├─ records.jsonl
   ├─ vectors/
   │  ├─ text.npy
   │  └─ image.npy
   ├─ crops/
   │  └─ <source-image-id>.png
   └─ embedding_inputs/
      ├─ metadata.json
      └─ <source-image-id>-overview.png
```

失敗時不會發布 `knowledge_base/`，checkpoint 會保存在 `knowledge_base.failed/`。每次 resume 另外產生 `stdout.resume-NN.log`、`stderr.resume-NN.log`、`command.resume-NN.json` 與 `resume-NN/metadata/`；`status.json.attempts` 保留每次執行狀態。若 process 被強制中止，可能留下可接續的 `.knowledge_base.tmp/`。

#### 檔案用途與對應方式

- `manifest.json`：保存 schema/model/revision、dimension、dtype、normalization、來源 index hash、record/vector 數量、warnings 與所有 KB artifact hash。
- `records.jsonl`：每行一筆可檢索 record；保留處理後文字、來源 metadata 及 vector row。
- `vectors/text.npy`：所有 `text_vector`，shape 為 `(text_count, dimension)`。
- `vectors/image.npy`：所有 `image_vector`，shape 為 `(image_count, dimension)`。
- `crops/`：只有建立 image vector 的純圖 region；保存去重後的原始代表 crop。
- `embedding_inputs/*-overview.png`：EXIF transpose、RGB conversion 與 resize 後，實際送入模型的圖片。
- `embedding_inputs/metadata.json`：連結原始 crop、overview、resize 尺寸、`image_grid_thw`、來源 regions、去重資訊與 image vector row；同時保存 resume checkpoint。

`records.jsonl` 的主要欄位：

```json
{
  "id": "deterministic-record-id",
  "embedding_text": "[type=table]\n[section=...]\n...",
  "plain_text": "...",
  "raw_html": "...",
  "metadata": {
    "scope": "page",
    "route": "text_vector",
    "region_ids": ["surya-p4-r12"],
    "types": ["table"],
    "page_index": 4,
    "heading_path": ["..."]
  },
  "vector_ref": {
    "kind": "text_vector",
    "row": 12
  }
}
```

`vector_ref.kind` 決定要讀取哪個 `.npy`，`vector_ref.row` 是該矩陣的列索引；provenance-only record 的 `vector_ref` 為 `null`。Image record 另有 `metadata.source_image_id` 與 `metadata.image_metadata_ref`，可在 `embedding_inputs/metadata.json` 找到對應 crop。

最小讀取範例：

```python
import json
from pathlib import Path

import numpy as np

kb = Path("output/MMDDHHmm_qwen3vl/knowledge_base")
records = [json.loads(line) for line in (kb / "records.jsonl").read_text().splitlines()]
text_vectors = np.load(kb / "vectors/text.npy", mmap_mode="r")
image_vectors = np.load(kb / "vectors/image.npy", mmap_mode="r")
image_metadata = json.loads((kb / "embedding_inputs/metadata.json").read_text())
images_by_id = {item["source_image_id"]: item for item in image_metadata["items"]}

record = next(item for item in records if item["vector_ref"] is not None)
ref = record["vector_ref"]
matrix = text_vectors if ref["kind"] == "text_vector" else image_vectors
vector = matrix[ref["row"]]

if ref["kind"] == "image_vector":
    image = images_by_id[record["metadata"]["source_image_id"]]
    crop_path = kb / image["source_crop"]["path"]
```

本 pipeline 目前只建立知識庫，尚未包含 query embedding、Top-K retrieval、reranker、vector DB、API 或前端。

模型與 runtime 預設值位於 `config/embedding.yml`；`config/local.yml` 會自動覆寫，單次執行的 `--config` 優先序最高。`HF_TOKEN`、`HF_HOME` 等秘密或主機值應由 shell／deployment environment 注入。

`config/local.example.yml` 中的 `embedding.image_worker.max_images_per_process` 是預留的 worker recycling 設定，目前程式尚未讀取；現階段仍需使用 `--resume-from` 啟動新的 process。

## Output naming

Run folder 使用本機時間 `MMDDHHmm_<mode>`。同分鐘、同 mode 的後續 run 依序使用 `_02`、`_03`。

## Configuration

- `config/default.yml`：output root、render DPI、heartbeat、metadata sampling 與 logging。
- `config/surya2.yml`：Surya environment、version 與實際 native command。
- `config/marker.yml`：Marker environment、version 與 worker command。
- `config/embedding.yml`：embedding mode、model adapter、runtime 與 preprocessing。
- `--config` 指定的 YAML 最後套用，可覆蓋以上設定。

Batch size 固定為 10，不開放調整。
