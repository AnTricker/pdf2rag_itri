# Reviewable PDF RAG Pipeline

## 1. End-to-end pipeline

~~~mermaid
flowchart LR
    PDF["PDF"] --> ING["Stage 1: ingest"]
    ING --> TXT["Text extraction"]
    ING --> CROP["Crop model (once)"]
    CROP --> CLASS["VLM classifier"]
    CLASS -->|table| TEX["Table extractor"]
    TEX --> TSUM["Text-only table summary"]
    CLASS -->|figure| FIG["Figure extractor"]
    CLASS -->|unusable| SKIP["Keep review metadata only"]
    TXT --> P1["pending/001_records.pretty.json"]
    TSUM --> P1
    FIG --> P1
    SKIP --> P1
    P1 --> REV["Stage 2: review"]
    REV --> PN["pending/NNN_records.pretty.json"]
    PN --> BUILD["Stage 3: build selected revision"]
    BUILD --> JSONL["records.jsonl"]
    BUILD --> EMB["embeddings.npy"]
    BUILD --> PROFILE["corpus_profile.json"]
    BUILD --> MANIFEST["manifest.json"]
    JSONL --> SERVE["serve --output"]
    EMB --> SERVE
~~~

每次 ingest 建立 runtime/outputs 下新的 timestamp output。Stage 1 與 Stage 2 只維護人工可讀 pretty；只有 Stage 3 會產生正式 RAG build。

## 2. Output lifecycle

~~~text
runtime/outputs/<timestamp>/
├─ crops/
├─ pending/
│  ├─ 001_records.pretty.json
│  └─ 002_records.pretty.json
└─ builds/
   ├─ 001/
   └─ 002/
~~~

- crops 只在 Stage 1 產生一次，review 不 recrop。
- pending 以數字版本保存；程式不覆寫既有 revision。
- builds 使用對應 review 版本；已存在時拒絕覆寫。
- build 不保存原始 PDF、pretty 副本或 crop 副本。
- serve 驗證 build，使用指定 output 中編號最大的有效 build。

## 3. Stage sequence

~~~mermaid
sequenceDiagram
    actor U as User
    participant CLI
    participant APP as PdfRagApplication
    participant PDF as Text/Crop pipeline
    participant VLM as Local model
    participant FS as Output filesystem
    participant EMB as Embedding backend

    U->>CLI: ingest --input PDF --mode multi
    CLI->>APP: ingest
    APP->>FS: create timestamp output
    APP->>PDF: extract text and crops
    loop each crop
        APP->>VLM: classify(image)
        alt table
            APP->>VLM: extract_table(image)
            APP->>VLM: summarize_table(cells)
        else figure
            APP->>VLM: extract_figure(image)
        else unusable
            APP->>APP: keep metadata, no caption
        end
    end
    APP->>FS: write pending/001 atomically

    U->>CLI: review --output timestamp
    CLI->>APP: review latest numeric revision
    APP->>FS: read latest pretty
    APP->>VLM: call only missing tasks
    alt changed
        APP->>FS: atomically write next revision
    else unchanged
        APP-->>U: nothing to update
    end

    U->>CLI: build --output timestamp --review NNN
    CLI->>APP: build selected revision
    APP->>APP: validate and filter records
    APP->>VLM: build corpus profile (text-only)
    APP->>EMB: embed formal record content
    APP->>FS: write builds/NNN atomically

    U->>CLI: serve --output timestamp
    CLI->>FS: select latest valid build
    CLI->>APP: load JSONL, embeddings and shared crops
~~~

## 4. Image decision tree

~~~mermaid
flowchart TD
    A["ReviewImageRecord"] --> B{"content_type missing?"}
    B -->|yes| C["classifier(image)"]
    B -->|no| D{"content_type"}
    C --> D
    D -->|unusable| Z["stop; never enter JSONL"]
    D -->|table| E{"table_extraction missing?"}
    E -->|yes| F["table extractor(image)"]
    E -->|no| G{"table_summary missing?"}
    F --> G
    G -->|yes| H["summary generator(cells only)"]
    G -->|no| I["complete table"]
    H --> I
    D -->|figure| J{"figure_extraction missing?"}
    J -->|yes| K["figure extractor(image)"]
    J -->|no| L["complete figure"]
    K --> L
~~~

~~~python
if record.content_type is None:
    record.content_type = classify(crop)

if record.content_type == "unusable":
    return

elif record.content_type == "table":
    if record.table_extraction is None:
        draft = extract_table(crop)
        record.table_extraction = add_deterministic_ids_and_dimensions(draft)
    if record.table_summary is None:
        summary = summarize_table(record.table_extraction)
        validate_evidence_cell_ids(summary)
        record.table_summary = summary

elif record.content_type == "figure":
    if record.figure_extraction is None:
        record.figure_extraction = extract_figure(crop)
~~~

API schema validation失敗時，同一任務立即重試一次；仍失敗就保持欄位缺少，未來 review 會再次嘗試。

## 5. Backend data contracts

Classifier 只輸出 content_type 與繁中 reason。detector label 是弱提示，只存在 pretty。unusable 僅用於近乎空白、完全損壞或確實無內容的 crop。

Table extractor 輸出 title、notes 與所有 cells。每個 cell 包含原文 text、1-based row/column、span 與封閉 cell type。空白 cell 必須保留空字串。程式負責：

- 產生 r1c1 格式的 id。
- 計算 row_count 與 column_count。
- 驗證 span、重疊與缺格。
- 驗證 summary evidence IDs。
- 逐 cell 生成 deterministic retrieval content。

正式 table Record 同時保存 structured cells 與 retrieval content，一表一 Record。若完整 content 超過 embedding token limit，build 失敗，不截斷或省略 notes。

Figure extractor 輸出繁中 visual_description 與原文 text_blocks；每個 text block 保存自由文字 location。Figure 不另呼叫 summary model。

## 6. Build filtering and failure boundaries

~~~python
for review_record in selected_review.records:
    if text:
        include()
    elif content_type == "unusable":
        exclude_and_count()
    elif table and extraction_and_summary_complete:
        include_complete_table()
    elif figure and extraction_complete:
        include_figure()
    else:
        exclude_incomplete_and_count()
~~~

- 所有圖片被排除時，只要 Text Records 存在，允許 text-only build 並寫 warning。
- 完整 table 超過 embedding limit、正式 Records 為空、profile 不符文件或 checksum 不一致時，整次 build 失敗。
- build 先在 temporary directory 完整寫入與驗證，成功後才 rename 到 builds/NNN。
- pretty envelope 內列出 editable/read-only fields；人工不直接修改 JSONL。