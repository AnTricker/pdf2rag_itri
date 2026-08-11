# Local RAG MVP：目前實作流程與模組圖

本文件描述目前 active implementation。完整的 reviewable ingest、sequence、判斷樹與偽碼請見 REVIEWABLE_PIPELINE.md。

## Current architecture

~~~mermaid
flowchart LR
    CLI["main.py CLI"] --> APP["PdfRagApplication"]
    APP --> ADAPTERS["PyPDF / Ultralytics / Ollama / Embedding"]
    APP --> OUTPUT["runtime/outputs/<timestamp>"]
    OUTPUT --> REVIEW["pending revisions"]
    REVIEW --> BUILD["versioned builds"]
    BUILD --> INDEX["FileVectorIndex"]
    INDEX --> CHAT["ChatEngine"]
    CHAT --> WEB["Flask UI"]
~~~

## CLI

| Command | Current behavior |
|---|---|
| doctor | 驗證本機模型、embedding、Poppler 與 detector 設定 |
| ingest | 建立 timestamp output、crops 與第一版 pretty；不建立 RAG |
| review | 只補最新 pretty 中缺少的 classifier/extraction/summary |
| build | 指定 review revision，建立版本化正式 RAG |
| serve | 指定 output，載入其中最新有效 build |

## Runtime files

~~~text
runtime/outputs/<timestamp>/
├─ crops/
├─ pending/NNN_records.pretty.json
└─ builds/NNN/
   ├─ build_report.json
   ├─ corpus_profile.json
   ├─ embeddings.npy
   ├─ manifest.json
   └─ records.jsonl
~~~

原始 PDF 不複製到 runtime。crops 在 output 內共享，不會複製進每個 build。

## HTTP interface

| Route | Purpose |
|---|---|
| GET / | Chat UI |
| GET /api/health | Build、embedding 與 local model readiness |
| POST /api/chat | 文件問答 |
| GET/DELETE /api/history/<session_id> | Session history |
| POST /api/session/<session_id>/end | 結束 session |
| GET /api/crops/<record_id> | 回傳正式 Image Record 所引用的共享 crop |

沒有原始 PDF 公開 route。Crop path 必須位於指定 output 的 crops 目錄，並通過 manifest checksum 與 path containment 驗證。

## Main seams

- PdfRagApplication：ingest/review/build orchestration。
- OllamaImageReviewBackend：classifier、table extractor、table summary、figure extractor。
- BuildWriter：不可覆寫的 build staging、validation 與 atomic publish。
- FileVectorIndex：載入指定 build 並驗證共享 crops。
- ChatEngine：多題拆分、routing、retrieval、evidence gate 與回答。