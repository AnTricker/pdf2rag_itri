# Marker 第三 Mode 與 PNG-to-PDF 實作計畫

## 摘要

- 新增 `marker <PDF>`，以 Marker 2.x 官方流程分析單一 PDF。
- 同一次分析產生官方原始 JSON、Markdown及原生圖片，不做結構轉換或 embedding。
- 唯一附加輸出為 block 級文字來源：`pdftext` 或 `ocr`。
- 新增獨立 optional command `png-to-pdf`。
- 沿用 `output/MMDDHHmm_<mode>/`，不產生 `attempt/`，也不刪除歷史輸出。

## 主要變更

### Marker mode

- CLI：`digital-pdf-toolkit marker <PDF> [--config ...]`，並新增 Windows/Linux wrapper。
- 建立獨立 `digital-pdf-marker` Conda environment，安裝 `marker-pdf>=2,<3` 及本專案 editable package。
- Core runner 透過該環境執行 Marker worker，沿用現有 command、environment、status、log及硬體 metadata 紀錄。
- Marker worker 使用 `PdfConverter.build_document()` 分析一次，再以官方 `JSONRenderer` 與 `MarkdownRenderer` 渲染同一份 Document；不啟用 `use_llm`、`force_ocr` 或自訂 processors，mode 由 Marker 官方依裝置決定。
- 輸出維持 Marker 官方內容與圖片檔名：

```text
output/MMDDHHmm_marker/
├─ result.json
├─ result.md
├─ result_meta.json
├─ <Marker 原生 Picture/Figure/Diagram 圖片>
├─ block_provenance.json
├─ command.json
├─ environment.json
├─ status.json
├─ metadata/
├─ stdout.log
├─ stderr.log
├─ run.log
└─ events.jsonl
```

- `block_provenance.json` 逐一記錄文字 block 的 `block_id`、`block_type`、`page_id`、Marker 原始 `text_extraction_method`，並將 `surya` 明確標為 `ocr`；不得依內容猜測來源。結構型 block 若 Marker 沒有來源值，忠實保存 `null`。
- 不建立額外 crop、assets index、normalized schema或 embedding records。

### PNG-to-PDF

- 新增獨立命令：

```text
png-to-pdf <PNG資料夾>
png-to-pdf <PNG資料夾> --output <PDF路徑>
```

- 只讀取頂層 `.png`，忽略其他檔案，以既有 natural sorting 決定頁序。
- 使用現有 PyMuPDF，將每張 PNG 原尺寸轉為一頁並合併；不 OCR、不重新排序、不修改來源圖片。
- 預設輸出 `<輸入資料夾>/<資料夾名稱>.pdf`；拒絕空資料夾、非資料夾輸入及覆寫既有輸出。
- 新增 Windows/Linux wrapper，介面與 `heic-to-png` 相同。

### 設定與文件

- 新增 `config/marker.yml`，保存 environment、Marker version與 worker command。
- Config loader合併 Marker 預設設定；既有 `extract`、`surya2` 行為保持不變。
- README 更新為四類 command，補充 Marker與 PNG-to-PDF 的輸入、輸出及環境建置方式。

## 驗證

- CLI 能解析 `extract`、`surya2`、`marker`；`png-to-pdf` 維持獨立 optional command。
- PNG 測試驗證 natural order、PDF 頁數、各頁尺寸、忽略非 PNG、拒絕空輸入與拒絕覆寫。
- Marker runner 以替身程序驗證命令、成功／失敗狀態、root-level輸出及完全沒有 `attempt/`。
- Marker worker 以最小測試 Document 驗證 JSON、Markdown、原生圖片及 `block_provenance.json`，並確認 `pdftext`／`surya→ocr` 映射。
- 最後只執行上述相關測試，不跑完整測試套件。

## 固定假設

- Marker 僅接受單一 PDF；PNG 必須先使用 `png-to-pdf`。
- Marker raw output 不被改寫。
- 圖片只採 Marker 官方抽出的 `Picture/Figure/Diagram`。
- 暫不實作 embedding、table/list normalization、額外 block crop或 VLM caption。
