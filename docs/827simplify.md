# 專案簡化與 Surya2 自動化實作計畫

## 摘要

將專案收斂為三個唯一 command：`heic-to-png`、`extract`、`surya2`。完整移除 preprocess、pages、resolution、finalize、generic analyze、attempt、page-set、manifest、Paddle 與 MinerU；既有歷史 output 不搬移或刪除。

## 核心變更

- CLI／scripts：
  - `heic-to-png <folder> [--output <folder>]`
  - `extract <pdf> [--config <yaml>]`
  - `surya2 <png-folder|pdf> [--config <yaml>]`
  - Linux 與 Windows wrappers 同步更新；移除所有 `create/retry/report` 次層。
- Output run 命名改為 `output/MMDDHHmm_<mode>[_NN]/`，同分鐘撞名由 `_02` 起遞增。
- `extract` 將 native facts、page JSON、renders、embedded images、status 與 logs 直接寫入 run root；移除 `extract/attempt/`、`observed/`、page-set 與 manifest。
- `surya2` 直接執行 `surya_ocr`，移除 `analyze/attempt/` 與 `raw/`；將唯一的 `results.json` 正規化至 result root，保留 crops、overlays 與 `assets/index.json` 後處理。
- PDF 輸入先 render 至該 run 的 `rendered_png/`：
  - ≤10 頁：結果寫入 `result/`。
  - >10 頁：rendered images 分入 `rendered_png/1/…`，結果對應寫入 run root 的 `1/…`。
- PNG folder：
  - ≤10 張：結果直接寫入 run root。
  - >10 張：依 case-insensitive natural sort，直接移動至原 input 的 `1/、2/…`；每批 10 張，最後一批 1–10 張。
  - 再次執行既有 batches 時全部重跑；要求編號連續，每批允許 1–10 張。root 同時有 PNG 與數字 subfolders 時報錯。
- Batches 嚴格循序執行。單批失敗後繼續，其 `status.json` 記錄失敗；任一批失敗時整體 exit code 非 0，最後只額外顯示 completed／failed batch 統計。
- 建立 filename-based mapping，記錄原始 filename、global page index、batch 與 batch-local position。Crop、overlay、assets index 使用 global page index；缺頁、重複 stem 或不唯一 mapping 使該 batch 失敗。
- 將 metadata logger 移入 package，整合為 `surya2` 自動流程：
  - 每個 result／batch 各有自己的 `metadata/summary.json` 與 `samples.csv`。
  - Linux AMD 收集完整指標；其他平台或缺少 `amd-smi` 時以 null＋warning best-effort 記錄，不影響 Surya 成敗。
  - `extract` 不建立 metadata。
- 設定收斂為 `config/default.yml` 與 `config/surya2.yml`；custom YAML 可覆蓋預設。Batch size 固定為 10，其他調參只透過設定檔。
- 保留 `digital-pdf-core`、`digital-pdf-surya` environments；setup 僅支援這兩者。
- 完整刪除舊 modes、generic tool runner、page-set／attempt／manifest helpers、Paddle／MinerU adapters、configs、environments、fixtures、tests、scripts、profiles 與過時文件。

## 驗證

- CLI parsing：三個 command 均無 `create`，舊 commands 明確不可用。
- Output naming：時間格式、mode postfix 與同分鐘 `_02/_03`。
- Extract：輸出攤平、native facts 與 renders 正確，無 attempt／observed／metadata。
- Surya2 PNG：空 folder、≤10、>10、natural sort、原圖移動、既有 batch 重跑、混合／缺號／批量錯誤。
- Surya2 PDF：≤10 與 >10 的 render/result trees，以及 render DPI config。
- Batch failure：後續批次繼續、各自 logs/status/metadata、最終 exit code 與統計。
- Mapping：跨 batch global page index、任意 filename、缺頁與 duplicate stem。
- Metadata：完整採樣、缺少 `amd-smi`、Windows fallback、採樣失敗不干擾結果。
- Surya 後處理：`results.json` 位於 result root，crops／overlays 正確對應來源圖片。
- HEIC：維持原有 natural sort、EXIF orientation、`--output` 與非空 output 拒絕行為。
- 僅執行新流程相關的 targeted tests，不執行未另行核准的 full test suite 或 Git workflow。

## 固定假設

- Timestamp 使用執行主機本機時間，格式為 `MMDDHHmm`。
- Surya executable、package 與 conda environment 保留上游原名；本案公開 mode 與內部 orchestration 改名為 `surya2`。
- 非 PNG 檔案與非數字 subfolders 會被忽略；沒有可用 PNG 時報錯。
- 舊 output 不提供 migration 或相容讀取。
