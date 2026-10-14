# develop DIARY

### 918
- 完成 `qwen3vl` embedding knowledge-base pipeline；直式文件、橫式文件與 PPT 共用相同 schema，reading order 僅用於文字排列與超長頁切割。
- Surya text/HTML syntax 經 BeautifulSoup 產生 `raw_html`、`plain_text`、model-neutral `embedding_text`；頁面為主要 chunk，table/form 可獨立，heading stack 跨頁延續，重複 header/footer 聚合為 `document_root`。
- 圖片 routing 定案：有 OCR 文字只建立 `text_vector`、不保存 crop；完全無文字且 crop 可讀才建立 `image_vector` 並保存圖片；error 或無可用 crop 僅保留 provenance。不做 chart 特判、caption 或 text+image joint vector。
- 純圖先做跨頁 pHash complete-link 去重；至少 3 頁重複時只保留一張依面積、清晰度、頁碼、reading order、ID 決定的代表圖，同時保存全部來源 metadata。
- 知識庫輸出包含 `records.jsonl`、`text.npy`、`image.npy`、原始代表 crop、實際 resize 後的 embedding input 與 `metadata.json`；`vector_ref.row` 對應 NumPy matrix row，圖片可由 `source_image_id` 回查原 crop。
- 模型、runtime 與 image pixel budget 由 `embedding.yml`／`local.yml` 解耦；目前採 `Qwen/Qwen3-VL-Embedding-8B`，本機開發不執行真實模型推論。
- 新增 `knowledge_base.failed/` 原地 resume、嚴格 checkpoint 驗證、GPU synchronize 與分次 log；實測可由 335 筆接續至 676 筆，既有成功 vectors 與 embedding inputs 不重算。
- 持續定位 ROCm/MIOpen 長時間 process 約 340 張後的 Qwen3VL patch-embedding Conv3D launch failure；下一步為 worker 定期重啟，`max_images_per_process: 250` 目前僅預留設定。
- README 已補齊 qwen3vl 設定、routing、輸出結構、resume 與知識庫讀取方式；retrieval、vector DB、API 與前端仍不在目前範圍。

### 827 



### 826 
- v0, up to github 
