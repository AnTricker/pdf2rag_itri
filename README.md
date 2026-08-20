# Local PDF RAG MVP

獨立、可搬移的 localhost PDF 問答 pipeline。每次 ingest 以 timestamp output 隔離；人工確認 pretty JSON 後才 build 正式 records.jsonl、embeddings.npy、corpus_profile.json 與 manifest.json。crop artifacts 只保存一份，不使用 MCP 或 Qdrant。
## Setup
> 測試機端(nvi)可直接按照下方設置

> amd端 建議另設 'pdf2rag-amd'環境，並參考最下方補充之設定步驟

```powershell
conda activate pdf2rag
cd C:\ITRI\PDF2json\local_rag_mvp
 
python -m pip install -r requirements.txt
 
conda install -c conda-forge poppler
 
Copy-Item .env.example .env
```

編輯 `.env`，至少設定本機 embedding model 路徑與 Ollama LLM model。若要建立圖片 Record，再設定 Poppler、detector model 與 VLM model；未設定 detector 時仍可跑純文字 ingestion。

#### poppler
```bash
### 確認poppler安裝
which pdfinfo
pdfinfo -v
### 將輸出路徑 .../bin 貼到 .env > LOCAL_RAG_POPPLER_PATH
```

## Commands

~~~powershell
python main.py doctor

# Stage 1：建立新的 timestamp output；只產生 crops 與第一版 pretty
python main.py ingest --input input\a.pdf --mode multi

# Stage 2：補齊指定 output 最新 pretty 的缺少欄位
python main.py review --output 2026-08-11_15-30-45-123456

# Stage 3：以指定 pretty revision 建立正式 RAG build
python main.py build --output 2026-08-11_15-30-45-123456 --review 002

# 使用指定 output 中最新的有效 build
python main.py serve --output 2026-08-11_15-30-45-123456
~~~

text、image、multi mode 分別控制 Stage 1 的文字與圖片支線。Stage 1/2 永遠不更新正式 RAG；只有 build 會在 output 內建立版本化索引。人工只修改 pending/NNN_records.pretty.json 中標示為 editable 的欄位，不直接修改 JSONL。

### 人工審閱流程

1. 執行 `ingest`，記下 CLI 回傳的 `<timestamp>`。產物為 `crops/` 與 `pending/001_records.pretty.json`；若部分 VLM 任務失敗，CLI 會回傳非零狀態，但仍保留這些產物供補跑。
2. 檢查最新的 `pending/NNN_records.pretty.json`。只修改 `review_instructions.editable_fields` 列出的欄位；來源、crop path、detector 資訊與 schema version 不可修改。
3. 如有 `null` 的 classifier／extraction／summary 欄位，執行 `review`。它只補缺少欄位，保留人工內容；有變更時產生下一版 `pending/(NNN+1)_records.pretty.json`，可重複執行。
4. 可選擇產生方便逐圖檢查的 Markdown：

   ~~~powershell
   python scripts\generate_manual_caption_review.py --output <timestamp> --review NNN
   ~~~

   產物為 `runtime/outputs/<timestamp>/manual_caption_review_NNN.md`，內含 crop 引用與目前結構化 caption；它只供閱讀，人工修改仍以 pretty JSON 為準。

5. 確認指定 revision 後執行 `build --output <timestamp> --review NNN`。產生 `builds/NNN/`；`unusable` 或 caption 不完整的圖片不會進入正式 `records.jsonl`。

   可用圖片會依結構化 cell／text block 切成多個 schema 2.1 embedding Records；每筆保留完整 table／figure 結構並共享 crop。有效 token 上限取 `.env` 的 chunk size 與 embedding model 上限較小者，overlap 亦讀取 `.env`。`build_report.json` 分別記錄原始圖片數與 image chunk 數。

6. 執行 `serve --output <timestamp>`，載入該 output 編號最大的有效 build。新 build 完成後需重啟 serve。

~~~text
runtime/outputs/<timestamp>/
├─ crops/
├─ pending/NNN_records.pretty.json
└─ builds/NNN/
   ├─ records.jsonl
   ├─ embeddings.npy
   ├─ corpus_profile.json
   ├─ manifest.json
   └─ build_report.json
~~~
#### (8/15) update ```--serve```
- Serve 啟動時會先 warm LLM；warm 失敗時直接退出，成功後才由 Waitress 開放連線。部署前必須設定 `LOCAL_RAG_SESSION_SECRET` 與 `LOCAL_RAG_ADMIN_PASSWORD`，AMD 單 GPU 建議以 `OLLAMA_NUM_PARALLEL=1` 啟動 Ollama。

- Chat 使用一次 Query Planner 完成拆題、routing 與 retrieval queries，再由程式檢索，最後以最少必要的 Answer batches 產生回答。一般單題只需兩次 LLM 呼叫。Planner 失敗不重試；Answer schema 或引用驗證失敗才重試一次，HTTP timeout 不重試。

- 前端以 server-signed cookie 識別 browser access session，各分頁另以 `sessionStorage` 保存獨立 chat session。透過 job queue 與 SSE 在對話框內顯示排隊、規劃、檢索、圖片分析及回答階段。每次 QA 可選擇或直接貼上最多三張 JPEG/PNG/WebP；附件只用於當次 QA，完成後刪除，不加入知識庫。Enter 可直接送出，Shift+Enter 換行。

#### (8/18) update ```--serve```
- 空對話會顯示 CMP 操作規範歡迎訊息與 hardcode 提示問題。回答依正常查詢、超出範圍、安全限制及文件資訊不足呈現不同狀態；每個 QA block 可複製、按讚或倒讚。引用單頁顯示「第 n 頁」，跨頁顯示「第 n~m 頁」，引用到 image record 時會在參考資料顯示 crop。

- 「輸出對話報告」會依目前已完成的 QA 即時下載 Markdown，伺服器不保存 `.md`。報告包含 Session／模型／build metadata、問題、回答、feedback、附件 metadata 與引用，不包含 prompt、hidden thinking 或附件內容。回答進行中或沒有完整 QA 時無法輸出。

- `runtime/logs/chat_sessions/<timestamp>.jsonl` 保存 machine events；同名 `.pretty.json` 在每個重要事件後即時 atomic 更新，並依 `session`、`qa_blocks`、`session_actions`、`diagnostics` 分組。QA block 保存 feedback 與複製／提示問題操作時間，Session actions 保存輸出報告、清除及結束操作；log 排除完整 prompt、raw output、hidden thinking 與附件內容。

- 按「清除對話」會結束目前分頁的 chat session 並建立新對話。按「結束對話」會取消目前分頁的工作、完成 log 並嘗試關閉分頁；其他分頁不受影響。Refresh 會完成舊 chat log 並建立新 chat session。

- `runtime/logs/access_sessions/` 保存每個 browser access session 的 JSONL access log，`runtime/monitoring.sqlite3` 關聯 IP、access session、tab session 與 chat log path。每個 tab／refresh 都保留空／有狀態；只有送出第一個有效問題後才建立 chat JSON，避免保存無內容的 chat logs。唯讀查詢介面位於 `/admin/monitoring`，僅允許 server 本機並需管理密碼登入；IP 直接取自 `request.remote_addr`。

---
## AMD ROCm / PyTorch Setup Notes
This project was tested on an AMD GPU environment with ROCm 7.2.1 and Python 3.12.
 
#### 1. Install ROCm-specific Triton
The ROCm PyTorch wheel depends on AMD's patched Triton build, which is not available from the default PyPI index.
Download and install it manually:
```bash
wget "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/triton-3.5.1%2Brocm7.2.1.gita272dfa8-cp312-cp312-linux_x86_64.whl%22
 
pip install --no-cache-dir \
  ./triton-3.5.1+rocm7.2.1.gita272dfa8-cp312-cp312-linux_x86_64.whl
  ```
 
#### 2. Install ROCm PyTorch wheels
Because pip may attempt to resolve the ROCm-specific Triton dependency from PyPI and fail, install the local PyTorch wheels with``` --no-deps```:
```bash
pip install --no-cache-dir --no-deps \
  ./torch-2.9.1+rocm7.2.1.lw.gitff65f5bc-cp312-cp312-linux_x86_64.whl \
  ./torchvision-0.24.0+rocm7.2.1.gitb919bd0c-cp312-cp312-linux_x86_64.whl
  ```
Then install the remaining project dependencies normally:
```bash
pip install -r requirements.txt
```
Avoid using``` --force-reinstall ```on the ROCm PyTorch stack unless necessary, because pip may try to fetch the ROCm-specific Triton version from PyPI again.
 
#### 3. Configure ROCm device permissions
ROCm applications require access to ```/dev/kfd``` and ```/dev/dri/renderD*```.
 
Check current permissions:
```bash
groups
ls -l /dev/kfd /dev/dri/renderD*
rocminfo | head -50
```
If the current user is not in the render group, ask an administrator to run:
```bash
sudo usermod -aG render,video <username>
```
Afterward, completely log out and log back in so the new group membership takes effect.
 
Verify:
```bash
groups
rocminfo | head -50
```
#### 4. Verify PyTorch ROCm
```bash
python -c "import torch; print(torch.__version__); print(torch.version.hip); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```
Expected result:
```bash
torch.cuda.is_available() == True
```
ROCm PyTorch still uses the ```torch.cuda``` API for GPU access.
 
To check the detected GPU:
```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
```
If ```rocminfo``` works and ```torch.cuda.is_available()```returns ```True```, the ROCm runtime and PyTorch GPU environment are configured correctly.
