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
python main.py serve --output 2026-08-12_07-40-43-839095
~~~

### q3Importer：匯入既有 Qwen3-VL Knowledge Records

`q3Importer` 接受具備正式 Qwen3-VL artifacts 的 knowledge base。輸入可為
embedding worker 的 run 目錄或其中的 `knowledge_base/`；可重複提供 `--input`，
將多個知識庫整合成一個現行 `serve` 可載入的不可覆寫 build。Importer 不以
`schema_version` 或 `mode` 標籤阻擋可相容資料。

Importer 會強制驗證會影響搜尋正確性的 records、vectors 與代表 crop checksum，
並檢查 record/vector counts、vector rows、model/revision/dimension/normalization
與 preprocessing。不參與最終 build 的 metadata／overview checksum 會直接忽略；
只要 metadata 的 image ID、vector row 與 crop reference 一致即可。沒有向量的
`provenance_only` records 只列入報告，不進搜尋索引。Image record 必須透過
`source_image_id` 與 `embedding_inputs/metadata.json` 指向代表 crop；不支援舊版
直接使用 `metadata.crop` 的格式。

若直接指定 `knowledge_base/`，仍須保留同一 run 上層的 `resolved_config.json`；
除非 manifest 本身已包含 `preprocessing`。含圖片的來源必須保留 `crops/` 與
`embedding_inputs/metadata.json`；只有實際匯入的代表 crop 必須有 manifest checksum。

~~~bash
python main.py q3Importer \
  --input runtime/A_qwen3vl \
  --input runtime/B_qwen3vl \
  --output qwen3vl_combined \
  --name "ITRI Qwen3-VL Knowledge Collection" \
  --profile config/q3_profile.example.json

# --profile 可省略；省略時依 heading、type 與 record 統計產生 deterministic profile。
~~~

`--profile` 格式可參考 `config/q3_profile.example.json`，只能覆寫摘要、topics、
sections 等內容欄位；document ID、name、source 與 schema/version 由 importer 決定。

產物固定為：

~~~text
runtime/outputs/qwen3vl_combined/
├─ crops/
└─ builds/001/
   ├─ records.jsonl
   ├─ embeddings.npy
   ├─ corpus_profile.json
   ├─ manifest.json
   └─ build_report.json
~~~

Importer 沿用來源的預計算 vectors，不載入 embedding model 或呼叫 Ollama。
輸入 vectors 會依 record 順序合併並轉成 serve 使用的單一 float32 matrix；來源
record ID、KB ID、image ID、region IDs、source indexes 與 content types 會保留供
引用追蹤。
`--output` 若已存在會直接拒絕；來源或 profile 改變時請使用新的 output ID。

#### 部署機安裝 Qwen3-VL query model

Importer 只合併預先算好的 vectors，本身不需要載入模型；`serve` 才需要同一個
Qwen3-VL embedding model 產生 query vector。本專案使用本機離線載入，因此請先在
可連網的部署機下載完整模型：

~~~bash
conda activate pdf2rag-amd
python -m pip install -r requirements.txt

mkdir -p models
hf download Qwen/Qwen3-VL-Embedding-8B \
  --local-dir models/Qwen3-VL-Embedding-8B

# 確認至少存在模型設定；這一步不會載入模型。
test -f models/Qwen3-VL-Embedding-8B/config.json
~~~

模型來源為 [Qwen/Qwen3-VL-Embedding-8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B)，
下載方式可參考 [Hugging Face CLI 文件](https://huggingface.co/docs/huggingface_hub/guides/download#download-from-the-cli)。
若 `hf` 指令不存在，先執行：

~~~bash
python -m pip install --upgrade huggingface_hub
~~~

接著把 `.env` 的 model path 設為下載目錄的絕對路徑。CUDA 與 ROCm PyTorch 都使用
`cuda` device 名稱：

~~~dotenv
LOCAL_RAG_EMBEDDING_MODEL=/absolute/path/to/Qwen3-VL-Embedding-8B
LOCAL_RAG_EMBEDDING_DEVICE=cuda
LOCAL_RAG_EMBEDDING_DTYPE=float16
LOCAL_RAG_EMBEDDING_ATTENTION=sdpa
LOCAL_RAG_EMBEDDING_BATCH_SIZE=1

LOCAL_RAG_LLM_URL=http://localhost:11434
LOCAL_RAG_LLM_MODEL=<Ollama回答模型>
LOCAL_RAG_SESSION_SECRET=<隨機長字串>
LOCAL_RAG_ADMIN_PASSWORD=<管理密碼>
~~~

若主機上已經有完整的同一模型，不必重複下載，直接填該模型目錄的絕對路徑即可。

#### 32GB 單一內顯／UMA 記憶體注意事項

本 serve 會讓 SentenceTransformer 的 Qwen3-VL embedding model 常駐，Ollama 在
回答期間也會載入 Gemma。若 Gemma 實測約 18GB、Qwen 約 10GB 以上，兩者權重就已
接近 28–30GB；再加上 KV cache、activations、ROCm allocator 與系統共用記憶體，
32GB 單一 memory pool 高機率 OOM。

優先方案：

1. 若系統 RAM 足夠，將 query embedding 放在 CPU，避免占用 GPU/UMA 顯存：

   ~~~dotenv
   LOCAL_RAG_EMBEDDING_DEVICE=cpu
   LOCAL_RAG_EMBEDDING_DTYPE=auto
   LOCAL_RAG_EMBEDDING_BATCH_SIZE=1
   ~~~

   Query embedding 會變慢，但每次只處理少量 query，通常比兩個大型模型同駐安全。

2. 若有第二張 GPU，將 Qwen 指定到另一裝置，例如 `cuda:1`。

3. 必須共用 32GB GPU 時，降低 Gemma context，並同步讓 input budget 小於 context：

   ~~~dotenv
   LOCAL_RAG_LLM_CONTEXT_TOKENS=8192
   LOCAL_RAG_ANSWER_INPUT_BUDGET_TOKENS=6144
   ~~~

   這只能降低 KV cache，無法消除兩套模型權重同時存在的峰值；若仍 OOM，應改用
   CPU query embedding、較小的回答模型或增加可用記憶體。可用 `ollama ps` 查看
   Gemma 的 `PROCESSOR` 與實際 context/offload 狀態。

例如 repository 位於 `/home/r300_465035/文件/pdf2rag_itri`，則 model path 應為：

~~~dotenv
LOCAL_RAG_EMBEDDING_MODEL=/home/r300_465035/文件/pdf2rag_itri/models/Qwen3-VL-Embedding-8B
~~~

確認 Ollama 已在部署主機啟動後：

~~~bash
python main.py serve --output qwen3vl_combined
~~~

瀏覽 `http://127.0.0.1:8000`。若 query model dimension 與 build 不一致，serve
會在啟動階段明確失敗，不會等到第一次查詢才報錯。

部署機可執行下列驗收；開發 code 機禁止執行：

~~~bash
python -m unittest \
  tests.test_q3_importer \
  tests.test_embedding_runtime \
  tests.test_rendering \
  tests.test_http_chat

python main.py q3Importer \
  --input runtime/<run>_qwen3vl \
  --output q3_smoke \
  --name "Q3 Smoke Test"

python main.py serve --output q3_smoke
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
- Serve 啟動時不預載 LLM，Web 會立即開放；首次實際需要 Planner／Answer 時才按需載入模型。所有 LLM request 使用 `LOCAL_RAG_LLM_KEEP_ALIVE`（預設 `15m`），最後一次推論後由 Ollama 自動卸載；冷啟動 timeout 由 `LOCAL_RAG_LLM_STARTUP_TIMEOUT_SECONDS`（預設 300 秒）控制。部署前必須設定 `LOCAL_RAG_SESSION_SECRET` 與 `LOCAL_RAG_ADMIN_PASSWORD`，AMD 單 GPU 建議以 `OLLAMA_NUM_PARALLEL=1` 啟動 Ollama。

- Chat 使用一次 Query Planner 完成拆題、routing 與 retrieval queries，再由程式檢索，最後以最少必要的 Answer batches 產生回答。一般單題只需兩次 LLM 呼叫。Planner 失敗不重試；Answer schema 或引用驗證失敗才重試一次，HTTP timeout 不重試。

- 前端以 server-signed cookie 識別 browser access session，各分頁另以 `sessionStorage` 保存獨立 chat session。透過 job queue 與 SSE 在對話框內顯示排隊、模型冷啟動、規劃、檢索、圖片分析及回答階段。冷啟動時會顯示累計等待秒數，模型 ready 後自動進入規劃。每次 QA 可選擇或直接貼上最多三張 JPEG/PNG/WebP；附件只用於當次 QA，完成後刪除，不加入知識庫。Enter 可直接送出，Shift+Enter 換行。

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
