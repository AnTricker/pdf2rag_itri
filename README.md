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

詳細流程、schema、判斷樹與偽碼請見 docs/REVIEWABLE_PIPELINE.md。
Chat 會先用 `corpus_profile.json` 做多題拆分與 `document_question`、`out_of_scope`、`security_request` 語意 routing；後端驗證每個原始問題都忠實來自最新輸入，失敗只重試一次。一般問題使用 `focused` retrieval；整份文件概述使用語意 hits 加跨頁／章節代表 Records 的 `overview` retrieval。圖片題至少需要一筆合格 Image Record，圖文問題使用 `hybrid`。Answer JSON 無效或引用未知 Record 時只重試一次；仍未受文件證據支持則由後端直接回覆固定文字。

`runtime/logs/chat_sessions/<YYYY-MM-DD_HH-mm-ss-ffffff>.jsonl` 以結構化 `input`／`output` 記錄 preprocessor、LLM、query embedding、retrieval、evidence gate、answer、rendering、history 與 final response；包含 backend/model、attempt、timing、retrieval rank、頁碼、score 與 content preview。固定 prompt/system instructions、secret 與 embedding vector 不落盤。按「結束對話」後會另產生解析 raw model JSON 並移除所有機器識別欄位的同名 `.pretty.json`，清除 server history並輪替新 session ID。

瀏覽 `http://127.0.0.1:8000`。MVP 不支援 hot reload；建立新 build 後需重啟 serve 才會載入。原始 PDF 不複製到 output，也沒有公開檢視 route。

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