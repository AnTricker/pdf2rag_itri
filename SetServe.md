# Set chatbot serve on Linux
### Permission required
- 步驟 1「取得 Conda Python 路徑」：你的普通帳號
- 步驟 2「建立 EnvironmentFile」：root
- 步驟 3「確認 runtime 權限」：先用普通帳號；權限不足才由 root 處理
- 步驟 4「建立 systemd unit」：root
- 步驟 5「啟用與啟動」：root
- 步驟 6「驗證」：普通帳號

## 1. 取得 Conda Python 路徑

```bash
conda run -n pdf2rag-amd python -c "import sys; print(sys.executable)"
```

記下輸出，以下以 `<CONDA_PYTHON>` 表示。

## 2. 建立 root-only EnvironmentFile

```bash
sudo install -d -m 700 /etc/local-rag
sudoedit /etc/local-rag/local-rag.env
```

填入：

```text
RAG_OUTPUT=<OUTPUT_TIMESTAMP>
LOCAL_RAG_SESSION_SECRET=<RANDOM_SECRET>
LOCAL_RAG_ADMIN_PASSWORD="<ADMIN_PASSWORD>"
LOCAL_RAG_WEB_HOST=0.0.0.0
LOCAL_RAG_WEB_PORT=<PORT>
```

產生 random secret：

```bash
openssl rand -hex 32
```

設定權限：

```bash
sudo chown root:root /etc/local-rag/local-rag.env
sudo chmod 600 /etc/local-rag/local-rag.env
```

## 3. 確認 runtime 寫入權限

```bash
mkdir -p <REPO_PATH>/runtime/logs
```

確認 `<LINUX_USER>` 對 `<REPO_PATH>/runtime` 有寫入權限。

## 4. 建立 systemd unit

```bash
sudoedit /etc/systemd/system/local-rag.service
```

內容：

```ini
[Unit]
Description=Local PDF RAG Web Service
Wants=network-online.target
After=network-online.target ollama.service
Requires=ollama.service
StartLimitIntervalSec=300
StartLimitBurst=3

[Service]
Type=simple
User=<LINUX_USER>
WorkingDirectory=<REPO_PATH>
EnvironmentFile=/etc/local-rag/local-rag.env
Environment=PYTHONUNBUFFERED=1
ExecStart=<CONDA_PYTHON> <REPO_PATH>/main.py serve --output ${RAG_OUTPUT}

Restart=on-failure
RestartSec=15
TimeoutStopSec=60
UMask=0077

[Install]
WantedBy=multi-user.target
```

## 5. 啟用並啟動

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now local-rag.service
```

查看狀態：

```bash
sudo systemctl status local-rag.service
```

查看即時 logs：

```bash
sudo journalctl -u local-rag.service -f
```

## 6. 驗證

內網 client：

```text
http://<SERVER_IP>:<PORT>/
```

健康檢查：

```bash
curl http://127.0.0.1:<PORT>/api/health
```

Admin 僅能在 Linux server 本機開啟：

```text
http://127.0.0.1:<PORT>/admin/monitoring
```

## 更新 build

修改：

```bash
sudoedit /etc/local-rag/local-rag.env
```

更新 `RAG_OUTPUT` 後：

```bash
sudo systemctl restart local-rag.service
sudo journalctl -u local-rag.service -n 100 --no-pager
```

## pipeline後端系統更新

Root 帳號重啟服務：

```bash
systemctl restart local-rag.service
systemctl status local-rag.service
```

普通帳號在瀏覽器執行 hard refresh：

```text
Ctrl + Shift + R
```

只有修改 `static/app.js` 或 CSS 時通常不用重啟，但可能有 browser cache；修改 HTML template 時建議重啟。