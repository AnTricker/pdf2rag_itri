# 地端TTS語音轉換

### 設定內網，確認連接
```powershell
IP address: 140.96.30.xxx # xxx!=108 否則會衝突到TTS主機端
Mask: 255.255.255.0
Gate: 140.96.30.254
DNS: 140.96.216.194
```
---

### 運行

將api key放入`API_TOKEN.txt`
(api key需進入 http://140.96.30.108 網站獲取)

設定環境(本案使用miniconda)
```cmd
conda create -n {你的環境} python=3.11 -y

conda activate {你的環境}

pip install -r TTS_requirement.yml
```

執行指令，輸入欲轉換的文本

```cmd
python .\local_tts.py "歡迎使用語音合成服務" --voice Easton_news --lang-type TL --name local_test
```
