"""使用內部 ATEN TTS REST API 產生 MP3 與 SRT。"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests


DEFAULT_BASE_URL = "http://140.96.30.108/api/v1"
DEFAULT_VOICE = "Easton_news"
DEFAULT_LANG_TYPE = "TL"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputfolder"
TERMINAL_ERROR_STATES = {"Error", "Failed"}


def load_api_token(token_file: Path | None) -> str:
    """優先從環境變數讀取 token，其次讀取指定檔案。"""
    token = os.environ.get("TTS_API_TOKEN", "").strip()
    if token:
        return token

    if token_file and token_file.is_file():
        token = token_file.read_text(encoding="utf-8").strip()
        if token:
            return token

    raise RuntimeError(
        "找不到 API token；請設定環境變數 TTS_API_TOKEN，"
        "或使用 --token-file 指定 token 檔案。"
    )


def build_ssml(text: str, voice: str, lang_type: str) -> str:
    """建立 SSML，並跳脫使用者文字中的 XML 保留字元。"""
    safe_text = html.escape(text, quote=True)
    safe_voice = html.escape(voice, quote=True)
    safe_lang_type = html.escape(lang_type, quote=True)
    return (
        "<speak xmlns='http://www.w3.org/2001/10/synthesis' "
        "version='1.5' xml:lang='zh-TW'>"
        f"<voice name='{safe_voice}'>"
        f"<lang lang_type='{safe_lang_type}'>{safe_text}</lang>"
        "</voice></speak>"
    )


class LocalTTSClient:
    def __init__(self, base_url: str, api_token: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/") + "/"
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": api_token,
                "Content-Type": "application/json",
            }
        )

    def _request(self, method: str, path_or_url: str, **kwargs) -> requests.Response:
        url = self._resolve_url(path_or_url)
        try:
            response = self.session.request(
                method, url, timeout=self.timeout, **kwargs
            )
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            detail = ""
            if getattr(exc, "response", None) is not None:
                detail = f"；回應內容：{exc.response.text[:500]}"
            raise RuntimeError(f"API 請求失敗：{method} {url}：{exc}{detail}") from exc

    def _resolve_url(self, path_or_url: str) -> str:
        parsed = urlparse(path_or_url)
        if parsed.scheme in {"http", "https"}:
            return path_or_url
        return urljoin(self.base_url, path_or_url.lstrip("/"))

    def list_models(self) -> list[dict]:
        response = self._request("GET", "models/api_token")
        data = response.json()
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            return data["data"]
        raise RuntimeError(f"無法識別模型列表回應：{data}")

    def synthesize(
        self,
        text: str,
        voice: str,
        lang_type: str,
        name: str,
        audio_format: str = "mp3",
    ) -> dict:
        payload = {
            "name": name,
            "ssml": build_ssml(text, voice, lang_type),
            "silence_scale": 1.0,
            "is_customized_poly_list_used": False,
            "customized_poly_list": [],
            "audio_format": audio_format,
            "is_for_partial_test": False,
            "action": "create",
        }
        response = self._request("POST", "syntheses/api_token", json=payload)
        result = response.json()
        if not isinstance(result, dict) or not result.get("synthesis_id"):
            raise RuntimeError(f"合成 API 未回傳 synthesis_id：{result}")
        return result

    def wait_until_complete(
        self, synthesis_id: str, poll_interval: float, max_wait: float
    ) -> dict:
        deadline = time.monotonic() + max_wait
        last_status = None

        while time.monotonic() < deadline:
            response = self._request(
                "GET", f"syntheses/{synthesis_id}/api_token"
            )
            result = response.json()
            status = result.get("status")

            if status != last_status:
                print(f"合成狀態：{status or 'Unknown'}")
                last_status = status

            if status == "Success":
                return result
            if status in TERMINAL_ERROR_STATES:
                raise RuntimeError(f"語音合成失敗：{result}")

            time.sleep(poll_interval)

        raise TimeoutError(f"等待合成完成超過 {max_wait:g} 秒")

    def download(self, url: str, destination: Path) -> Path:
        response = self._request("GET", url, stream=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as output:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    output.write(chunk)
        return destination

    def download_bytes(self, url: str) -> bytes:
        """下載合成音檔至記憶體，供 Web API 直接回傳。"""
        return self._request("GET", url).content


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", value).strip("._")
    return cleaned or "local_tts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="透過 140.96.30.108 的 REST API 產生 MP3 與 SRT。"
    )
    parser.add_argument("text", nargs="*", help="要合成的文字；每個參數產生一組檔案")
    parser.add_argument("--voice", default=DEFAULT_VOICE, help="聲優 model_id")
    parser.add_argument(
        "--lang-type",
        default=DEFAULT_LANG_TYPE,
        choices=("TW", "EN", "TL", "TB"),
        help="文字的主要語言類型",
    )
    parser.add_argument("--name", default="local_tts", help="合成任務與輸出檔名前綴")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="輸出目錄"
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="內部 API base URL")
    parser.add_argument(
        "--token-file",
        type=Path,
        default=Path(__file__).resolve().parent / "API_TOKEN.txt",
        help="API token 檔；環境變數 TTS_API_TOKEN 的優先權較高",
    )
    parser.add_argument("--poll-interval", type=float, default=1.0, help="輪詢秒數")
    parser.add_argument("--max-wait", type=float, default=300.0, help="最長等待秒數")
    parser.add_argument(
        "--list-models", action="store_true", help="列出此 token 可用的模型後結束"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        token = load_api_token(args.token_file)
        client = LocalTTSClient(args.base_url, token)

        if args.list_models:
            print(json.dumps(client.list_models(), ensure_ascii=False, indent=2))
            return 0

        if not args.text:
            raise ValueError("請提供要合成的文字，或使用 --list-models")

        prefix = safe_filename(args.name)
        results = []
        for index, text in enumerate(args.text):
            file_prefix = prefix if len(args.text) == 1 else f"{prefix}_{index:03d}"
            print(f"開始合成：{file_prefix}")
            created = client.synthesize(
                text=text,
                voice=args.voice,
                lang_type=args.lang_type,
                name=file_prefix,
            )
            completed = client.wait_until_complete(
                created["synthesis_id"], args.poll_interval, args.max_wait
            )

            synthesis_path = completed.get("synthesis_path") or created.get(
                "synthesis_path"
            )
            srt_path = completed.get("srt_path") or created.get("srt_path")
            if not synthesis_path or not srt_path:
                raise RuntimeError(f"API 未提供音檔或字幕 URL：{completed}")

            mp3_path = client.download(
                synthesis_path, args.output_dir / f"{file_prefix}.mp3"
            )
            subtitle_path = client.download(
                srt_path, args.output_dir / f"{file_prefix}.srt"
            )
            result = {
                "text": text,
                "voice": args.voice,
                "lang_type": args.lang_type,
                "synthesis_id": created["synthesis_id"],
                "mp3_path": str(mp3_path.resolve()),
                "srt_path": str(subtitle_path.resolve()),
            }
            results.append(result)
            print(f"完成：{mp3_path}")
            print(f"字幕：{subtitle_path}")

        manifest = args.output_dir / f"{prefix}_results.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"結果清單：{manifest}")
        return 0
    except (RuntimeError, TimeoutError, ValueError) as exc:
        print(f"錯誤：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
