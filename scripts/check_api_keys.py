"""Check that the API credentials work before spending money on a real run.

Exercises the three network paths the batch and OpenRouter runners depend on,
with the smallest possible request for each:

  1. OpenRouter chat completion  -- auth, model slug, served provider
  2. Gemini generateContent      -- auth, and whether modelVersion comes back
     (the panel sets `model_version_capture: required`)
  3. Gemini File API upload      -- the resumable upload used by
     run_gemini_batch.py submit, which no offline test can cover

No images are sent and output is capped, so the whole check costs a fraction
of a cent. It creates no batch job. Keys are never printed.

Usage:
  cd Coding && python scripts/check_api_keys.py
  python scripts/check_api_keys.py --openrouter-model mistralai/ministral-8b-2512
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import urllib.error
import urllib.request

_here = Path(__file__).resolve()
_root = next((p for p in _here.parents if (p / "analysis").is_dir()), _here.parent)
sys.path.insert(0, str(_root))

from analysis.backends import api as api_backend

ENV_PATH = str(_root / ".env")
BASE = "https://generativelanguage.googleapis.com"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
PROBE = "Reply with the single word: ready"


def mask(value: str) -> str:
    """Never print a credential: prefix and length only."""
    return f"{value[:4]}…{len(value)} chars" if value else "MISSING"


def _request(url: str, data, headers, method="POST") -> bytes:
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def _fail(label: str, exc: Exception) -> bool:
    detail = ""
    if isinstance(exc, urllib.error.HTTPError):
        try:
            detail = exc.read().decode("utf-8", "replace")[:300]
        except Exception:
            detail = ""
        print(f"  {label}: FAIL HTTP {exc.code} {detail}")
    else:
        print(f"  {label}: FAIL {type(exc).__name__}: {str(exc)[:200]}")
    return False


def check_openrouter(model: str) -> bool:
    key = api_backend.get_api_key("OPENROUTER_API_KEY", ENV_PATH)
    print(f"\nOPENROUTER_API_KEY: {mask(key or '')}")
    if not key:
        return False
    payload = {"model": model,
               "messages": [{"role": "user", "content": PROBE}],
               "temperature": 0.0, "max_tokens": 16}
    try:
        resp = json.loads(_request(
            OPENROUTER_URL, json.dumps(payload).encode(),
            {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}))
    except Exception as exc:
        return _fail("chat completion", exc)
    if resp.get("error"):
        print(f"  chat completion: FAIL {str(resp['error'])[:200]}")
        return False
    usage = resp.get("usage", {}) or {}
    print(f"  chat completion : OK")
    print(f"    served_provider: {resp.get('provider')}")
    print(f"    served_model   : {resp.get('model')}")
    print(f"    tokens         : {usage.get('prompt_tokens')} in / "
          f"{usage.get('completion_tokens')} out")
    return True


def check_gemini_generate(model: str) -> bool:
    key = api_backend.get_api_key("GEMINI_API_KEY", ENV_PATH)
    print(f"\nGEMINI_API_KEY: {mask(key or '')}")
    if not key:
        return False
    payload = {"contents": [{"parts": [{"text": PROBE}]}],
               "generationConfig": {"temperature": 0.0, "maxOutputTokens": 16}}
    try:
        resp = json.loads(_request(
            f"{BASE}/v1beta/models/{model}:generateContent",
            json.dumps(payload).encode(),
            {"x-goog-api-key": key, "Content-Type": "application/json"}))
    except Exception as exc:
        return _fail("generateContent", exc)
    version = resp.get("modelVersion")
    meta = resp.get("usageMetadata", {}) or {}
    print(f"  generateContent : OK")
    print(f"    modelVersion   : {version or 'ABSENT'}"
          f"{'' if version else '   <-- panel requires this'}")
    print(f"    tokens         : {meta.get('promptTokenCount')} in / "
          f"{meta.get('candidatesTokenCount')} out")
    return bool(version)


def check_gemini_upload() -> bool:
    """Exercise the resumable upload that run_gemini_batch.py submit uses."""
    key = api_backend.get_api_key("GEMINI_API_KEY", ENV_PATH)
    if not key:
        return False
    blob = (json.dumps({"key": "probe",
                        "request": {"contents": [{"parts": [{"text": PROBE}]}]}})
            + "\n").encode()
    try:
        start = urllib.request.Request(
            f"{BASE}/upload/v1beta/files", data=b"", method="POST",
            headers={"x-goog-api-key": key,
                     "X-Goog-Upload-Protocol": "resumable",
                     "X-Goog-Upload-Command": "start",
                     "X-Goog-Upload-Header-Content-Length": str(len(blob)),
                     "X-Goog-Upload-Header-Content-Type": "application/jsonl",
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(start, timeout=60) as resp:
            upload_url = resp.headers.get("X-Goog-Upload-URL")
        if not upload_url:
            print("  file upload     : FAIL no X-Goog-Upload-URL header")
            return False
        body = _request(upload_url, blob,
                        {"Content-Length": str(len(blob)),
                         "X-Goog-Upload-Offset": "0",
                         "X-Goog-Upload-Command": "upload, finalize"})
        name = json.loads(body)["file"]["name"]
    except Exception as exc:
        return _fail("file upload", exc)
    print(f"  file upload     : OK ({name})")
    try:
        _request(f"{BASE}/v1beta/{name}", None,
                 {"x-goog-api-key": key}, method="DELETE")
        print(f"    cleaned up     : yes")
    except Exception:
        print(f"    cleaned up     : no (harmless; files expire on their own)")
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--openrouter-model", default="qwen/qwen3-vl-8b-instruct")
    p.add_argument("--gemini-model", default="gemini-2.5-pro")
    p.add_argument("--skip-upload", action="store_true")
    args = p.parse_args()
    print(f"Reading credentials from {ENV_PATH}")
    ok = [check_openrouter(args.openrouter_model),
          check_gemini_generate(args.gemini_model)]
    if not args.skip_upload:
        ok.append(check_gemini_upload())
    print("\n" + ("ALL CHECKS PASSED" if all(ok) else "SOME CHECKS FAILED"))
    return 0 if all(ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
