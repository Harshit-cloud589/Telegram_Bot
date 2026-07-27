# logging/logger.py
import asyncio
import json
import os
import tempfile
import threading
import time

_log_lock = asyncio.Lock()

from google.cloud import storage

BUCKET_NAME = "your-bot-logs-bucket"
OBJECT_NAME = "run.jsonl"
LOG_URL = f"https://storage.googleapis.com/{BUCKET_NAME}/{OBJECT_NAME}"
creds_json = os.environ.get("GCS_CREDENTIALS_JSON")
if creds_json:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(creds_json)
        cred_path = f.name
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path
_storage_client = storage.Client()  # uses GOOGLE_APPLICATION_CREDENTIALS env var
_bucket = _storage_client.bucket(BUCKET_NAME)
_lock = (
    threading.Lock()
)  # GCS has no native append; we read-modify-write, so serialize access


def _append_log_lines_sync(lines: str):
    blob = _bucket.blob(OBJECT_NAME)
    try:
        existing = blob.download_as_text()
    except Exception:
        existing = ""
    blob.upload_from_string(
        existing + lines + "\n", content_type="application/x-ndjson"
    )


async def append_log_lines(entries: list[dict]):
    """Append a batch of JSONL entries to the public log object."""
    if not entries:
        return
    lines = "\n".join(json.dumps(e, default=str) for e in entries)
    async with _log_lock:
        with _lock:
            blob = _bucket.blob(OBJECT_NAME)
            try:
                existing = blob.download_as_text()
            except Exception:
                existing = ""  # object doesn't exist yet — first write

            new_content = existing + (lines + "\n")
            blob.upload_from_string(new_content, content_type="application/x-ndjson")
            await asyncio.to_thread(_append_log_lines_sync, lines)

    print(f"[LOG] appended {len(entries)} lines to {LOG_URL}")


def make_logger(chat_id: int, qid: str = None) -> tuple[callable, list]:
    """Returns (log_fn, buffer) — log_fn appends to buffer in-memory during the run;
    call flush_log(buffer) once the run completes."""
    buffer = []

    def log_fn(entry: dict):
        entry["ts"] = time.time()
        entry["chat_id"] = chat_id
        if qid:
            entry["qid"] = qid
        buffer.append(entry)
        print("[LOG]", entry)

    return log_fn, buffer


def flush_log(buffer: list):
    try:
        append_log_lines(buffer)
    except Exception as e:
        print(f"[LOG ERROR] failed to upload: {e}")
        # Don't raise — the bot must still reply. Log locally as a fallback.
        with open("fallback_log.jsonl", "a") as f:
            for entry in buffer:
                f.write(json.dumps(entry, default=str) + "\n")
    finally:
        buffer.clear()
