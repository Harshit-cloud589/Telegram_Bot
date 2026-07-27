print("!!!! PROCESS STARTED - BUILD MARKER v99 !!!!", flush=True)

import asyncio
import json
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # load TELEGRAM_API_ID/TELEGRAM_API_HASH from .env if present
from handlers import handle_message
from telethon import TelegramClient, events


class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"OK")

    def log_message(self, format, *args):
        # suppress noisy default logging for every ping
        pass


def run_health_server():
    port = int(os.getenv("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
    print(f"[health] listening on 0.0.0.0:{port}")
    server.serve_forever()


async def main():
    print("[step 1] starting health thread", flush=True)
    health_thread = threading.Thread(target=run_health_server, daemon=True)
    health_thread.start()
    print("[step 2] health thread started", flush=True)
    api_id = os.getenv("TELEGRAM_API_ID")
    api_hash = os.getenv("TELEGRAM_API_HASH")
    bot_token = os.getenv("BOT_TOKEN")
    print(
        f"[step 3] env check: api_id={bool(api_id)} api_hash={bool(api_hash)} bot_token={bool(bot_token)}",
        flush=True,
    )
    session_path = Path(__file__).parent / "fake_student_bot_session"
    print(f"[step 4] session path = {session_path}", flush=True)
    client = TelegramClient(str(session_path), api_id, api_hash)
    print("[step 5] client object created, calling start()...", flush=True)
    await client.start(bot_token=bot_token)
    print("[step 6] client.start() completed", flush=True)
    me = await client.get_me()
    print(f"[step 7] bot @{me.username} confirmed running", flush=True)

    @client.on(events.NewMessage)
    async def handler(event):
        text = event.raw_text or ""
        lower = text.lower()
        await asyncio.sleep(2)

        result = await handle_message(event, None)
        if isinstance(result, dict) and "answer" in result:
            await event.respond(json.dumps(result))
            print(f"[HANDLER DONE] replied: {result}", flush=True)
        else:
            print("[HANDLER ERROR]", flush=True)
            await event.respond("ok")

    print("[step 8] entering run_until_disconnected()", flush=True)
    await client.run_until_disconnected()


if __name__ == "__main__":
    asyncio.run(main())
