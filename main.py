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
    health_thread = threading.Thread(target=run_health_server, daemon=True)
    health_thread.start()
    api_id = os.getenv("TELEGRAM_API_ID")
    api_hash = os.getenv("TELEGRAM_API_HASH")
    bot_token = os.getenv("BOT_TOKEN")

    session_path = Path(__file__).parent / "fake_student_bot_session"
    client = TelegramClient(str(session_path), api_id, api_hash)
    await client.start(bot_token=bot_token)
    me = await client.get_me()
    print(f"fake student bot @{me.username} is running - Ctrl+C to stop")

    @client.on(events.NewMessage)
    async def handler(event):
        text = event.raw_text or ""
        lower = text.lower()
        await asyncio.sleep(2)

        result = await handle_message(event, None)
        if isinstance(result, dict) and "answer" in result:
            await event.respond(json.dumps(result))
        else:
            await event.respond("ok")

    await client.run_until_disconnected()


if __name__ == "__main__":
    asyncio.run(main())
