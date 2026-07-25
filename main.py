import asyncio
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # load TELEGRAM_API_ID/TELEGRAM_API_HASH from .env if present
from telethon import TelegramClient, events


async def main():
    api_id = int(os.environ["TELEGRAM_API_ID"])
    api_hash = os.environ["TELEGRAM_API_HASH"]
    bot_token = os.environ["BOT_TOKEN"]

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

        result = {"answer": "test", "log_url": "https://example.com/run.jsonl"}
        await event.respond(json.dumps(result))

    await client.run_until_disconnected()


if __name__ == "__main__":
    asyncio.run(main())
