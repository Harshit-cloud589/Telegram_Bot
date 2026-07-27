import os
import time
from typing import Any

import google.genai as genai
from formatter import format_final_answer
from google.genai import types

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_NAME = "gemini-3.1-flash-lite"
import json
import time

from tools_web import fetch_table_from_url, run_python, web_fetch

SYSTEM_PROMPT = """You are a data-analysis agent. You receive a data-analysis question via Telegram.

Rules:
1. The question text specifies the EXACT JSON shape required for the final answer — extract it precisely (key names, nesting, types).
2. If the question embeds data inline, use it directly. If it references a public dataset (MOSPI, data.gov.in, etc.), use the web_fetch/fetch_table_from_url tools to locate and retrieve real data.
3. For ANY numeric computation, use the run_python tool. Never compute numbers by reasoning alone.
4. Once you have a grounded answer, respond with ONLY the final answer object matching the requested shape — no extra commentary, no markdown fences.
5. Your final response must be valid JSON matching exactly what was asked, nothing else.
"""

TOOLS = [web_fetch, fetch_table_from_url, run_python]


async def run_agent_and_format(
    messages: list[dict], timeout_seconds: int = 60, log_fn=None
) -> dict:
    start = time.monotonic()
    budget = timeout_seconds * 0.75  # leave margin for formatting + send

    # Build conversation history for Gemini
    contents = []
    for m in messages:
        role = "user" if m["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=m["text"])]))

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=TOOLS,  # SDK auto-builds function-calling schema from these Python fns
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            maximum_remote_calls=8,  # cap tool-call iterations
        ),
    )

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=config,
        )
        raw_text = response.text.strip()
    except Exception as e:
        raw_text = ""
        print(f"[AGENT ERROR] {e}")

    elapsed = time.monotonic() - start
    print(f"[AGENT] finished in {elapsed:.1f}s, raw output: {raw_text[:300]}")

    return format_final_answer(raw_text, original_question=messages[-1]["text"])
