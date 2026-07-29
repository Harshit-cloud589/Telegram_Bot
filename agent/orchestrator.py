import os
import time
from typing import Any

import google.genai as genai
from agent.formatter import format_final_answer
from google.genai import types
from logger import LOG_URL

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_NAME = "gemini-3.1-flash-lite"
import json
import time

from agent.tools_web import fetch_table_from_url, run_python, web_fetch

SYSTEM_PROMPT = """You are a data-analysis agent. You receive a data-analysis question via Telegram.

The question will specify a required JSON reply shape like:
{"answer": <something>, "log_url": "<url>"}

Rules:
1. Do NOT output the full object with "answer" and "log_url" keys yourself.
2. Output ONLY the VALUE that should go inside "answer" — matching the exact type/shape
   requested (e.g. if it asks for {"answer": {"state": "..."}}, output just {"state": "..."};
   if it asks for {"answer": <number>}, output just the number, e.g. 4).
3. Use tools to fetch real data and compute exact answers — never guess numbers.
4. Output nothing else — no markdown fences, no commentary, no surrounding envelope.

When a question references MOSPI (Ministry of Statistics and Programme Implementation) data:
1. Prefer these official entry points over generic search results:
   - https://www.mospi.gov.in/ (main site)
   - https://mospi.gov.in/publication (publications list)
   - https://www.mospi.gov.in/press-release (latest press releases with headline stats)
   - https://data.gov.in (many MOSPI datasets are also mirrored here with structured CSV/API access)
2. MOSPI data is often released as PDF or Excel files, not HTML tables. If a fetched page
   is a listing/index page, look for the actual download link (.xls, .xlsx, .pdf, .csv) and
   fetch that file directly rather than trying to parse the index page's HTML.
3. Prefer the PRIMARY MOSPI release over secondary news articles reporting on it — news
   articles can be delayed, rounded, or incorrect.
4. Key MOSPI datasets and where to find them:
   - National Accounts Statistics / GDP data → mospi.gov.in, "National Accounts Statistics" section
   - Periodic Labour Force Survey (PLFS) → mospi.gov.in, "PLFS" section, quarterly/annual bulletins
   - Consumer Price Index (CPI) → mospi.gov.in, "Price Statistics" section
   - Annual Survey of Industries (ASI) → mospi.gov.in, "Industrial Statistics" section
5. If you cannot locate exact primary data after 2-3 search attempts, state your best estimate
   clearly is uncertain rather than fabricating a precise-looking number.
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

    return format_final_answer(
        raw_text, original_question=messages[-1]["text"], log_url=LOG_URL
    )
