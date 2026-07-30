import os
import time
from typing import Any

import google.genai as genai
from agent.formatter import format_final_answer
from google.genai import types
from groq import Groq
from logger import LOG_URL

# client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL_NAME = "llama-3.1-8b-instant"
import json
import time

from agent.tools_schema import TOOL_FUNCTIONS, TOOL_SCHEMAS
from agent.tools_web import with_timeout

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


async def run_agent_and_format(
    messages: list[dict], timeout_seconds: int = 60, log_fn=None
) -> dict:
    start = time.monotonic()
    deadline = start + (timeout_seconds * 0.75)

    chat_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    chat_messages += [{"role": m["role"], "content": m["text"]} for m in messages]

    final_text = ""

    for iteration in range(8):  # max tool-call iterations
        if time.monotonic() > deadline:
            if log_fn:
                log_fn({"event": "budget_exceeded", "iteration": iteration})
            break

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=chat_messages,
                tools=TOOL_SCHEMAS,
            )
        except Exception as e:
            err_str = str(e)
            if (
                "tool_use_failed" in err_str
                or "Failed to parse tool call arguments" in err_str
            ):
                if log_fn:
                    log_fn({"event": "tool_parse_retry", "error": err_str[:500]})
                chat_messages.append(
                    {
                        "role": "user",
                        "content": "Your last tool call had invalid JSON arguments. Retry the run_python call with the code compressed to a single line using \\n for line breaks, properly JSON-escaped.",
                    }
                )
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME, messages=chat_messages, tools=TOOL_SCHEMAS
                )
            except Exception as e2:
                print(f"[AGENT ERROR] {e}")
                if log_fn:
                    log_fn({"event": "llm_error", "error": str(e)})
                break

        msg = response.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)

        if not tool_calls:
            final_text = (msg.content or "").strip()
            break

        # model wants to call one or more tools — append its request, then run each
        chat_messages.append(
            {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            }
        )

        for tc in tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments)
            except Exception:
                fn_args = {}

            if log_fn:
                log_fn({"event": "tool_call", "tool": fn_name, "args": fn_args})

            fn = TOOL_FUNCTIONS.get(fn_name)
            if fn is None:
                result = f"ERROR: unknown tool {fn_name}"
            else:
                result = with_timeout(fn, 20, **fn_args)
            chat_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(result),
                }
            )

    elapsed = time.monotonic() - start
    print(f"[AGENT] finished in {elapsed:.1f}s, raw output: {final_text[:300]}")

    return format_final_answer(
        final_text, original_question=messages[-1]["text"], log_url=LOG_URL
    )
