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
import re
import time

from agent.tools_schema import TOOL_FUNCTIONS, TOOL_SCHEMAS
from agent.tools_web import with_timeout


def try_parse_pseudo_function_call(text: str):
    """Detect Groq's malformed <function=name>{...}</function> or <function(name)>{...}</function>
    pattern and extract name + args if present."""
    m = re.search(
        r"<function[=\(]([a-zA-Z_]+)[\)\]]?\s*(\{.*?\})\s*</function>", text, re.DOTALL
    )
    if m:
        name = m.group(1)
        try:
            args = json.loads(m.group(2))
            return name, args
        except Exception:
            return None
    return None


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


def build_chat_messages(session_messages: list[dict]) -> list[dict]:
    chat_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in session_messages:
        role = m.get("role")
        text = m.get("text")
        if role not in ("user", "assistant") or not isinstance(text, str):
            print(f"[WARN] skipping malformed session entry: {m}", flush=True)
            continue
        chat_messages.append({"role": role, "content": text})
    return chat_messages


async def run_agent_and_format(
    messages: list[dict], timeout_seconds: int = 60, log_fn=None
) -> dict:
    start = time.monotonic()
    deadline = start + (timeout_seconds * 0.75)

    chat_messages = build_chat_messages(messages)

    final_text = ""

    for iteration in range(8):  # max tool-call iterations
        if time.monotonic() > deadline:
            if log_fn:
                log_fn({"event": "budget_exceeded", "iteration": iteration})
            break

        try:
            print(
                f"[DEBUG] iteration {iteration}, sending {len(chat_messages)} messages:",
                flush=True,
            )
            for i, m in enumerate(chat_messages):
                print(f"  [{i}] {json.dumps(m, default=str)[:300]}", flush=True)
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
                    print(f"[AGENT ERROR] retry also failed: {e2}", flush=True)
                    if log_fn:
                        log_fn({"event": "llm_error", "error": str(e2)})
                    break
            else:
                # Not a tool-parse error — don't blindly retry, log the REAL error and stop
                print(f"[AGENT ERROR] {err_str}", flush=True)
                if log_fn:
                    log_fn({"event": "llm_error", "error": err_str})
                break
        msg = response.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)

        if not tool_calls:
            content = (msg.content or "").strip()
            pseudo = try_parse_pseudo_function_call(content)
            if pseudo:
                fn_name, fn_args = pseudo
                if log_fn:
                    log_fn(
                        {
                            "event": "pseudo_function_call_recovered",
                            "tool": fn_name,
                            "args": fn_args,
                        }
                    )
                fn = TOOL_FUNCTIONS.get(fn_name)
                result = (
                    with_timeout(fn, 20, **fn_args)
                    if fn
                    else f"ERROR: unknown tool {fn_name}"
                )
                chat_messages.append({"role": "assistant", "content": content})
                chat_messages.append(
                    {
                        "role": "user",
                        "content": f"Tool result: {result}\n\nNow provide your final answer as valid JSON only.",
                    }
                )
                continue  # loop again instead of treating this as final
            final_text = content
            break

        # model wants to call one or more tools — append its request, then run each
        # IMPORTANT: mode="json" forces enums/objects into plain JSON-safe values
        chat_messages.append(msg.model_dump(mode="json", exclude_none=True))

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
    print(
        f"[AGENT] finished in {elapsed:.1f}s, raw output: {final_text[:300]}",
        flush=True,
    )

    return format_final_answer(
        final_text, original_question=messages[-1]["text"], log_url=LOG_URL
    )
