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

FOCUS AND EFFICIENCY:
5. Stay strictly focused on exactly what the question asks. Do NOT research or fetch
   information unrelated to the specific question — e.g. if asked for a country's capital,
   do not also look up its GDP, exports, or other unrelated facts.
6. As soon as you have found the specific information needed to answer the question, STOP
   calling tools and provide your final answer immediately. Do not continue gathering
   additional information "just in case."
7. Before giving your final answer, re-read the original question one more time and confirm
   your answer directly and specifically addresses what was asked — not a tangential fact you
   happened to find along the way.
8. Each run_python call executes in a fresh, isolated environment — variables and imports
   from previous calls are NOT available. Write each code snippet to be fully self-contained.

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
When computing standard deviation or variance, use SAMPLE statistics (dividing by N-1,
i.e. Python's statistics.stdev / statistics.variance or pandas' default .std()/.var()),
unless the question explicitly asks for population statistics.
ALWAYS use run_python for ANY computation, including simple ones like finding a maximum,
minimum, index/position, count, or sum — even if it looks trivial enough to do mentally.
Do not compute or count anything by reasoning alone.
LINKED DATASET QUESTIONS: If a question provides a URL to a CSV, Excel, JSON, or other data
file (or asks you to download/analyze a dataset), follow this exact two-step process:
1. Call fetch_dataset(url) first — this downloads the file and shows you its columns,
   types, and a preview of the data.
2. Then call run_python with code that starts with:
   df = get_cached_dataset("<the same url>")
   ...and perform the actual analysis/computation on df using pandas.
Never try to answer a dataset question from the preview text alone — always load the real
DataFrame via get_cached_dataset and compute the exact answer with pandas/numpy.
If a given URL is a webpage (not a direct file), first use web_fetch to find the actual
download link (look for .csv, .xlsx, .xls, .json hrefs in the page), then call
fetch_dataset on that direct file URL.
NEVER FABRICATE: Never invent, guess, or hardcode a plausible-looking answer when you don't
have real data. If a tool fails or returns nothing useful, try a different tool or query — do
not write code that just prints a guessed literal value instead of using real fetched data.
If you truly cannot find the answer after genuine attempts, respond with {"answer": null}
rather than inventing a number or name.
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

    for iteration in range(3):
        if iteration <= 1 and any(
            "ERROR" not in str(r) for r in [tc for tc in chat_messages[-1:]]
        ):
            chat_messages.append(
                {
                    "role": "user",
                    "content": "You likely already have enough information to answer now. If so, respond with ONLY the final JSON answer immediately — do not search for anything else.",
                }
            )  # max tool-call iterations
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
