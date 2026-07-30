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
MODEL_NAME = "llama-3.3-70b-versatile"
import inspect
import json
import re
import time

from agent.tools_schema import TOOL_FUNCTIONS, TOOL_SCHEMAS
from agent.tools_web import with_timeout


def clean_tool_args(fn, args):

    sig = inspect.signature(fn)

    return {k: v for k, v in args.items() if k in sig.parameters}


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


SYSTEM_PROMPT = """
You are a data-analysis assistant.

You answer questions by using the available tools whenever external information
or computation is required.

The user may ask for the answer in a JSON format such as:

{"answer": <value>}

IMPORTANT:

- Return ONLY the value that belongs inside "answer".
- Never return the full {"answer":...} object.
- Never add explanations unless explicitly requested.
- Never use markdown.

General rules:

1. Never invent or guess information.
2. If a tool provides the answer directly, return it immediately.
3. Use as few tool calls as possible.
4. Stop calling tools as soon as the answer is known.
5. Stay focused only on the user's question.

Tool usage:

• web_search_tool
    Use when you need to discover a webpage.

• web_fetch
    Use to read HTML webpages.

• fetch_pdf_tables
    Use to read tables from PDF files.

• fetch_excel_table
    Use to read Excel files.

• fetch_dataset
    Use to download CSV, TSV, JSON or Excel datasets.
    The dataset becomes available inside run_python via:
        get_cached_dataset(url)

• analyze_image
    Use for charts, screenshots and images.

• run_python
    Use ONLY when actual computation is required.

Examples of run_python:

✓ statistics
✓ filtering
✓ dataframe operations
✓ sorting
✓ grouping
✓ aggregation
✓ mathematical calculations

Do NOT use run_python for:

✗ reading webpages
✗ reading PDFs
✗ extracting text
✗ extracting values already visible
✗ searching the web
✗ finding links

Datasets:

If a question provides a downloadable dataset:

1. Call fetch_dataset().
2. Then use run_python with:
       df = get_cached_dataset(url)

Never answer from the preview alone.

Images:

If a question is about a chart or graph,
use analyze_image first.

Statistics:

Use SAMPLE standard deviation/variance unless the user explicitly requests population statistics.

If no tool can determine the answer after reasonable attempts,
return null.

Never fabricate an answer.
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

    for iteration in range(5):
        if time.monotonic() > deadline:
            if log_fn:
                log_fn({"event": "budget_exceeded", "iteration": iteration})
            break

        try:
            print(
                f"[DEBUG] iteration {iteration}, sending {len(chat_messages)} messages:",
                flush=True,
            )
            print(json.dumps(chat_messages, indent=2))
            for i, m in enumerate(chat_messages):
                print(f"  [{i}] {json.dumps(m, default=str)[:300]}", flush=True)
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=chat_messages,
                tools=TOOL_SCHEMAS,
                temperature=0,
            )
        except Exception as e:
            err = str(e)
            if log_fn:
                log_fn({"event": "llm_error", "error": err})

            if "tool_use_failed" in err:
                raise RuntimeError(f"Groq rejected tool call:\n{err}")
            raise

            # Groq frequently emits malformed XML tool calls.
            # Tell the model exactly how to recover.

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
            except json.JSONDecodeError:
                fn_args = {}

            if log_fn:
                log_fn({"event": "tool_call", "tool": fn_name, "args": fn_args})

            fn = TOOL_FUNCTIONS.get(fn_name)

            if fn is None:
                result = f"ERROR: unknown tool {fn_name}"

            else:
                # ⭐ Remove hallucinated arguments
                fn_args = clean_tool_args(fn, fn_args)

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
