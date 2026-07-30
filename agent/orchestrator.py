import os
import time
from typing import Any

import google.genai as genai
from agent.formatter import format_final_answer
from google.genai import types
from groq import APIStatusError, Groq, RateLimitError
from logger import LOG_URL

# client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL_NAME = "openai/gpt-oss-120b"
import inspect
import json
import re
import time
from urllib.parse import urlparse

from agent.tools_schema import TOOL_FUNCTIONS, TOOL_SCHEMAS
from agent.tools_web import with_timeout

# ---------------------------------------------------------------------------
# Deterministic tool routing — don't trust the LLM for things we can check
# from the URL/extension alone.
# ---------------------------------------------------------------------------

FETCH_FAMILY = {
    "web_fetch",
    "fetch_pdf_tables",
    "fetch_excel_table",
    "fetch_dataset",
    "fetch_table_from_url",
}

TOOL_TIMEOUTS = {
    "fetch_pdf_tables": 45,
    "fetch_excel_table": 30,
    "fetch_dataset": 30,
    "web_fetch": 15,
    "web_search_tool": 15,
    "analyze_image": 20,
    "run_python": 20,
}
DEFAULT_TOOL_TIMEOUT = 20


def resolve_fetch_tool(requested_name: str, args: dict) -> str:
    """If the requested tool is a fetch-family tool and the resource type is
    deterministically inferable from the URL, override the LLM's choice
    instead of letting a wrong pick (e.g. web_fetch on a .pdf) burn a
    timeout + an extra reasoning iteration."""
    if requested_name not in FETCH_FAMILY:
        return requested_name

    url = args.get("url", "")
    path = urlparse(url).path.lower()

    if path.endswith(".pdf"):
        return "fetch_pdf_tables"
    if path.endswith((".xls", ".xlsx")):
        return "fetch_excel_table"
    if path.endswith((".csv", ".tsv", ".json")):
        return "fetch_dataset"

    return requested_name  # ambiguous — trust the model's original choice


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


def try_extract_failed_generation(exc: Exception):
    """Groq sometimes rejects a tool call at the HTTP layer (400 tool_use_failed)
    instead of returning it as normal message content. The malformed generation
    is usually available in exc.body['error']['failed_generation'] or embedded
    in str(exc). Try both, then run it through the same pseudo-parser used for
    the 200-OK case so both failure shapes get one recovery path."""
    body = getattr(exc, "body", None)
    failed_gen = None

    if isinstance(body, dict):
        failed_gen = body.get("error", {}).get("failed_generation")

    if not failed_gen:
        m = re.search(r"'failed_generation':\s*'(.*?)'\}\}?\s*$", str(exc), re.DOTALL)
        if m:
            failed_gen = m.group(1).encode().decode("unicode_escape")

    if not failed_gen:
        return None

    return try_parse_pseudo_function_call(failed_gen)


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

- web_search_tool
    Use when you need to discover a webpage.

- web_fetch
    Use to read HTML webpages only. Never use on URLs ending in .pdf, .xls,
    .xlsx, .csv, .tsv, or .json — those are routed automatically.

- fetch_pdf_tables
    Use to read tables from PDF files.

- fetch_excel_table
    Use to read Excel files.

- fetch_dataset
    Use to download CSV, TSV, JSON or Excel datasets.
    The dataset becomes available inside run_python via:
        get_cached_dataset(url)

- analyze_image
    Use for charts, screenshots and images.

- run_python
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


def _run_resolved_tool(
    fn_name: str, fn_args: dict, log_fn=None, event_name: str = "tool_call"
):
    """Shared execution path: resolve → log → clean args → run with a
    per-tool timeout. Used by both the native tool_calls path and the
    pseudo-function-call recovery path so they can't drift apart."""
    corrected_name = resolve_fetch_tool(fn_name, fn_args)
    if corrected_name != fn_name:
        if log_fn:
            log_fn(
                {
                    "event": "tool_rerouted",
                    "requested": fn_name,
                    "used": corrected_name,
                    "args": fn_args,
                }
            )
        fn_name = corrected_name

    if log_fn:
        log_fn({"event": event_name, "tool": fn_name, "args": fn_args})

    fn = TOOL_FUNCTIONS.get(fn_name)
    if fn is None:
        return fn_name, f"ERROR: unknown tool {fn_name}"

    fn_args = clean_tool_args(fn, fn_args)
    timeout = TOOL_TIMEOUTS.get(fn_name, DEFAULT_TOOL_TIMEOUT)
    result = with_timeout(fn, timeout, **fn_args)
    return fn_name, result


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
        except RateLimitError as e:
            # Never let this crash the handler with no reply. Surface it as
            # a final answer the formatter can present gracefully.
            if log_fn:
                log_fn({"event": "rate_limited", "error": str(e)})
            final_text = "null"
            if log_fn:
                log_fn({"event": "agent_degraded_reply", "reason": "rate_limited"})
            break
        except APIStatusError as e:
            err = str(e)
            if log_fn:
                log_fn({"event": "llm_error", "error": err})

            if "tool_use_failed" in err:
                pseudo = try_extract_failed_generation(e)
                if pseudo:
                    fn_name, fn_args = pseudo
                    if log_fn:
                        log_fn(
                            {
                                "event": "pseudo_function_call_recovered_from_error",
                                "tool": fn_name,
                                "args": fn_args,
                            }
                        )
                    used_name, result = _run_resolved_tool(fn_name, fn_args, log_fn)
                    # We never got an assistant message for this turn (Groq
                    # rejected it before returning one), so reconstruct a
                    # minimal one so the tool result has something to attach to.
                    chat_messages.append(
                        {
                            "role": "assistant",
                            "content": f"<function={fn_name}>{json.dumps(fn_args)}</function>",
                        }
                    )
                    chat_messages.append(
                        {
                            "role": "user",
                            "content": f"Tool result: {result}\n\nNow provide your final answer as valid JSON only.",
                        }
                    )
                    continue
                raise RuntimeError(
                    f"Groq rejected tool call and no recoverable generation was found:\n{err}"
                )
            raise

        msg = response.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)

        if not tool_calls:
            content = (msg.content or "").strip()
            pseudo = try_parse_pseudo_function_call(content)
            if pseudo:
                fn_name, fn_args = pseudo
                used_name, result = _run_resolved_tool(
                    fn_name,
                    fn_args,
                    log_fn,
                    event_name="pseudo_function_call_recovered",
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

            _, result = _run_resolved_tool(fn_name, fn_args, log_fn)

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
