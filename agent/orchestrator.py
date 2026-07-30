import os
import time
from dataclasses import dataclass, field
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
def compress_tool_result(result, limit=600):
    text = str(result).strip()

    if len(text) <= limit:
        return text

    lines = text.splitlines()

    # keep only first few useful lines
    return "\n".join(lines[:8])[:limit] + "\n...[truncated]"


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
        chat_messages.append({"role": role, "content": compress_tool_result(text)})
    return chat_messages


def _run_resolved_tool(
    fn_name: str, fn_args: dict, log_fn=None, event_name: str = "tool_call"
):
    """Shared execution path: resolve → log → clean args → run with a
    per-tool timeout. Used by both the native tool_calls path and the
    pseudo-function-call recovery path so they can't drift apart."""
    corrected_name = resolve_fetch_tool(fn_name, fn_args)
    if corrected_name != fn_name:
        # if log_fn:
        #     log_fn(
        #         {
        #             "event": "tool_rerouted",
        #             "requested": fn_name,
        #             "m": corrected_name,
        #             "args": fn_args,
        #         }
        #     )
        fn_name = corrected_name

    # if log_fn:
    #     log_fn({"event": event_name, "tool": fn_name, "args": fn_args})

    fn = TOOL_FUNCTIONS.get(fn_name)
    if fn is None:
        return fn_name, f"ERROR: unknown tool {fn_name}"

    fn_args = clean_tool_args(fn, fn_args)
    timeout = TOOL_TIMEOUTS.get(fn_name, DEFAULT_TOOL_TIMEOUT)
    result = with_timeout(fn, timeout, **fn_args)
    return fn_name, result


MAX_ITERATIONS = 15
MAX_TOOL_CALLS = 25
MAX_HISTORY = 30


@dataclass
class AgentState:
    chat_messages: list

    final_answer: str | None = None

    evidence: list[str] = field(default_factory=list)

    visited_urls: set[str] = field(default_factory=set)

    searched_queries: set[str] = field(default_factory=set)

    executed_tools: list[tuple] = field(default_factory=list)

    iteration: int = 0


def add_tool_message(state, tc, result):

    text = compress_tool_result(result)

    state.evidence.append(text)

    state.chat_messages.append(
        {
            "role": "tool",
            "tool_call_id": tc.id,
            "content": text,
        }
    )


def trim_history(state):

    if len(state.chat_messages) <= MAX_HISTORY:
        return

    state.chat_messages = [state.chat_messages[0]] + state.chat_messages[
        -(MAX_HISTORY - 1) :
    ]


def should_stop(state):

    if state.final_answer:
        return True

    if state.iteration >= MAX_ITERATIONS:
        return True

    if len(state.executed_tools) >= MAX_TOOL_CALLS:
        return True

    return False


# ============================================================
# Tool execution
# ============================================================


def execute_tool_call(tc, state, log_fn=None):

    fn_name = tc.function.name
    print(f"[TOOL] {fn_name}")

    try:
        fn_args = json.loads(tc.function.arguments)

    except Exception:
        fn_args = {}

    if fn_name == "web_search_tool":
        q = fn_args.get("query")

        if q:
            if q in state.searched_queries:
                return "Search already executed."

            state.searched_queries.add(q)

    if "url" in fn_args:
        url = fn_args["url"]

        if url in state.visited_urls:
            return "URL already fetched."

        state.visited_urls.add(url)

    used_name, result = _run_resolved_tool(
        fn_name,
        fn_args,
        log_fn,
    )
    print(result[:1000] if isinstance(result, str) else result)
    state.executed_tools.append(
        (
            used_name,
            fn_args,
        )
    )
    print(state.chat_messages[-1])
    return result


# ============================================================
# LLM
# ============================================================


def ask_llm(chat_messages):

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=chat_messages,
        tools=TOOL_SCHEMAS,
        temperature=0,
    )

    return response.choices[0].message


async def run_agent_and_format(
    messages: list[dict],
    timeout_seconds: int = 60,
    log_fn=None,
) -> dict:

    start = time.monotonic()
    deadline = start + timeout_seconds

    state = AgentState(chat_messages=build_chat_messages(messages))

    while not should_stop(state):
        state.iteration += 1

        if time.monotonic() >= deadline:
            state.final_answer = "null"
            break

        trim_history(state)

        try:
            msg = ask_llm(state.chat_messages)

        except RateLimitError:
            state.final_answer = "null"
            break

        except APIStatusError as e:
            err = str(e)

            if "tool_use_failed" not in err:
                raise

            pseudo = try_extract_failed_generation(e)

            if pseudo is None:
                state.final_answer = "null"
                break

            fn_name, fn_args = pseudo

            class FakeFunction:
                def __init__(self, name, arguments):
                    self.name = name
                    self.arguments = json.dumps(arguments)

            class FakeTool:
                def __init__(self, name, args):
                    self.id = "pseudo"
                    self.function = FakeFunction(name, args)

            fake_tool = FakeTool(fn_name, fn_args)

            result = execute_tool_call(
                fake_tool,
                state,
                log_fn,
            )

            state.chat_messages.append(
                {
                    "role": "assistant",
                    "content": f"<function={fn_name}>{json.dumps(fn_args)}</function>",
                }
            )

            add_tool_message(
                state,
                fake_tool,
                result,
            )

            continue
        print("=" * 80)
        print(msg.model_dump(mode="json"))
        print("=" * 80)
        tool_calls = getattr(msg, "tool_calls", None)

        ##############################################################
        # FINAL ANSWER
        ##############################################################

        if not tool_calls:
            content = (msg.content or "").strip()

            pseudo = try_parse_pseudo_function_call(content)

            if pseudo:
                fn_name, fn_args = pseudo

                class FakeFunction:
                    def __init__(self, name, arguments):
                        self.name = name
                        self.arguments = json.dumps(arguments)

                class FakeTool:
                    def __init__(self, name, args):
                        self.id = "pseudo"
                        self.function = FakeFunction(name, args)

                fake_tool = FakeTool(
                    fn_name,
                    fn_args,
                )

                result = execute_tool_call(
                    fake_tool,
                    state,
                    log_fn,
                )

                state.chat_messages.append(
                    {
                        "role": "assistant",
                        "content": content,
                    }
                )

                add_tool_message(
                    state,
                    fake_tool,
                    result,
                )

                continue

            state.final_answer = content

            break

        ##############################################################
        # TOOL CALLS
        ##############################################################

        state.chat_messages.append(
            {
                "role": "assistant",
                "tool_calls": msg.model_dump(mode="json")["tool_calls"],
            }
        )

        for tc in tool_calls:
            result = execute_tool_call(
                tc,
                state,
                log_fn,
            )

            add_tool_message(
                state,
                tc,
                result,
            )

        ##############################################################
        # Force completion after enough evidence
        ##############################################################

        if len(state.executed_tools) >= 8 and len(state.evidence) >= 5:
            summary = "\n\n".join(state.evidence[-5:])

            state.chat_messages.append(
                {
                    "role": "user",
                    "content": (
                        "Enough evidence has been collected.\n\n"
                        "Evidence:\n\n"
                        f"{summary}\n\n"
                        "Do NOT call any more tools.\n"
                        "Answer the original question."
                    ),
                }
            )

    elapsed = time.monotonic() - start

    print(
        f"[AGENT] iterations={state.iteration}",
        flush=True,
    )

    print(
        f"[AGENT] tools={len(state.executed_tools)}",
        flush=True,
    )

    print(
        f"[AGENT] evidence={len(state.evidence)}",
        flush=True,
    )

    print(
        f"[AGENT] finished in {elapsed:.2f}s",
        flush=True,
    )

    answer = state.final_answer

    if not answer:
        answer = "null"

    return format_final_answer(
        answer,
        original_question=messages[-1]["text"],
        log_url=LOG_URL,
    )
