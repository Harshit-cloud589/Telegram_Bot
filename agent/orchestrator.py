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
MODEL_NAME = "llama-3.3-70b-versatile"
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
DEFAULT_TOOL_TIMEOUT = 60


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

If a search result already contains the requested numeric answer,
DO NOT fetch the webpage.

Return the answer immediately.

Only fetch a webpage if the search snippet is insufficient.
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

==========================
DATASET RULES
==========================

If the user's message contains ANY URL ending in

.csv
.tsv
.xlsx
.xls
.json

you MUST call fetch_dataset first.

Examples:

User:
https://.../sales.csv
What is the average revenue?

Assistant:
fetch_dataset(url)

Then

run_python(
df = get_cached_dataset(url)
...)

Never answer from the URL alone.

Never ignore a dataset URL.

Never attempt to estimate values.

Always fetch the dataset first.
Print ONLY the final value(s) needed for the answer.
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


def should_force_answer(state: AgentState) -> bool:
    """
    Decide whether the agent has started looping.
    """

    if len(state.executed_tools) < 4:
        return False

    recent = state.executed_tools[-4:]

    tool_names = [t[0] for t in recent]

    searches = tool_names.count("web_search_tool")

    fetches = sum(
        x
        in (
            "web_fetch",
            "fetch_table_from_url",
            "fetch_pdf_tables",
            "fetch_dataset",
            "fetch_excel_table",
        )
        for x in tool_names
    )

    # same search twice
    if searches >= 2:
        return True

    # lots of fetching
    if fetches >= 3:
        return True

    # nearing iteration limit
    if state.iteration >= MAX_ITERATIONS - 2:
        return True

    return False


def ask_llm_without_tools(chat_messages):

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=chat_messages,
        temperature=0,
    )

    return response.choices[0].message


MAX_ITERATIONS = 8
MAX_TOOL_CALLS = 10
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
        key = (
            fn_name,
            fn_args["url"],
        )

        if key in state.visited_urls:
            return f"{fn_name} already executed on this URL."

        state.visited_urls.add(key)

    used_name, result = _run_resolved_tool(
        fn_name,
        fn_args,
        log_fn,
    )
    if used_name == "fetch_dataset":
        state.chat_messages.append(
            {
                "role": "system",
                "content": (
                    "The dataset has been loaded.\n"
                    "Your next action MUST be run_python.\n"
                    "Do not answer yet.\n"
                    "Do not call fetch_dataset again."
                ),
            }
        )
    if used_name == "web_search_tool" and isinstance(result, str):
        result += """

    SYSTEM NOTE:
    If the requested answer is already present in these search results,
    DO NOT call another tool.
    Return the final answer immediately.
    Only fetch webpages if the search results are insufficient.
    """

    print(result[:1000] if isinstance(result, str) else result)

    state.executed_tools.append(
        (
            used_name,
            fn_args,
        )
    )

    return result


# ============================================================
# LLM
# ============================================================


def ask_llm(chat_messages):
    print("ENTER ask_llm", flush=True)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=chat_messages,
            tools=TOOL_SCHEMAS,
            temperature=0,
        )
        print("CREATE RETURNED", flush=True)
        return response.choices[0].message

    except BaseException as e:
        print("CREATE FAILED:", type(e).__name__, repr(e), flush=True)
        raise


async def run_agent_and_format(
    messages: list[dict],
    timeout_seconds: int = 60,
    log_fn=None,
) -> dict:

    start = time.monotonic()
    deadline = start + timeout_seconds

    state = AgentState(chat_messages=build_chat_messages(messages))

    while not should_stop(state):
        print(
            f"TOP OF LOOP | iter={state.iteration} "
            f"answer={state.final_answer!r} "
            f"tools={len(state.executed_tools)}"
        )

        if time.monotonic() >= deadline:
            print("[TIMEOUT] Asking for final answer...")

            final_messages = list(state.chat_messages)

            final_messages.append(
                {
                    "role": "system",
                    "content": "You have no time left.\n"
                    "Do NOT call any more tools.\n"
                    "Using ONLY the information already collected, answer the user's original question.\n"
                    "If the answer cannot be determined, return null.",
                }
            )

            try:
                msg = ask_llm_without_tools(final_messages)
                state.final_answer = (msg.content or "").strip()
            except Exception:
                state.final_answer = "null"

            break
        trim_history(state)

        try:
            print("before ask_llm")
            msg = ask_llm(state.chat_messages)
            print("after ask_llm")
            if state.iteration >= 6:
                state.chat_messages.append(
                    {
                        "role": "system",
                        "content": "This is your final reasoning step.\n"
                        "You are no longer allowed to use any tools.\n"
                        "Answer using the evidence already collected.",
                    }
                )

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
        except Exception as e:
            print("ASK_LLM FAILED:", type(e).__name__, repr(e))
            raise
        print("=" * 80)
        print(msg.model_dump(mode="json"))
        print("=" * 80)
        tool_calls = getattr(msg, "tool_calls", None)

        ##############################################################
        # FINAL ANSWER
        ##############################################################

        if not tool_calls:
            print("BREAKING because no tool calls")
            print("CONTENT =", repr(msg.content))
            content = (msg.content or "").strip()
            if not content and state.executed_tools:
                print("Empty response after tool call, continuing...")
                continue
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
        if should_force_answer(state):
            evidence = "\n\n".join(state.evidence[-6:])

            final_messages = list(state.chat_messages)

            final_messages.append(
                {
                    "role": "system",
                    "content": (
                        "You have enough evidence.\n"
                        "You are NOT allowed to call tools anymore.\n"
                        "Use ONLY the collected evidence.\n"
                        "If the answer cannot be determined, reply with null."
                    ),
                }
            )

            final_messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Evidence:\n\n{evidence}\n\nReturn the final answer now."
                    ),
                }
            )

            final_msg = ask_llm_without_tools(final_messages)

            state.final_answer = (final_msg.content or "").strip()

            break
        # -------------------------------------------------------
        # Stop endless search/fetch loops
        # -------------------------------------------------------

        last_tools = [name for name, _ in state.executed_tools[-4:]]

        searches = last_tools.count("web_search_tool")
        fetches = sum(
            t
            in (
                "web_fetch",
                "fetch_table_from_url",
                "fetch_pdf_tables",
                "fetch_dataset",
                "fetch_excel_table",
            )
            for t in last_tools
        )

        if searches >= 2 or fetches >= 3:
            evidence = "\n\n".join(state.evidence[-6:])

            state.chat_messages.append(
                {
                    "role": "system",
                    "content": "STOP USING TOOLS.\n\n"
                    "You already have enough evidence.\n\n"
                    f"{evidence}\n\n"
                    "Your next response MUST be the final answer.\n"
                    "Do not call another tool.",
                }
            )
        state.iteration += 1
    print(
        "should_stop:",
        should_stop(state),
        state.final_answer,
        state.iteration,
        len(state.executed_tools),
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
