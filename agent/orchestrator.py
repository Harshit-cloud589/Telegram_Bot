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
DEFAULT_TOOL_TIMEOUT = 60

# ---------------------------------------------------------------------------
# Dataset URL detection — run BEFORE the reasoning loop so we never depend on
# the model deciding to call fetch_dataset on its own.
# ---------------------------------------------------------------------------
DATASET_URL_RE = re.compile(
    r"https?://\S+?\.(?:csv|tsv|xlsx|xls|json)(?:\?\S*)?(?=[\s\)\]\}\"'<>]|$)",
    re.IGNORECASE,
)


def detect_dataset_urls(text: str) -> list[str]:
    """Return every dataset-looking URL (csv/tsv/xlsx/xls/json) found in the
    given text, in order of first appearance, de-duplicated."""
    if not isinstance(text, str) or not text:
        return []

    seen = []
    for match in DATASET_URL_RE.findall(text):
        cleaned = match.strip().rstrip(".,;:")
        if cleaned and cleaned not in seen:
            seen.append(cleaned)
    return seen


# ---------------------------------------------------------------------------
# "null"-ish answer detection — GPT-OSS sometimes returns the literal text
# "null" (or a quoted/punctuated variant of it) as its final content instead
# of calling a needed tool (e.g. run_python right after a dataset was
# prefetched). This must be treated the same as an empty response, NOT
# accepted immediately, when we actually have evidence to work with.
# ---------------------------------------------------------------------------
def is_null_like_answer(text: str) -> bool:
    if not isinstance(text, str):
        return False
    stripped = text.strip().strip("\"'` ").strip(".,;:!").strip()
    return stripped.lower() in (
        "null",
        "none",
        "n/a",
        "na",
        "unknown",
        "{}",
        '{"answer": null}',
    )


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


def should_force_answer(state: "AgentState") -> bool:
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

    # Set once we've already retried the LLM one time after an empty
    # assistant response, so we don't retry forever.
    empty_retry_used: bool = False

    # Set once we've already retried the LLM one time after it returned a
    # literal "null"-ish answer while evidence was available, instead of
    # actually using that evidence / calling the needed tool.
    null_retry_used: bool = False


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


def _final_synthesis_or_null(state: "AgentState") -> str:
    """Last-resort recovery path. When the agent hits a dead end (repeated
    empty/null-ish assistant responses, an unrecoverable API error, etc.)
    but has already gathered some evidence, give the model one final
    no-tools pass over that evidence instead of immediately giving up with
    "null"."""
    if not state.evidence:
        return "null"

    try:
        evidence = "\n\n".join(state.evidence[-6:])

        final_messages = list(state.chat_messages)
        final_messages.append(
            {
                "role": "system",
                "content": (
                    "You are NOT allowed to call tools anymore.\n"
                    "Using ONLY the evidence already collected below, answer "
                    "the user's original question.\n"
                    "If the evidence contains a dataset that was fetched but "
                    "never actually analyzed, compute the answer yourself "
                    "from the evidence shown (e.g. row counts, columns, "
                    "sample rows) as best you can instead of giving up.\n\n"
                    f"Evidence:\n\n{evidence}\n\n"
                    "If the answer truly cannot be determined from this "
                    "evidence, reply with null."
                ),
            }
        )

        msg = ask_llm_without_tools(final_messages)
        content = (msg.content or "").strip()
        if content and not is_null_like_answer(content):
            return content
        return content if content else "null"
    except Exception:
        return "null"


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
                    "The dataset is available. Your next tool call must be run_python."
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
# Deterministic dataset pre-fetch (runs before the reasoning loop)
# ============================================================


# def prefetch_dataset_urls(state: "AgentState", user_text: str, log_fn=None) -> None:
#     """Scan the user's message for dataset URLs and fetch them immediately,
#     deterministically, instead of trusting the model to notice and call
#     fetch_dataset itself. Results are injected as evidence + a system note so
#     the model can move straight to run_python."""
#     urls = detect_dataset_urls(user_text)

#     for url in urls:
#         key = ("fetch_dataset", url)
#         if key in state.visited_urls:
#             continue

#         state.visited_urls.add(key)

#         used_name, result = _run_resolved_tool(
#             "fetch_dataset",
#             {"url": url},
#             log_fn,
#         )

#         text = compress_tool_result(result)
#         state.evidence.append(text)
#         state.executed_tools.append((used_name, {"url": url}))

#         print(f"[PREFETCH] fetch_dataset({url}) -> {text[:200]}")

#         state.chat_messages.append(
#             {
#                 "role": "system",
#                 "content": (
#                     f"The dataset at {url} has already been fetched "
#                     "automatically before you started reasoning.\n"
#                     f"Result:\n{text}\n\n"
#                     "Do NOT call fetch_dataset on this URL again.\n"
#                     "You MUST now call run_python with "
#                     "get_cached_dataset(url) to actually compute the "
#                     "answer. Do not answer with null or any other value "
#                     "until you have done this — a dataset being cached is "
#                     "not the same as the answer being known."
#                 ),
#             }
#         )


# ============================================================
# LLM
# ============================================================


def ask_llm(chat_messages):
    print("ENTER ask_llm")

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=chat_messages,
            tools=TOOL_SCHEMAS,
            temperature=0,
        )
        print("CREATE FINISHED")
        print(response)

        return response.choices[0].message

    except BaseException as e:
        print("ASK_LLM EXCEPTION")
        print(type(e))
        print(repr(e))
        raise

    finally:
        print("EXIT ask_llm")


async def run_agent_and_format(
    messages: list[dict],
    timeout_seconds: int = 60,
    log_fn=None,
) -> dict:

    start = time.monotonic()
    deadline = start + timeout_seconds

    state = AgentState(chat_messages=build_chat_messages(messages))

    # ------------------------------------------------------------------
    # Deterministic dataset detection BEFORE entering the reasoning loop.
    # ------------------------------------------------------------------
    last_user_text = ""
    if messages:
        last_user_text = messages[-1].get("text", "") or ""

    # prefetch_dataset_urls(state, last_user_text, log_fn)

    while not should_stop(state):
        print(
            f"TOP OF LOOP | iter={state.iteration} "
            f"answer={state.final_answer!r} "
            f"tools={len(state.executed_tools)}"
        )
        state.iteration += 1

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
                content = (msg.content or "").strip()
                if content and not is_null_like_answer(content):
                    state.final_answer = content
                else:
                    state.final_answer = _final_synthesis_or_null(state)
            except Exception:
                state.final_answer = _final_synthesis_or_null(state)

            break
        trim_history(state)

        try:
            print("CALLING LLM", state.iteration)

            msg = ask_llm(state.chat_messages)

            print("LLM RETURNED", state.iteration)
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
            state.final_answer = _final_synthesis_or_null(state)
            break

        except APIStatusError as e:
            err = str(e)

            if "tool_use_failed" not in err:
                raise

            pseudo = try_extract_failed_generation(e)

            if pseudo is None:
                state.final_answer = _final_synthesis_or_null(state)
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
            print("ASK_LLM CRASHED")
            print(type(e))
            print(e)
            raise
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
                state.empty_retry_used = False
                state.null_retry_used = False

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

            if not content:
                # GPT-OSS sometimes returns a completely empty assistant
                # message with no tool calls. Never treat that as "null"
                # right away — retry the LLM call exactly once first.
                if not state.empty_retry_used:
                    state.empty_retry_used = True
                    print(
                        "[INFO] Empty assistant response with no tool "
                        "calls — retrying LLM once."
                    )
                    continue

                # Already retried once and still got nothing. If we've
                # gathered any evidence along the way, give the model one
                # last no-tools synthesis pass instead of giving up.
                print(
                    "[INFO] Empty assistant response persisted after "
                    "retry — attempting final synthesis from evidence."
                )
                state.final_answer = _final_synthesis_or_null(state)
                break

            # -----------------------------------------------------------
            # Model returned real, non-empty content — but it may still be
            # a literal "null"-ish string returned prematurely (e.g. right
            # after a dataset was prefetched, instead of calling
            # run_python). Don't accept that at face value if we have
            # evidence sitting unused.
            # -----------------------------------------------------------
            if is_null_like_answer(content) and state.evidence:
                if not state.null_retry_used:
                    state.null_retry_used = True
                    print(
                        "[INFO] Model returned a null-ish answer despite "
                        "having evidence — nudging it to use the "
                        "evidence/tools and retrying once."
                    )
                    state.chat_messages.append(
                        {
                            "role": "system",
                            "content": (
                                "You answered null, but there is evidence "
                                "already collected (including any fetched "
                                "datasets) that has not been used yet.\n"
                                "If a dataset was fetched, you MUST call "
                                "run_python with get_cached_dataset(url) to "
                                "compute the answer before concluding it is "
                                "null.\n"
                                "Only answer null if the evidence truly "
                                "cannot answer the question."
                            ),
                        }
                    )
                    continue

                print(
                    "[INFO] Model returned a null-ish answer again after "
                    "retry — attempting final synthesis from evidence."
                )
                state.final_answer = _final_synthesis_or_null(state)
                break

            state.empty_retry_used = False
            state.null_retry_used = False
            state.final_answer = content

            break

        ##############################################################
        # TOOL CALLS
        ##############################################################

        state.empty_retry_used = False
        state.null_retry_used = False

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

            content = (final_msg.content or "").strip()
            if content and not is_null_like_answer(content):
                state.final_answer = content
            else:
                state.final_answer = _final_synthesis_or_null(state)

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

    if not answer or is_null_like_answer(answer):
        answer = _final_synthesis_or_null(state)

    if not answer:
        answer = "null"

    return format_final_answer(
        answer,
        original_question=messages[-1]["text"],
        log_url=LOG_URL,
    )
