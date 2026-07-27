import concurrent.futures
import io

import pandas as pd
import requests


def with_timeout(fn, timeout_s, *args, **kwargs):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(fn, *args, **kwargs)
        try:
            return future.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError:
            return f"ERROR: tool timed out after {timeout_s}s"


def web_fetch(url: str) -> str:
    """Fetch the raw text/HTML content of a URL. Use for MOSPI pages, data.gov.in, etc.

    Args:
        url: The full URL to fetch.
    """
    try:
        resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        return resp.text[:15000]  # cap size to avoid blowing context
    except Exception as e:
        return f"ERROR fetching {url}: {e}"


def fetch_table_from_url(url: str, table_index: int = 0) -> str:
    """Fetch a URL and parse the Nth HTML table on the page into CSV text.

    Args:
        url: URL containing an HTML table.
        table_index: Which table on the page to extract (0-indexed).
    """
    try:
        tables = pd.read_html(url)
        return tables[table_index].to_csv(index=False)
    except Exception as e:
        return f"ERROR parsing table from {url}: {e}"


def run_python(code: str) -> str:
    """Execute a Python snippet for data computation (pandas/numpy/statistics available).
    Print the final result — only stdout is returned. Use this for ALL numeric/formula
    computations; never compute numbers manually.

    Args:
        code: Python source code to execute.
    """
    import contextlib

    allowed_globals = {
        "pd": pd,
        "pandas": pd,
        "io": io,
        "__builtins__": __builtins__,
    }
    try:
        import statistics

        import numpy as np

        allowed_globals["np"] = np
        allowed_globals["statistics"] = statistics
    except ImportError:
        pass

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, allowed_globals)
        return buf.getvalue() or "(no output — did you forget to print?)"
    except Exception as e:
        return f"ERROR executing code: {e}"
