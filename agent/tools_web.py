import concurrent.futures
import io
import os
import statistics

import numpy as np
import pandas as pd
import pdfplumber
import requests


def with_timeout(fn, timeout_s, *args, **kwargs):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(fn, *args, **kwargs)
        try:
            return future.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError:
            return f"ERROR: tool timed out after {timeout_s}s"


def web_search_tool(query: str) -> str:
    resp = requests.get(
        "https://api.serper.dev/search",
        headers={"X-API-KEY": os.environ["SERPER_API_KEY"]},
        json={"q": query},
        timeout=15,
    )
    results = resp.json().get("organic", [])[:5]
    return "\n".join(
        f"{r['title']} - {r['link']}\n{r.get('snippet', '')}" for r in results
    )


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


def fetch_pdf_tables(url: str) -> str:
    """Fetch a PDF from a URL and extract all tables into CSV text.

    Args:
        url: URL of the PDF file.
    """
    try:
        resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        pdf = pdfplumber.open(io.BytesIO(resp.content))
        output = []
        for i, page in enumerate(pdf.pages[:10]):  # cap pages to avoid huge output
            tables = page.extract_tables()
            for t in tables:
                output.append(f"--- page {i + 1} table ---")
                output.append(
                    "\n".join(",".join(str(c) if c else "" for c in row) for row in t)
                )
        return "\n".join(output) if output else "No tables found in first 10 pages."
    except Exception as e:
        return f"ERROR parsing PDF {url}: {e}"


def fetch_excel_table(url: str) -> str:
    """Download an Excel file and return its contents as CSV text.

    Args:
        url: Direct URL to an .xls or .xlsx file.
        sheet_name: Optional sheet name; if omitted, reads the first sheet.
    """
    try:
        resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        df = pd.read_excel(io.BytesIO(resp.content), sheet_name=sheet_name or 0)
        return df.to_csv(index=False)[:10000]  # cap size
    except Exception as e:
        return f"ERROR parsing Excel {url}: {e}"


def datagovin_search(query: str) -> str:
    """Search data.gov.in's open data catalog for a dataset matching the query.

    Args:
        query: Search terms, e.g. "state wise per capita income"
    """
    resp = requests.get(
        "https://api.data.gov.in/catalog",
        params={
            "api-key": os.environ.get("DATA_GOV_IN_API_KEY", ""),
            "format": "json",
            "filters[title]": query,
        },
        timeout=15,
    )
    return resp.text[:5000]
