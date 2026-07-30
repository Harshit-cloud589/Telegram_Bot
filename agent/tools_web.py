import concurrent.futures
import io
import os
import statistics
from xmlrpc import client

import numpy as np
import pandas as pd
import pdfplumber
import requests
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


def with_timeout(fn, timeout_s, *args, **kwargs):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(fn, *args, **kwargs)
        try:
            return future.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError:
            return f"ERROR: tool timed out after {timeout_s}s"


def web_search_tool(query: str) -> str:
    api_key = os.getenv("SERPER_API_KEY", "")
    if not api_key:
        return "ERROR: SERPER_API_KEY not configured"
    try:
        resp = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": query},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("organic", [])[:5]
        if not results:
            return f"No results found for: {query}"
        return "\n".join(
            f"{r.get('title', '')} - {r.get('link', '')}\n{r.get('snippet', '')}"
            for r in results
        )
    except Exception as e:
        return f"ERROR searching: {e}"


def web_fetch(url: str) -> str:
    """Fetch and extract the main readable text content of a URL (HTML stripped)."""
    import requests
    from bs4 import BeautifulSoup

    try:
        resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        return text[:8000]  # cap size
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


def get_cached_dataset(url: str):
    """Helper exposed inside run_python's exec globals."""
    return _dataset_cache.get(url)


def run_python(code: str) -> str:
    """Execute a Python snippet. pandas/numpy/statistics available. If a dataset was
    previously fetched via fetch_dataset, load it with get_cached_dataset(url).
    Print the final result — only stdout is captured.
    """
    import contextlib
    import io as _io

    allowed_globals = {
        "pd": __import__("pandas"),
        "np": __import__("numpy"),
        "statistics": __import__("statistics"),
        "get_cached_dataset": get_cached_dataset,
        "__builtins__": __builtins__,
    }
    buf = _io.StringIO()
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


def fetch_dataset(url: str) -> str:
    """Download a CSV, Excel, JSON, or TSV file from a URL and return a preview
    (column names, dtypes, first 10 rows) so you can plan how to analyze it.
    Use this whenever a question links to a downloadable dataset file.

    Args:
        url: Direct URL to a CSV/XLSX/XLS/TSV/JSON data file.
    """
    import io

    import pandas as pd
    import requests

    try:
        resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        content = resp.content

        # try to guess format from URL / content-type
        url_lower = url.lower()
        if url_lower.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(content))
        elif url_lower.endswith(".json"):
            df = pd.read_json(io.BytesIO(content))
        elif url_lower.endswith(".tsv"):
            df = pd.read_csv(io.BytesIO(content), sep="\t")
        else:
            # default: try CSV, fall back to sniffing
            try:
                df = pd.read_csv(io.BytesIO(content))
            except Exception:
                df = pd.read_csv(io.BytesIO(content), sep=None, engine="python")

        # cache it globally so run_python can access it without re-downloading
        _dataset_cache[url] = df

        preview = (
            f"Loaded dataset from {url}\n"
            f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n"
            f"Columns: {list(df.columns)}\n"
            f"Dtypes:\n{df.dtypes.to_string()}\n"
            f"First 10 rows:\n{df.head(10).to_string()}\n\n"
            f"To analyze this data, use run_python with: "
            f"df = get_cached_dataset({url!r}) — then compute what's needed."
        )
        return preview[:6000]
    except Exception as e:
        return f"ERROR fetching dataset {url}: {e}"


_dataset_cache: dict[str, "pd.DataFrame"] = {}


def analyze_image(image_url: str, question: str) -> str:
    """Fetch an image (chart, graph, table, screenshot) from a URL and analyze it
    using a vision-capable model to extract data or answer a question about it.

    Args:
        image_url: Direct URL to an image (PNG, JPG, etc.) containing a chart, graph, or table.
        question: What specifically to extract or answer from the image.
    """
    import base64

    import requests

    try:
        resp = requests.get(
            image_url, timeout=20, headers={"User-Agent": "Mozilla/5.0"}
        )
        resp.raise_for_status()
        image_b64 = base64.b64encode(resp.content).decode("utf-8")

        content_type = resp.headers.get("Content-Type", "image/png")
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        vision_response = client.chat.completions.create(
            model="qwen/qwen3.6-27b",  # confirm exact current model name
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Analyze this image and answer: {question}. If it's a chart/graph, extract exact numeric values where possible. If it's a table, transcribe the relevant data precisely.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{content_type};base64,{image_b64}"
                            },
                        },
                    ],
                }
            ],
        )
        return vision_response.choices[0].message.content
    except Exception as e:
        return f"ERROR analyzing image {image_url}: {e}"
