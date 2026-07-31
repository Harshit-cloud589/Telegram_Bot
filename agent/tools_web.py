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

FORBIDDEN_PATTERNS = [
    "requests.get",
    "requests.post",
    "BeautifulSoup",
    "urllib.request",
    "from web_fetch",
]


def with_timeout(fn, timeout_s, *args, **kwargs):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(fn, *args, **kwargs)
        try:
            return future.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError:
            return f"ERROR: tool timed out after {timeout_s}s"


def web_search_tool(query: str, **kwargs) -> str:
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


def web_fetch(url: str, **kwargs) -> str:
    """
    Fetch a webpage and return the main readable content.

    The function:
    - Detects PDFs/images/datasets instead of blindly parsing HTML.
    - Removes boilerplate.
    - Extracts page title.
    - Extracts metadata.
    - Returns cleaned text.
    """

    from urllib.parse import urlparse

    import requests
    from bs4 import BeautifulSoup

    USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/139.0 Safari/537.36"
    )

    try:
        response = requests.get(
            url,
            timeout=20,
            headers={"User-Agent": USER_AGENT},
            allow_redirects=True,
        )

        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "").lower()

        final_url = response.url.lower()

        ####################################################
        # Detect wrong tool
        ####################################################

        if ".pdf" in final_url or "application/pdf" in content_type:
            return "RESOURCE_TYPE=PDF\nUse fetch_pdf_tables() instead."

        if (
            ".csv" in final_url
            or ".xlsx" in final_url
            or ".xls" in final_url
            or ".json" in final_url
            or "text/csv" in content_type
        ):
            return "RESOURCE_TYPE=DATASET\nUse fetch_dataset() instead."

        if content_type.startswith("image/"):
            return "RESOURCE_TYPE=IMAGE\nUse analyze_image()."

        ####################################################
        # Parse HTML
        ####################################################

        soup = BeautifulSoup(
            response.text,
            "lxml",
        )

        ####################################################
        # Remove junk
        ####################################################

        for tag in soup(
            [
                "script",
                "style",
                "svg",
                "noscript",
                "iframe",
                "header",
                "footer",
                "nav",
                "aside",
                "form",
                "button",
            ]
        ):
            tag.decompose()

        ####################################################
        # Remove common navigation divs
        ####################################################

        for node in soup.find_all(True):
            classes = " ".join(node.get("class", [])).lower()

            ident = (node.get("id") or "").lower()

            junk = (
                "menu",
                "nav",
                "footer",
                "header",
                "breadcrumb",
                "sidebar",
                "cookie",
                "banner",
                "share",
                "social",
                "advert",
                "pagination",
            )

            if any(x in classes for x in junk):
                node.decompose()
                continue

            if any(x in ident for x in junk):
                node.decompose()

        ####################################################
        # Metadata
        ####################################################

        title = ""

        if soup.title:
            title = soup.title.get_text(strip=True)

        description = ""

        meta = soup.find(
            "meta",
            attrs={"name": "description"},
        )

        if meta:
            description = meta.get("content", "")

        ####################################################
        # Main content
        ####################################################

        article = None

        selectors = [
            "article",
            "main",
            "#content",
            ".content",
            ".article",
            "#main-content",
            "#main",
        ]

        for selector in selectors:
            article = soup.select_one(selector)

            if article:
                break

        if article is None:
            article = soup.body

        if article is None:
            return "ERROR: No readable HTML."

        text = article.get_text(
            "\n",
            strip=True,
        )

        ####################################################
        # Cleanup
        ####################################################

        cleaned = []

        seen = set()

        for line in text.splitlines():
            line = " ".join(line.split())

            if len(line) < 3:
                continue

            if line in seen:
                continue

            seen.add(line)

            cleaned.append(line)

        text = "\n".join(cleaned)

        ####################################################
        # Cap output
        ####################################################

        MAX_CHARS = 12000

        if len(text) > MAX_CHARS:
            text = text[:MAX_CHARS]

        return (
            f"TITLE: {title}\n\n"
            f"DESCRIPTION: {description}\n\n"
            f"URL: {response.url}\n\n"
            f"CONTENT:\n{text}"
        )

    except Exception as e:
        return f"ERROR fetching {url}: {e}"


def fetch_table_from_url(url: str, table_index: int = 0, **kwargs) -> str:
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
    return DATASET_CACHE.get(url)


def run_python(code: str, **kwargs) -> str:
    for pattern in FORBIDDEN_PATTERNS:
        if pattern in code:
            return (
                f"ERROR: run_python cannot make web requests directly. "
                f"Use web_fetch(url), web_search_tool(query), fetch_dataset(url), "
                f"or fetch_pdf_tables(url) instead, then pass the RESULT into run_python."
            )

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


def fetch_pdf_tables(url: str, **kwargs) -> str:
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


def fetch_excel_table(url: str, **kwargs) -> str:
    """Download an Excel file and return its contents as CSV text.

    Args:
        url: Direct URL to an .xls or .xlsx file.
        sheet_name: Optional sheet name; if omitted, reads the first sheet.
    """
    try:
        resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        df = pd.read_excel(
            io.BytesIO(resp.content), sheet_name=kwargs.get("sheet_name") or 0
        )
        return df.to_csv(index=False)[:10000]  # cap size
    except Exception as e:
        return f"ERROR parsing Excel {url}: {e}"


def datagovin_search(query: str, **kwargs) -> str:
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


DATASET_CACHE: dict[str, "pd.DataFrame"] = {}


def fetch_dataset(url: str) -> str:
    import pandas as pd

    global DATASET_CACHE

    if url in DATASET_CACHE:
        df = DATASET_CACHE[url]
    else:
        if url.endswith(".csv"):
            df = pd.read_csv(url)
        elif url.endswith((".xlsx", ".xls")):
            df = pd.read_excel(url)
        elif url.endswith(".json"):
            df = pd.read_json(url)
        else:
            raise ValueError(f"Unsupported dataset format: {url}")

        DATASET_CACHE[url] = df

    # -------- Small Preview --------

    preview = df.head(3).to_string(index=False)

    info = f"""
    Loaded dataset successfully.

    URL:
    {url}

    Rows: {len(df)}
    Columns: {len(df.columns)}

    Column names:
    {", ".join(map(str, df.columns))}

    Preview (first 3 rows):

    {preview}

    """

    return info


def analyze_image(image_url: str, question: str, **kwargs) -> str:
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
