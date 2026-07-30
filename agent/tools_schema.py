from agent.tools_web import (
    datagovin_search,
    fetch_dataset,
    fetch_excel_table,
    fetch_pdf_tables,
    fetch_table_from_url,
    run_python,
    web_fetch,
    web_search_tool,
)

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch the raw text/HTML content of a URL. Use for MOSPI pages, data.gov.in, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The full URL to fetch."}
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_table_from_url",
            "description": "Fetch a URL and parse the Nth HTML table on the page into CSV text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "table_index": {
                        "type": "integer",
                        "description": "0-indexed table position",
                        "default": 0,
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Execute a Python snippet for data computation. IMPORTANT: the 'code' argument "
                "must be a valid JSON string — use \\n for newlines, do not include raw line breaks. "
                "Keep code compact; prefer semicolons or a single expression where possible. "
                "Always end with a print() statement for the result — only stdout is captured."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python source, JSON-escaped, must end with print(...)",
                    }
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search_tool",
            "description": "Search the web for a query and return top result URLs + snippets. Use to locate MOSPI/data.gov.in pages before fetching.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_pdf_tables",
            "description": "Download a PDF and extract tables from it as CSV text. Use for MOSPI PDF releases.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_excel_table",
            "description": "Download an Excel file and return its contents as CSV text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "sheet_name": {
                        "type": "string",
                        "description": "Optional sheet name",
                    },
                },
                "required": ["url"],
            },
        },
    },
]
TOOL_SCHEMAS.append(
    {
        "type": "function",
        "function": {
            "name": "fetch_dataset",
            "description": (
                "Download a CSV/Excel/JSON/TSV file from a URL and return a preview "
                "(columns, dtypes, sample rows). The full dataset is cached and can be "
                "loaded in run_python via get_cached_dataset(url) for real analysis. "
                "Use this whenever a question links to a downloadable data file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Direct URL to the data file.",
                    }
                },
                "required": ["url"],
            },
        },
    }
)

# map name -> actual callable, for dispatch
TOOL_FUNCTIONS = {
    "web_fetch": web_fetch,
    "fetch_table_from_url": fetch_table_from_url,
    "run_python": run_python,
    "fetch_dataset": fetch_dataset,
    "web_search_tool": web_search_tool,
    "fetch_pdf_tables": fetch_pdf_tables,
    "fetch_excel_table": fetch_excel_table,
}
