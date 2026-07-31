from agent.tools_web import (
    analyze_image,
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
            "name": "web_search_tool",
            "description": (
                "Search the web for relevant pages. "
                "Use ONLY when the URL is unknown. "
                "After finding a page, use web_fetch, fetch_dataset, "
                "fetch_pdf_tables or fetch_excel_table."
            ),
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
            "name": "web_fetch",
            "description": (
                "Fetch readable text from an HTML webpage. "
                "If the requested answer appears directly in the page, "
                "return it immediately. "
                "Do NOT use run_python unless actual calculations are needed."
            ),
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
            "name": "fetch_table_from_url",
            "description": (
                "Extract an HTML table from a webpage. "
                "Use ONLY if the page contains HTML tables."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "table_index": {
                        "type": "integer",
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
            "name": "fetch_pdf_tables",
            "description": (
                "Download a PDF and extract tables. "
                "If the answer appears directly in an extracted table, "
                "return it without using run_python."
            ),
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
            "description": ("Download an Excel file and extract its contents."),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "sheet_name": {"type": "string"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_dataset",
            "description": (
                "MANDATORY for every downloadable dataset.\n"
                "If the user message contains a URL ending in "
                ".csv, .tsv, .xls, .xlsx or .json, "
                "this MUST be the first tool called.\n"
                "Never answer dataset questions without calling this tool first."
            ),
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
            "name": "run_python",
            "description": (
                "Execute Python code.\n"
                "IMPORTANT: Pass ONE JSON object with a single field named 'code'.\n"
                "Example:\n"
                '{"code":"print(1+1)"}\n'
                "The value of 'code' must be a string containing valid Python."
            ),
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_image",
            "description": (
                "Analyze a chart, graph, screenshot or table image. "
                "Extract values or answer questions from the image. "
                "Use run_python afterwards only if calculations are required."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_url": {"type": "string"},
                    "question": {"type": "string"},
                },
                "required": [
                    "image_url",
                    "question",
                ],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "web_search_tool": web_search_tool,
    "web_fetch": web_fetch,
    "fetch_table_from_url": fetch_table_from_url,
    "fetch_pdf_tables": fetch_pdf_tables,
    "fetch_excel_table": fetch_excel_table,
    "fetch_dataset": fetch_dataset,
    "run_python": run_python,
    "analyze_image": analyze_image,
}
