import json


def format_final_answer(raw_text: str, original_question: str) -> dict:
    """Placeholder — Phase 6 will replace this with strict schema validation."""
    try:
        parsed = json.loads(raw_text)
    except Exception:
        parsed = {"error": "could not parse model output", "raw": raw_text}
    return {"answer": parsed, "log_url": "https://your-host/run.jsonl"}
