import json
import re


def extract_json_value(raw_text: str):
    """Try to parse raw_text as JSON. Strip markdown fences if present."""
    text = raw_text.strip()
    # strip ```json ... ``` or ``` ... ``` fences if the model added them anyway
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    return json.loads(text)


def format_final_answer(raw_text: str, original_question: str, log_url: str) -> dict:
    try:
        answer_value = extract_json_value(raw_text)
    except Exception as e:
        print(f"[FORMAT ERROR] could not parse model output as JSON: {e}")
        answer_value = None  # fallback: raw string, better than crashing

    # Safety net: if the model ignored instructions and STILL wrapped it in
    # {"answer": ...}, unwrap one level automatically rather than double-nesting.
    if isinstance(answer_value, dict) and set(answer_value.keys()) == {"answer"}:
        print("[FORMAT WARN] model double-wrapped answer, auto-unwrapping")
        answer_value = answer_value["answer"]
    elif (
        isinstance(answer_value, dict)
        and "answer" in answer_value
        and "log_url" in answer_value
    ):
        print("[FORMAT WARN] model emitted full envelope, auto-unwrapping")
        answer_value = answer_value["answer"]

    return {"answer": answer_value, "log_url": log_url}
