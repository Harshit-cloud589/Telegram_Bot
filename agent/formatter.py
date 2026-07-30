import json
import re


def extract_json_value(raw_text: str):
    """Extract JSON from model output, removing markdown fences if present."""

    text = raw_text.strip()

    # Remove ```json ... ``` or ``` ... ```
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    text = text.strip()

    return json.loads(text)


def format_final_answer(raw_text: str, original_question: str, log_url: str) -> dict:
    """
    Convert the model's raw response into the required API format.

    The model is instructed to return ONLY the value that belongs
    inside "answer". This function wraps it with log_url.
    """

    try:
        answer_value = extract_json_value(raw_text)

    except Exception as e:
        print(f"[FORMAT ERROR] {e}")

        # Fallbacks

        text = raw_text.strip()

        # number
        try:
            answer_value = int(text)
        except ValueError:
            try:
                answer_value = float(text)
            except ValueError:
                # true / false
                if text.lower() == "true":
                    answer_value = True

                elif text.lower() == "false":
                    answer_value = False

                elif text.lower() == "null":
                    answer_value = None

                else:
                    # plain string
                    answer_value = text

    # Model accidentally returned {"answer": ...}
    if isinstance(answer_value, dict):
        if set(answer_value.keys()) == {"answer"}:
            print("[FORMAT] Auto-unwrapped answer")
            answer_value = answer_value["answer"]

        elif "answer" in answer_value and "log_url" in answer_value:
            print("[FORMAT] Auto-unwrapped full envelope")
            answer_value = answer_value["answer"]

    return {
        "answer": answer_value,
        "log_url": log_url,
    }
