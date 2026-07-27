import re
import time

from agent.orchestrator import run_agent_and_format
from logger import LOG_URL, flush_log, make_logger

# ---- simple in-memory session store (module-level, since Telethon has no built-in context.chat_data) ----
sessions: dict[int, dict] = {}


def get_session(chat_id: int) -> dict:
    if chat_id not in sessions:
        sessions[chat_id] = {"messages": [], "processed_ids": set()}
    return sessions[chat_id]


def expects_json_reply(text: str) -> bool:
    """Detect whether this message is asking for the final strict-JSON answer."""
    return bool(re.search(r"reply with (only\s+)?(this\s+)?json", text, re.IGNORECASE))


async def handle_message(event, context=None) -> dict:
    """
    event: Telethon NewMessage.Event
    context: unused here (kept for signature parity / future use), caller passes None
    Returns: a plain dict that the caller will json.dumps() and send as the reply.
    """
    chat_id = event.chat_id
    message_id = event.message.id
    text = event.raw_text or ""

    session = get_session(chat_id)

    # idempotency guard — skip if we've already processed this exact message
    if message_id in session["processed_ids"]:
        return {"status": "duplicate_skipped"}
    session["processed_ids"].add(message_id)

    session["messages"].append({"role": "user", "text": text, "ts": time.time()})

    print(f"[RECEIVED] chat_id={chat_id} msg_id={message_id} text={text!r}")
    print(f"[DEBUG] expects_json_reply({text!r}) = {expects_json_reply(text)}")
    if expects_json_reply(text):
        # Final turn — run the real agent pipeline and produce the strict answer.
        log_fn, log_buffer = make_logger(chat_id, qid=message_id)
        answer_payload = await run_agent_and_format(
            session["messages"],
            timeout_seconds=60,
            log_fn=log_fn,
        )
        log_fn({"event": "final_answer", "answer": answer_payload})

        # flush to public bucket BEFORE sending the Telegram reply
        flush_log(log_buffer)
        session["messages"].append({"role": "assistant", "text": str(answer_payload)})
        answer_payload["log_url"] = LOG_URL
        return answer_payload  # e.g. {"answer": ..., "log_url": "..."}
    else:
        # Intermediate turn — acknowledge, don't emit the final answer shape yet.
        ack = {"status": "received"}
        session["messages"].append({"role": "assistant", "text": str(ack)})
        return ack
