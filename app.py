"""
app.py
------
Flask web application (Chapter 3.8 - System Implementation / Web Interface Module).

Routes:
    GET  /                 -> chat UI
    POST /api/chat         -> send a message (+ optional image finding), get the
                              next turn of the multi-turn diagnosis conversation
    POST /api/feedback     -> mark a diagnosis as helpful / not helpful
    POST /api/reset        -> manually reset the conversation state for this session
    GET  /api/history      -> recent conversation log (for demo/admin viewing)
    GET  /api/stats        -> fault frequency counts

Image diagnosis note:
    Image *understanding* (Gemini Vision) runs client-side in the browser
    (static/js/vision.js), because this server's outbound network is
    sandboxed to a fixed allowlist that does not include Google's API host.
    The browser calls Gemini directly with the user's own API key, then
    posts the resulting finding (a short text description + suspected
    category) to /api/chat as `image_finding`, where it's merged into the
    same hybrid rule/ML pipeline used for text.
"""

import os
import uuid

from flask import Flask, render_template, request, jsonify, session

from dialogue_manager import DialogueManager
import database
import web_research

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")

# Load the model + knowledge base once at startup (not per-request)
dialogue = DialogueManager()
database.init_db()


GREETING_RESPONSES = {
    "greeting": "Hello! I'm your hardware diagnosis assistant. Describe the problem "
                "you're having with your computer, or attach a photo of it "
                "(e.g. \"my PC won't turn on\" or \"my laptop keeps overheating\") "
                "and I'll walk you through fixing it.",
    "thanks": "You're welcome! Feel free to describe another issue if you run into one.",
}


def _get_session_id():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return session["session_id"]


def _sanitize_history(raw_history):
    if not isinstance(raw_history, list):
        return []

    cleaned = []
    for item in raw_history[-12:]:
        if not isinstance(item, dict):
            continue
        role = (item.get("role") or "").strip().lower()
        content = (item.get("content") or "").strip()
        if role not in {"user", "assistant", "system"} or not content:
            continue
        cleaned.append({"role": role, "content": content})
    return cleaned


def _guess_research_category(user_message, image_description=None):
    text = " ".join(part for part in [image_description or "", user_message or ""] if part).lower()
    if any(term in text for term in [
        "sure recover", "operating system not found", "no operating system was found",
        "boot device", "startup repair", "content key", "recovery screen"
    ]):
        return "boot_recovery_issue"
    if any(term in text for term in ["bios", "uefi", "secure boot", "boot order", "firmware"]):
        return "bios_firmware_issue"
    if any(term in text for term in ["driver", "device manager", "rollback driver"]):
        return "driver_issue"
    if any(term in text for term in ["application", "not responding", "crash", "freeze", "software"]):
        return "application_issue"
    return None


@app.route("/")
def index():
    _get_session_id()
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    session_id = _get_session_id()
    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()
    image_finding = data.get("image_finding")  # optional dict from client-side Gemini call
    history = _sanitize_history(data.get("history"))

    if not user_message and not image_finding:
        return jsonify({"error": "Empty message"}), 400

    # If there's only an image with no caption, give the dialogue manager a
    # generic prompt so it still has text to run the NLP pipeline on.
    effective_message = user_message or "I'm attaching a photo of the issue."

    result = dialogue.handle_message(
        session_id,
        effective_message,
        image_finding=image_finding,
        conversation_history=history,
    )

    research = None
    if result.get("type") in {"diagnosis", "no_match"}:
        image_description = (image_finding or {}).get("description")
        category = result.get("category")
        if result.get("type") == "no_match":
            category = _guess_research_category(effective_message, image_description)
        research = web_research.research_issue(category, effective_message, image_description=image_description)

    return jsonify(_render_turn(session_id, effective_message, result, research))


def _render_turn(session_id, user_message, result, research=None):
    """Maps a DialogueManager result into the JSON contract the frontend expects."""
    rtype = result["type"]

    if rtype == "conversational":
        reply_text = GREETING_RESPONSES.get(result["category"], "How can I help you today?")
        return {"type": "conversational", "reply": reply_text}

    if rtype == "no_match":
        return {
            "type": "no_match",
            "reply": "I couldn't confidently identify the issue from that description. "
                     "Could you give more detail - what exactly happens, any sounds, "
                     "lights, or error messages you see? A photo can help too.",
            "research": research,
        }

    if rtype == "clarifying_question":
        return {
            "type": "clarifying_question",
            "question": result["question"],
            "question_number": result["question_number"],
            "question_total": result["question_total"],
        }

    if rtype == "diagnosis":
        conversation_id = database.log_conversation(
            session_id=session_id,
            user_message=user_message,
            fault_category=result.get("category"),
            confidence=result["confidence"],
            method=result["method"],
        )
        return {
            "type": "diagnosis",
            "conversation_id": conversation_id,
            "fault_label": result["fault_label"],
            "confidence": result["confidence"],
            "severity": result["severity"],
            "causes": result["causes"],
            "method": result["method"],
            "step": result["first_step"],
            "step_check": result["first_step_check"],
            "step_number": result["step_number"],
            "step_total": result["step_total"],
            "research": research,
        }

    if rtype == "next_step":
        return {
            "type": "next_step",
            "acknowledged_negative": result["acknowledged_negative"],
            "step": result["step"],
            "step_check": result["step_check"],
            "step_number": result["step_number"],
            "step_total": result["step_total"],
        }

    if rtype == "resolved":
        return {
            "type": "resolved",
            "reply": "Glad that fixed it! Let me know if anything else comes up.",
        }

    if rtype == "escalate":
        return {
            "type": "escalate",
            "reply": f"We've been through the standard steps for a "
                     f"{result['category_label'].lower()} and the issue is still "
                     f"there. At this point I'd recommend having a qualified "
                     f"technician take a look - it may need a hardware swap or "
                     f"diagnostic equipment I can't replicate here.",
        }

    return {"type": "error", "reply": "Something unexpected happened - let's start over."}


@app.route("/api/reset", methods=["POST"])
def reset():
    session_id = _get_session_id()
    dialogue.reset(session_id)
    return jsonify({"status": "ok"})


@app.route("/api/feedback", methods=["POST"])
def feedback():
    data = request.get_json(silent=True) or {}
    conversation_id = data.get("conversation_id")
    was_helpful = data.get("was_helpful")

    if conversation_id is None or was_helpful is None:
        return jsonify({"error": "conversation_id and was_helpful are required"}), 400

    database.log_feedback(conversation_id, bool(was_helpful))
    return jsonify({"status": "ok"})


@app.route("/api/history")
def history():
    rows = database.get_recent_conversations(limit=30)
    history_list = [
        {
            "id": r[0],
            "session_id": r[1],
            "message": r[2],
            "fault_category": r[3],
            "confidence": r[4],
            "method": r[5],
            "created_at": str(r[6]),
        }
        for r in rows
    ]
    return jsonify(history_list)


@app.route("/api/stats")
def stats():
    rows = database.get_fault_frequency()
    return jsonify([{"fault_category": r[0], "count": r[1]} for r in rows])


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
