"""
dialogue_manager.py
--------------------
Multi-turn conversation state machine layered on top of InferenceEngine.

Flow per session:
    IDLE
      -> user describes a problem
    CLARIFYING
      -> bot asks up to N clarifying questions one at a time (from the
         knowledge base's `clarifying_questions`) before committing to a
         diagnosis. Answers are folded back into the diagnostic text so the
         ML/rule layers can re-score with more information.
    DIAGNOSED
      -> bot presents the diagnosis (causes + full step list) once.
    GUIDING
      -> bot walks the user through `solution_steps` one at a time, waiting
         for a reply after each step (e.g. "did that work?") before moving on.
         If the user reports success, the session resolves. If all steps are
         exhausted without success, the bot recommends professional help.
    RESOLVED / ESCALATED
      -> terminal states; a new problem description restarts the flow.

State is persisted per session_id via database.get_dialogue_state /
save_dialogue_state, so it survives across stateless HTTP requests.
"""

import json
import re

from inference_engine import InferenceEngine
import database

MAX_CLARIFYING_QUESTIONS = 2  # keep it brisk — don't interrogate the user
NEGATIVE_PATTERNS = re.compile(
    r"\b(no|nope|not\s+really|didn'?t\s+work|doesn'?t\s+work|still\s+(not|broken|the\s+same)|"
    r"no\s+change|same\s+(issue|problem)|negative)\b", re.IGNORECASE
)
POSITIVE_PATTERNS = re.compile(
    r"\b(yes|yep|yeah|it\s+work(ed|s)?|that\s+(fixed|worked)|fixed\s+it|resolved|working\s+now|"
    r"solved|all\s+good|perfect)\b", re.IGNORECASE
)
RESTART_PATTERNS = re.compile(
    r"\b(new\s+(issue|problem)|different\s+(issue|problem)|another\s+(issue|problem)|"
    r"something\s+else|start\s+over)\b", re.IGNORECASE
)
SMALL_TALK_PATTERNS = re.compile(
    r"^\s*(hi|hello|hey|good\s+(morning|afternoon|evening)|thanks|thank\s+you)"
    r"(\s+there|\s+everyone)?\s*[!.?]*\s*$",
    re.IGNORECASE,
)


class DialogueManager:
    def __init__(self):
        self.engine = InferenceEngine()

    # ---------- state persistence ----------
    def _load_state(self, session_id):
        raw = database.get_dialogue_state(session_id)
        if not raw:
            return {"stage": "idle"}
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {"stage": "idle"}

    def _save_state(self, session_id, state):
        database.save_dialogue_state(session_id, json.dumps(state))

    def reset(self, session_id):
        database.clear_dialogue_state(session_id)

    def _compose_context(self, user_message, image_finding=None, conversation_history=None):
        parts = []

        if conversation_history:
            history_lines = []
            for item in conversation_history:
                role = item.get("role", "").strip().lower()
                content = (item.get("content") or "").strip()
                if role in {"user", "assistant"} and content:
                    history_lines.append(f"{role.title()}: {content}")
            if history_lines:
                parts.append("Conversation history:\n" + "\n".join(history_lines))

        if user_message:
            parts.append(f"Current user message: {user_message}")

        if image_finding and image_finding.get("description"):
            parts.append(f"Image shows: {image_finding['description']}")

        return "\n\n".join(parts) if parts else user_message

    def _is_small_talk(self, user_message):
        return bool(SMALL_TALK_PATTERNS.match(user_message or ""))

    def _small_talk_category(self, user_message):
        lowered = (user_message or "").lower()
        if any(word in lowered for word in ("thanks", "thank you")):
            return "thanks"
        return "greeting"

    # ---------- main entry point ----------
    def handle_message(self, session_id, user_message, image_finding=None, conversation_history=None):
        """
        image_finding: optional dict {description, suspected_category} produced
        by the Gemini vision analysis on the client, merged in as extra evidence.
        """
        state = self._load_state(session_id)
        stage = state.get("stage", "idle")

        if RESTART_PATTERNS.search(user_message) and stage != "idle":
            self.reset(session_id)
            state = {"stage": "idle"}
            stage = "idle"

        if self._is_small_talk(user_message):
            self.reset(session_id)
            return {
                "type": "conversational",
                "category": self._small_talk_category(user_message),
            }

        if stage == "idle":
            return self._start_diagnosis(session_id, state, user_message, image_finding, conversation_history)
        elif stage == "clarifying":
            return self._continue_clarifying(session_id, state, user_message, conversation_history)
        elif stage == "guiding":
            return self._continue_guiding(session_id, state, user_message)
        else:
            # diagnosed/resolved/escalated -> treat new message as a fresh problem
            self.reset(session_id)
            return self._start_diagnosis(session_id, {"stage": "idle"}, user_message, image_finding, conversation_history)

    # ---------- stage: starting a new diagnosis ----------
    def _start_diagnosis(self, session_id, state, user_message, image_finding, conversation_history=None):
        combined_text = self._compose_context(user_message, image_finding=image_finding, conversation_history=conversation_history)

        result = self.engine.diagnose(combined_text)
        entities_text = ", ".join(result.get("entities", [])) if result else ""

        if result.get("is_conversational"):
            self.reset(session_id)
            return {
                "type": "conversational",
                "category": result["category"],
            }

        # Image evidence can rescue a no_match, override a weak text-only
        # guess, or reinforce an agreeing one — so this runs BEFORE the
        # no_match early-return, not after.
        if image_finding and image_finding.get("suspected_category") in self.engine.kb:
            img_cat = image_finding["suspected_category"]
            text_confidence = result.get("confidence") or 0.0
            text_category = result.get("category")
            if text_category is None or text_confidence < 0.6 or img_cat == text_category:
                kb_entry = self.engine.kb[img_cat]
                prior_method = result.get("method") or "none"
                result = {
                    **result,
                    "category": img_cat,
                    "label": kb_entry["label"],
                    "causes": kb_entry["causes"],
                    "solution_steps": kb_entry["solution_steps"],
                    "severity": kb_entry["severity"],
                    "confidence": max(text_confidence, 0.75),
                    "method": (prior_method + "+image") if prior_method != "no_match" else "image_only",
                }

        if result["category"] is None:
            return {
                "type": "no_match",
            }

        clarifying = self.engine.kb.get(result["category"], {}).get("clarifying_questions", [])
        clarifying = clarifying[:MAX_CLARIFYING_QUESTIONS]

        state = {
            "stage": "clarifying" if clarifying else "diagnosed",
            "category": result["category"],
            "original_message": user_message,
            "pending_questions": clarifying,
            "asked_index": 0,
            "answers": {},
            "last_diagnosis": result,
        }

        if clarifying:
            self._save_state(session_id, state)
            return {
                "type": "clarifying_question",
                "diagnosis_preview": result,
                "question": clarifying[0]["prompt"],
                "question_number": 1,
                "question_total": len(clarifying),
            }

        # No clarifying questions defined for this category -> go straight to diagnosis
        return self._present_diagnosis(session_id, state)

    # ---------- stage: clarifying ----------
    def _continue_clarifying(self, session_id, state, user_message, conversation_history=None):
        idx = state["asked_index"]
        questions = state["pending_questions"]
        question_id = questions[idx]["id"]
        state["answers"][question_id] = user_message

        idx += 1
        state["asked_index"] = idx

        if idx < len(questions):
            self._save_state(session_id, state)
            return {
                "type": "clarifying_question",
                "question": questions[idx]["prompt"],
                "question_number": idx + 1,
                "question_total": len(questions),
            }

        # All clarifying questions answered -> re-run diagnosis with enriched context
        enriched_text = self._compose_context(
            state["original_message"] + ". " + " ".join(
                f"{q['id']}: {state['answers'].get(q['id'], '')}" for q in questions
            ),
            conversation_history=conversation_history,
        )
        result = self.engine.diagnose(enriched_text)

        # If re-diagnosis lands somewhere with lower confidence than the original
        # category guess, keep the original category (it had real keyword signal)
        # but use the refined confidence/method for transparency.
        if result["category"] != state["category"] and result["confidence"] < 0.5:
            result["category"] = state["category"]
            kb_entry = self.engine.kb[state["category"]]
            result["label"] = kb_entry["label"]
            result["causes"] = kb_entry["causes"]
            result["solution_steps"] = kb_entry["solution_steps"]
            result["severity"] = kb_entry["severity"]

        state["last_diagnosis"] = result
        state["category"] = result["category"]
        return self._present_diagnosis(session_id, state)

    # ---------- present full diagnosis, then move into guided steps ----------
    def _present_diagnosis(self, session_id, state):
        result = state["last_diagnosis"]
        steps = self.engine.kb[state["category"]]["solution_steps"]

        state["stage"] = "guiding"
        state["step_index"] = 0
        state["total_steps"] = len(steps)
        self._save_state(session_id, state)

        return {
            "type": "diagnosis",
            "category": state["category"],
            "fault_label": result["label"],
            "confidence": result["confidence"],
            "severity": result["severity"],
            "causes": result["causes"],
            "method": result["method"],
            "first_step": steps[0]["step"],
            "first_step_check": steps[0].get("check_prompt"),
            "step_number": 1,
            "step_total": len(steps),
        }

    # ---------- stage: guiding through steps ----------
    def _continue_guiding(self, session_id, state, user_message):
        steps = self.engine.kb[state["category"]]["solution_steps"]
        idx = state["step_index"]

        if POSITIVE_PATTERNS.search(user_message):
            self.reset(session_id)
            return {"type": "resolved"}

        negative = bool(NEGATIVE_PATTERNS.search(user_message))
        idx += 1

        if idx >= len(steps):
            self.reset(session_id)
            return {"type": "escalate", "category_label": self.engine.kb[state["category"]]["label"]}

        state["step_index"] = idx
        self._save_state(session_id, state)

        return {
            "type": "next_step",
            "acknowledged_negative": negative,
            "step": steps[idx]["step"],
            "step_check": steps[idx].get("check_prompt"),
            "step_number": idx + 1,
            "step_total": len(steps),
        }


if __name__ == "__main__":
    dm = DialogueManager()
    sid = "demo-session"
    dm.reset(sid)

    print(json.dumps(dm.handle_message(sid, "my pc randomly shuts down and it feels really hot"), indent=2))
    print(json.dumps(dm.handle_message(sid, "happens even when idle, not just gaming"), indent=2))
    print(json.dumps(dm.handle_message(sid, "about 2 years, never cleaned"), indent=2))
    print(json.dumps(dm.handle_message(sid, "no that didn't help"), indent=2))
    print(json.dumps(dm.handle_message(sid, "yes that fixed it!"), indent=2))
