"""
inference_engine.py
--------------------
The core hybrid diagnostic engine (Chapter 3.8 - Hybrid Rule-Based + ML
Classification Module).

Combines:
  1. ML classifier (TF-IDF + Linear SVM/Naive Bayes) -> probabilistic fault category
  2. Rule-based keyword/knowledge-base matching      -> deterministic confirmation
  3. Confidence fusion                               -> final diagnosis + confidence score

If the ML model and rule engine agree, confidence is boosted.
If they disagree, the rule engine (deterministic, explainable) takes priority,
since hardware fault domains are safety/cost-sensitive and explainability matters.
"""

import os
import json
import joblib
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.nlp_preprocessor import preprocess, extract_entities

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
KB_PATH = os.path.join(BASE_DIR, "data", "knowledge_base.json")

GREETING_CATEGORIES = {"greeting", "thanks"}
SPECIAL_RULE_OVERRIDES = [
    ("boot_recovery_issue", [
        "sure recover",
        "operating system not found",
        "no operating system was found",
        "boot device",
        "startup repair",
        "content key",
        "recovery screen",
        "restore from network",
        "restore from local drive",
    ]),
    ("bios_firmware_issue", [
        "bios",
        "uefi",
        "secure boot",
        "boot order",
        "firmware",
    ]),
    ("driver_issue", [
        "device manager",
        "rollback driver",
        "driver has stopped responding",
        "graphics driver",
        "update driver",
        "reinstall driver",
    ]),
    ("application_issue", [
        "not responding",
        "app crash",
        "application crash",
        "freeze",
        "software bug",
    ]),
]


class InferenceEngine:
    def __init__(self):
        self.kb = self._load_knowledge_base()
        self.model, self.labels, self.model_name = self._load_model()

    def _load_knowledge_base(self):
        with open(KB_PATH, "r") as f:
            return json.load(f)

    def _load_model(self):
        model_path = os.path.join(MODEL_DIR, "intent_classifier.pkl")
        labels_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
        name_path = os.path.join(MODEL_DIR, "model_name.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "Trained model not found. Run `python3 train_model.py` first."
            )
        model = joblib.load(model_path)
        labels = joblib.load(labels_path)
        model_name = joblib.load(name_path) if os.path.exists(name_path) else "unknown"
        return model, labels, model_name

    # ---------- Rule-based layer ----------
    def rule_based_match(self, raw_text: str):
        """
        Scores every knowledge-base category by counting how many of its
        keyword phrases appear in the user's message. Returns the best match
        and its score (number of distinct keyword hits).
        """
        text_lower = raw_text.lower()

        for category, phrases in SPECIAL_RULE_OVERRIDES:
            if any(phrase in text_lower for phrase in phrases):
                return category, max(2, sum(1 for phrase in phrases if phrase in text_lower))

        scores = {}
        for category, entry in self.kb.items():
            hits = sum(1 for kw in entry["keywords"] if kw in text_lower)
            if hits > 0:
                scores[category] = hits

        if not scores:
            return None, 0
        best_category = max(scores, key=scores.get)
        return best_category, scores[best_category]

    # ---------- ML layer ----------
    def _has_vocabulary_overlap(self, clean_text: str) -> bool:
        """
        Returns True if at least TWO distinct tokens in the cleaned input
        appear in the TF-IDF vocabulary learned from the training data.
        A linear classifier will still assign *some* class with non-zero
        confidence to pure noise (e.g. random keysmash), and a single
        generic overlapping word (e.g. "random") isn't enough evidence on
        its own — this acts as a sanity gate before trusting any ML-only
        weak-signal prediction.
        """
        try:
            vectorizer = self.model.named_steps["tfidf"]
            vocab = vectorizer.vocabulary_
        except (KeyError, AttributeError):
            return True  # if we can't introspect it, don't block on this check

        tokens = set(clean_text.split())
        matches = tokens & vocab.keys()
        return len(matches) >= 2

    def ml_predict(self, raw_text: str):
        clean = preprocess(raw_text, for_ml=True)
        if not clean.strip():
            return None, 0.0

        predicted = self.model.predict([clean])[0]

        # Confidence score: LinearSVC has decision_function, NB has predict_proba
        try:
            if hasattr(self.model.named_steps["clf"], "predict_proba"):
                proba = self.model.predict_proba([clean])[0]
                confidence = float(max(proba))
            else:
                decision = self.model.decision_function([clean])[0]
                # softmax-normalize the decision scores into a pseudo-probability
                import numpy as np
                exp_scores = np.exp(decision - np.max(decision))
                softmax = exp_scores / exp_scores.sum()
                confidence = float(max(softmax))
        except Exception:
            confidence = 0.5

        if not self._has_vocabulary_overlap(clean):
            # No recognizable hardware-related vocabulary at all — treat as
            # no signal rather than trusting a classifier guess made purely
            # on the (meaningless, for unseen tokens) decision boundary.
            confidence = 0.0

        return predicted, confidence

    # ---------- Fusion logic ----------
    def diagnose(self, raw_text: str):
        ml_category, ml_confidence = self.ml_predict(raw_text)
        rule_category, rule_hits = self.rule_based_match(raw_text)
        entities = extract_entities(raw_text)

        # Handle conversational categories early
        if ml_category in GREETING_CATEGORIES and rule_category is None:
            return {
                "category": ml_category,
                "is_conversational": True,
                "confidence": round(ml_confidence, 2),
                "entities": entities,
            }

        final_category = None
        confidence = 0.0
        method = ""

        if rule_category and ml_category == rule_category:
            # Agreement: highest confidence
            base = self.kb[rule_category]["confidence_base"]
            confidence = min(0.98, base + 0.05 * rule_hits)
            final_category = rule_category
            method = "hybrid_agreement"
        elif rule_category and rule_category not in self.labels:
            # Categories added to the knowledge base after model training
            # must rely on deterministic matching because the ML model cannot
            # predict them yet.
            base = self.kb[rule_category]["confidence_base"]
            confidence = min(0.96, base + 0.03 * rule_hits)
            final_category = rule_category
            method = "rule_based_new_category"
        elif rule_category and rule_hits >= 2:
            # Strong deterministic keyword evidence overrides a weak ML guess
            base = self.kb[rule_category]["confidence_base"]
            confidence = min(0.95, base)
            final_category = rule_category
            method = "rule_based_dominant"
        elif ml_category and ml_category not in GREETING_CATEGORIES and ml_confidence >= 0.4:
            final_category = ml_category
            confidence = ml_confidence
            method = "ml_classifier"
        elif rule_category:
            base = self.kb[rule_category]["confidence_base"]
            confidence = max(0.3, base - 0.2)
            final_category = rule_category
            method = "rule_based_weak"
        elif ml_category and ml_category not in GREETING_CATEGORIES and ml_confidence >= 0.2:
            # Weak ML-only signal — still surface it (with low confidence) rather
            # than dead-ending the conversation; the dialogue layer will ask
            # clarifying questions to firm this up before fully committing.
            final_category = ml_category
            confidence = ml_confidence
            method = "ml_classifier_weak"
        else:
            return {
                "category": None,
                "is_conversational": False,
                "confidence": 0.0,
                "entities": entities,
                "method": "no_match",
            }

        kb_entry = self.kb.get(final_category, {})
        return {
            "category": final_category,
            "label": kb_entry.get("label", final_category),
            "is_conversational": False,
            "confidence": round(confidence, 2),
            "causes": kb_entry.get("causes", []),
            "solution_steps": kb_entry.get("solution_steps", []),
            "severity": kb_entry.get("severity", "unknown"),
            "entities": entities,
            "method": method,
            "ml_suggestion": ml_category,
            "ml_confidence": round(ml_confidence, 2),
        }


if __name__ == "__main__":
    engine = InferenceEngine()
    tests = [
        "My computer won't turn on, no lights or fans at all",
        "I keep getting random blue screen errors and beep codes",
        "My laptop fan is loud and it shuts down because it's too hot",
        "hard drive is clicking and windows can't find boot device",
        "hi there",
    ]
    for t in tests:
        result = engine.diagnose(t)
        print("\nINPUT:", t)
        print(json.dumps(result, indent=2))
