"""
chat/services.py
----------------
Chatbot inference services for the Django backend.

The LightweightIntentPredictor is the primary runtime predictor when the full
SBERT+Keras model is not available (default for most deployments). It uses an
improved lexical scorer with:
  - Stop-word filtering (content-word aware scoring)
  - Exact/near-exact string match fast path
  - Phrase containment check
  - Mutual content-word coverage for specificity
"""
import json
import os
import random
import re
from difflib import SequenceMatcher
from pathlib import Path

# Lazy singleton — loaded once per process
_bot = None

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "intents.json"
MODEL_META_PATH = ROOT / "models" / "model_meta.json"

# Stop-words that don't contribute to intent disambiguation
_STOP_WORDS = {
    "is", "the", "of", "a", "an", "are", "do", "does", "was", "were",
    "what", "who", "when", "where", "how", "can", "i", "me", "my",
    "we", "you", "it", "in", "for", "to", "and", "or", "there",
    "please", "tell", "give", "let", "name", "any", "about", "at",
    "on", "with", "from", "that", "this", "they", "be", "have", "has",
    "not", "if", "will", "would", "should", "could", "get",
}


def _normalize_text(value: str) -> str:
    """Lowercase, collapse repeated chars, strip punctuation, normalize spaces."""
    value = (value or "").strip().lower()
    # Collapse 3+ repeated letters: cooool -> cool
    value = re.sub(r"(.)\1{2,}", r"\1\1", value)
    # Strip punctuation, keep letters/digits/spaces
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _score_pair(text: str, pattern: str) -> float:
    """
    Compute a similarity score between a user query and a single intent pattern.
    Returns a float in [0, 1].
    """
    text_norm = _normalize_text(text)
    pattern_norm = _normalize_text(pattern)
    if not text_norm or not pattern_norm:
        return 0.0

    text_words = set(text_norm.split())
    pattern_words = set(pattern_norm.split())
    content_words = text_words - _STOP_WORDS
    pattern_content = pattern_words - _STOP_WORDS

    # ── 1. Near-exact string match ──────────────────────────────────────────
    ratio = SequenceMatcher(None, text_norm, pattern_norm).ratio()
    if ratio >= 0.95:
        return 0.99

    # ── 2. Phrase containment ────────────────────────────────────────────────
    phrase_score = 0.0
    if len(text_words) >= 2 and text_norm in pattern_norm:
        phrase_score = 0.98
    elif len(pattern_words) >= 2 and pattern_norm in text_norm:
        phrase_score = 0.98
    elif len(text_words) == 1 and (text_words & pattern_words):
        phrase_score = 0.95

    # ── 3. Fuzzy word-level overlap (all words) ─────────────────────────────
    def fuzzy_matches(src_words, tgt_words):
        count = 0
        for w in src_words:
            for pw in tgt_words:
                if w == pw or (
                    min(len(w), len(pw)) >= 4
                    and SequenceMatcher(None, w, pw).ratio() >= 0.82
                ):
                    count += 1
                    break
        return count

    fuzz = fuzzy_matches(text_words, pattern_words)
    overlap = fuzz / max(len(text_words | pattern_words), 1)
    coverage = fuzz / max(len(text_words), 1)
    pattern_coverage = fuzz / max(len(pattern_words), 1)

    # ── 4. Content-word mutual coverage (rewards specificity) ───────────────
    mutual_content = 0.0
    if content_words and pattern_content:
        cf = fuzzy_matches(content_words, pattern_content)
        cc = cf / max(len(content_words), 1)
        pc = cf / max(len(pattern_content), 1)
        mutual_content = (cc + pc) / 2

    return max(
        phrase_score,
        overlap,
        (coverage * 0.55) + (pattern_coverage * 0.45),
        ratio * 0.72,
        mutual_content * 0.90,
    )


class LightweightIntentPredictor:
    """
    Rule-based intent predictor for demos/production when the trained SBERT
    model is unavailable. Uses the same improved scoring logic as src/utils.py.
    """

    THRESHOLD = 0.40          # minimum confidence to give a real answer
    NN_THRESHOLD = 0.50       # minimum for nearest-neighbour fallback

    def __init__(self, threshold: float = 0.40):
        self.threshold = float(threshold)

        with open(DATA_PATH, "r", encoding="utf-8") as f:
            intents_json = json.load(f)

        self.intents = intents_json.get("intents", [])
        self.classes = [intent["tag"] for intent in self.intents]

        self.tag_to_responses = {
            intent["tag"]: intent.get("responses", [])
            for intent in self.intents
        }
        self.intent_metadata = {
            intent["tag"]: {
                "follow_up_intents": intent.get("follow_up_intents", []),
                "metadata": intent.get("metadata", {}),
            }
            for intent in self.intents
        }

        # Flat list of (tag, pattern_text) pairs for scoring
        self.patterns: list = []
        for intent in self.intents:
            for p in intent.get("patterns", []):
                if p and p.strip():
                    self.patterns.append((intent["tag"], p.strip()))

    # ── Core prediction ──────────────────────────────────────────────────────

    def _compute_scores(self, text: str) -> dict:
        """Return per-tag best score against all patterns."""
        scores = {tag: 0.0 for tag in self.classes}
        for tag, pattern in self.patterns:
            s = _score_pair(text, pattern)
            if s > scores.get(tag, 0.0):
                scores[tag] = s
        return scores

    def predict_intent(self, text: str) -> dict:
        scores = self._compute_scores(text)
        if not scores:
            return {"tag": "fallback", "confidence": 0.0, "probs_by_tag": {}}
        best_tag = max(scores, key=scores.get)
        confidence = float(scores[best_tag])
        total = sum(scores.values()) or 1.0
        probs = {tag: float(v / total) for tag, v in scores.items()}
        return {"tag": best_tag, "confidence": confidence, "probs_by_tag": probs}

    def top_k(self, text: str, k: int = 3) -> list:
        scores = self._compute_scores(text)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

    # ── Answer generation ────────────────────────────────────────────────────

    def answer(self, user_text: str) -> str:
        """Return an answer for user_text alone (no history blending)."""
        pred = self.predict_intent(user_text)
        if pred["confidence"] < self.threshold:
            return "Sorry, I'm not sure about that. Could you rephrase?"
        responses = self.tag_to_responses.get(pred["tag"], [])
        if responses:
            return random.choice(responses)
        return f"(No response template for: {pred['tag']})"

    def _answer_for_tag(self, tag: str) -> str:
        responses = self.tag_to_responses.get(tag, [])
        return random.choice(responses) if responses else f"(No response for: {tag})"

    def get_prediction_with_uncertainty(self, text: str) -> dict:
        pred = self.predict_intent(text)
        uncertainty = max(0.0, min(1.0, 1.0 - pred["confidence"]))
        pred["uncertainty"] = uncertainty
        pred["is_uncertain"] = bool(
            uncertainty > 0.7 or pred["confidence"] < self.threshold
        )
        return pred

    def answer_with_followup_suggestions(
        self, user_text: str, conversation_history: list = None
    ) -> dict:
        """
        Predict on the CURRENT message only (precise), then optionally boost
        with history if confidence is low (avoids history contamination).
        """
        conversation_history = conversation_history or []

        # Primary prediction: score current message alone
        pred = self.get_prediction_with_uncertainty(user_text)

        # If primary confidence is low, try blending with recent context
        if pred["confidence"] < self.threshold and conversation_history:
            recent = " ".join(conversation_history[-2:]).strip()
            combined = f"{recent} {user_text}".strip()
            pred_ctx = self.get_prediction_with_uncertainty(combined)
            # Only use context-boosted result if it's genuinely more confident
            if pred_ctx["confidence"] > pred["confidence"]:
                pred = pred_ctx

        tag = pred["tag"]
        conf = pred["confidence"]

        if conf >= self.threshold:
            reply = self._answer_for_tag(tag)
        else:
            reply = "Sorry, I'm not sure about that. Could you rephrase?"

        metadata = self.intent_metadata.get(tag, {})
        return {
            "reply": reply,
            "tag": tag,
            "confidence": conf,
            "uncertainty": pred["uncertainty"],
            "is_uncertain": pred["is_uncertain"],
            "suggested_followups": metadata.get("follow_up_intents", []),
            "top3": self.top_k(user_text, 3),
            "context_used": bool(conversation_history),
        }


# ── Singleton factory ────────────────────────────────────────────────────────

def get_bot() -> LightweightIntentPredictor:
    """
    Lazy singleton. Loads the full SBERT+Keras model when the trained artifacts
    are present (SCB_ENABLE_FULL_MODEL=1). Falls back to the lightweight
    rule-based predictor otherwise.
    """
    global _bot
    if _bot is not None:
        return _bot

    should_try_full = os.environ.get("SCB_ENABLE_FULL_MODEL", "").lower() in {
        "1", "true", "yes",
    }

    if should_try_full and MODEL_META_PATH.exists():
        try:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            from src.utils import IntentPredictor  # noqa: PLC0415

            _bot = IntentPredictor(threshold=0.40, nn_sim_threshold=0.50)
            print("[INFO] Full SBERT+Keras intent model loaded.")
        except Exception as exc:
            print(f"[WARN] Full intent model unavailable, using lightweight fallback: {exc}")
            _bot = LightweightIntentPredictor(threshold=0.40)
    else:
        _bot = LightweightIntentPredictor(threshold=0.40)

    return _bot
