# src/utils.py
import json
import os
import re
import random
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
DATA_PATH = ROOT / "data" / "intents.json"

def _normalize_text(s: str) -> str:
    """Lowercase, trim, collapse repeated letters (heyyy -> heyy), strip extra spaces."""
    s = s.strip().lower()
    # collapse 3+ repeated letters to 2 (cooool -> cool)
    s = re.sub(r"(.)\1{2,}", r"\1\1", s)
    # keep letters, numbers, whitespace (strip punctuation)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    # normalize whitespace
    s = re.sub(r"\s+", " ", s)
    return s.strip()

class IntentPredictor:
    def __init__(
        self,
        models_dir: Path = MODELS,
        threshold: float = 0.45,
        nn_sim_threshold: float = 0.55,    # cosine sim for fallback
        top_k_return: int = 5
    ):
        self.clf = None
        self.le = None
        self.sbert = None
        self.use_semantic_model = False
        self.threshold = float(threshold)
        self.nn_sim_threshold = float(nn_sim_threshold)
        self.top_k_return = int(top_k_return)

        meta_path = models_dir / "model_meta.json"
        classifier_path = models_dir / "intent_classifier_keras"
        encoder_path = models_dir / "label_encoder.joblib"

        should_try_full_model = os.environ.get("SCB_ENABLE_FULL_MODEL", "").lower() in {
            "1",
            "true",
            "yes",
        }
        if should_try_full_model and meta_path.exists() and classifier_path.exists() and encoder_path.exists():
            try:
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                from sentence_transformers import SentenceTransformer
                import joblib
                import tensorflow as tf

                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                model_name = meta.get(
                    "sbert_model_name",
                    "sentence-transformers/distiluse-base-multilingual-cased-v2",
                )
                self.sbert = SentenceTransformer(model_name)
                self.clf = tf.keras.models.load_model(classifier_path)
                self.le = joblib.load(encoder_path)
                self.classes = list(self.le.classes_)
                self.use_semantic_model = True
            except Exception as exc:
                print(f"Full intent model unavailable, using local pattern matcher: {exc}")
                self.classes = []
        else:
            self.classes = []

        # load intents json (for responses + pattern bank)
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"intents.json not found at {DATA_PATH}")
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            intents_json = json.load(f)

        self.intent_metadata = {
            i["tag"]: {
                "context_set": i.get("context_set"),
                "priority": i.get("priority", 0),
                "follow_up_intents": i.get("follow_up_intents", []),
                "metadata": i.get("metadata", {}),
            }
            for i in intents_json.get("intents", [])
        }

        # map tag -> responses
        self.tag_to_responses = {i["tag"]: i.get("responses", []) for i in intents_json.get("intents", [])}

        # build pattern bank for NN fallback
        patterns: List[str] = []
        pattern_tags: List[str] = []
        for it in intents_json.get("intents", []):
            tag = it.get("tag")
            for p in it.get("patterns", []):
                p_norm = _normalize_text(p or "")
                if p_norm:
                    patterns.append(p_norm)
                    pattern_tags.append(tag)

        self.pattern_texts = patterns
        self.pattern_tags = np.array(pattern_tags)

        if patterns and self.use_semantic_model:
            # precompute embeddings (unit vectors) for pattern bank
            self.pattern_embs = self.sbert.encode(
                patterns, normalize_embeddings=True, convert_to_numpy=True, batch_size=128, show_progress_bar=False
            )  # shape: (N_patterns, dim)
        else:
            self.pattern_embs = np.array([])

        if not self.classes:
            self.classes = sorted(set(pattern_tags))

    def _embed(self, texts: List[str]) -> np.ndarray:
        return self.sbert.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

    def _lexical_scores(self, text: str) -> dict:
        text_norm = _normalize_text(text)
        scores = {tag: 0.0 for tag in self.classes}
        if not text_norm:
            return scores
        text_words = set(text_norm.split())
        # Common stop-words that should not contribute to specificity scoring
        STOP_WORDS = {"is", "the", "of", "a", "an", "are", "do", "does", "was",
                      "what", "who", "when", "where", "how", "can", "i", "me",
                      "my", "we", "you", "it", "in", "for", "to", "and", "or",
                      "there", "please", "tell", "give", "let", "name", "any"}
        # Content words carry real meaning
        content_words = text_words - STOP_WORDS

        for tag, pattern in zip(self.pattern_tags, self.pattern_texts):
            pattern_words = set(pattern.split())
            pattern_content = pattern_words - STOP_WORDS

            # --- Exact / near-exact string match ---
            ratio = SequenceMatcher(None, text_norm, pattern).ratio()
            if ratio >= 0.95:
                score = 0.99
                scores[str(tag)] = max(scores.get(str(tag), 0.0), score)
                continue

            # --- Phrase containment (substring) ---
            phrase_match = 0.0
            if len(text_words) >= 2 and text_norm in pattern:
                phrase_match = 0.98
            elif len(pattern_words) >= 2 and pattern in text_norm:
                phrase_match = 0.98
            elif len(text_words) == 1 and (text_words & pattern_words):
                phrase_match = 0.95

            # --- Word-level fuzzy overlap (all words) ---
            fuzzy_overlap = sum(
                1
                for word in text_words
                if any(
                    word == pw
                    or (
                        min(len(word), len(pw)) >= 4
                        and SequenceMatcher(None, word, pw).ratio() >= 0.82
                    )
                    for pw in pattern_words
                )
            )
            overlap = fuzzy_overlap / max(len(text_words | pattern_words), 1)
            coverage = fuzzy_overlap / max(len(text_words), 1)
            pattern_coverage = fuzzy_overlap / max(len(pattern_words), 1)

            # --- Content-word overlap (ignores stop-words, rewards specificity) ---
            content_fuzzy = sum(
                1
                for word in content_words
                if any(
                    word == pw
                    or (
                        min(len(word), len(pw)) >= 4
                        and SequenceMatcher(None, word, pw).ratio() >= 0.82
                    )
                    for pw in pattern_content
                )
            ) if content_words and pattern_content else 0
            content_coverage = content_fuzzy / max(len(content_words), 1)
            content_pattern_coverage = content_fuzzy / max(len(pattern_content), 1)
            # High mutual content coverage = very strong signal
            mutual_content = (content_coverage + content_pattern_coverage) / 2

            score = max(
                phrase_match,
                overlap,
                (coverage * 0.55) + (pattern_coverage * 0.45),
                ratio * 0.72,
                mutual_content * 0.90,  # content-word match is rewarded strongly
            )
            scores[str(tag)] = max(scores.get(str(tag), 0.0), score)
        return scores

    def predict_intent(self, text: str):
        """Primary classifier prediction with probabilities map."""
        if not self.clf:
            scores = self._lexical_scores(text)
            if not scores:
                return {"tag": "fallback", "confidence": 0.0, "probs_by_tag": {}}
            best_tag = max(scores, key=scores.get)
            confidence = float(scores[best_tag])
            total = sum(scores.values()) or 1.0
            return {
                "tag": best_tag,
                "confidence": confidence,
                "probs_by_tag": {tag: float(score / total) for tag, score in scores.items()},
            }
            
        text_norm = _normalize_text(text)
        emb = self._embed([text_norm])
        probs = self.clf.predict(emb, verbose=0)[0]
        idx = int(np.argmax(probs))
        return {
            "tag": self.le.classes_[idx],
            "confidence": float(probs[idx]),
            "probs_by_tag": {tag: float(p) for tag, p in zip(self.le.classes_, probs)}
        }

    def top_k(self, text: str, k: int = None) -> List[Tuple[str, float]]:
        """Top-k intents by classifier probability."""
        if not self.clf:
            if k is None:
                k = self.top_k_return
            scores = self._lexical_scores(text)
            return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]
        if k is None: k = self.top_k_return
        text_norm = _normalize_text(text)
        emb = self._embed([text_norm])
        probs = self.clf.predict(emb, verbose=0)[0]
        order = np.argsort(-probs)[:k]
        return [(self.le.classes_[i], float(probs[i])) for i in order]

    def _nn_fallback(self, text: str) -> Tuple[str, float]:
        """Nearest-neighbor over pattern embeddings (cosine sim via dot product)."""
        if self.pattern_embs.size == 0:
            scores = self._lexical_scores(text)
            if not scores:
                return "fallback", 0.0
            best_tag = max(scores, key=scores.get)
            return best_tag, float(scores[best_tag])
        q = _normalize_text(text)
        q_emb = self._embed([q])[0]  # unit vector
        sims = self.pattern_embs @ q_emb  # (N_patterns,)
        best_idx = int(np.argmax(sims))
        return self.pattern_tags[best_idx], float(sims[best_idx])

    def answer(self, user_text: str) -> str:
        """Predict + hook response, with semantic NN fallback when classifier is uncertain."""
        pred = self.predict_intent(user_text)
        tag, conf = pred["tag"], pred["confidence"]

        # 1) If model confident enough, use it
        if conf >= self.threshold:
            responses = self.tag_to_responses.get(tag, [])
            return random.choice(responses) if responses else f"(No response template for: {tag})"

        # 2) Otherwise: semantic NN fallback over patterns
        nn_tag, sim = self._nn_fallback(user_text)
        if sim >= self.nn_sim_threshold:
            responses = self.tag_to_responses.get(nn_tag, [])
            if responses:
                return random.choice(responses)

        # 3) Final fallback
        return "Sorry, I'm not sure about that. Could you rephrase?"

    def get_prediction_with_uncertainty(self, text: str) -> dict:
        """Return intent prediction with entropy-based uncertainty scoring."""
        pred = self.predict_intent(text)
        if not pred.get("probs_by_tag"):
            pred["uncertainty"] = 1.0
            pred["is_uncertain"] = True
            return pred

        if not self.clf:
            uncertainty = max(0.0, min(1.0, 1.0 - pred["confidence"]))
            pred["uncertainty"] = uncertainty
            pred["is_uncertain"] = bool(uncertainty > 0.7 or pred["confidence"] < self.threshold)
            return pred
            
        probs = np.array(list(pred["probs_by_tag"].values()), dtype=np.float64)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(max(len(probs), 2))
        uncertainty = float(entropy / max_entropy) if max_entropy else 0.0
        pred["uncertainty"] = uncertainty
        pred["is_uncertain"] = bool(uncertainty > 0.7 or pred["confidence"] < self.threshold)
        return pred

    def answer_with_context(self, user_text: str, conversation_history: List[str]) -> str:
        """Use the latest turns as light context for follow-up questions."""
        recent_context = " ".join(conversation_history[-3:]).strip()
        combined_input = f"{recent_context} {user_text}".strip() if recent_context else user_text
        return self.answer(combined_input)

    def answer_with_followup_suggestions(self, user_text: str, conversation_history: List[str] = None) -> dict:
        """Return a response payload with context, uncertainty, and suggested next intents."""
        conversation_history = conversation_history or []
        # Bias current prediction using previous turns
        recent_context = " ".join(conversation_history[-3:]).strip()
        combined_input = f"{recent_context} {user_text}".strip() if recent_context else user_text
        
        pred = self.get_prediction_with_uncertainty(combined_input)
        reply = self.answer(combined_input)
        metadata = self.intent_metadata.get(pred["tag"], {})

        return {
            "reply": reply,
            "tag": pred["tag"],
            "confidence": pred["confidence"],
            "uncertainty": pred["uncertainty"],
            "is_uncertain": pred["is_uncertain"],
            "suggested_followups": metadata.get("follow_up_intents", []),
            "top3": self.top_k(combined_input, 3),
            "context_used": bool(recent_context)
        }
