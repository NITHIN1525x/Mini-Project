# src/utils.py
import json
import re
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
import joblib
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
DATA_PATH = ROOT / "data" / "intents.json"

def _normalize_text(s: str) -> str:
    """Lowercase, trim, collapse repeated letters (heyyy -> heyy), strip extra spaces."""
    s = s.strip().lower()
    # collapse 3+ repeated letters to 2 (cooool -> cool)
    s = re.sub(r"(.)\1{2,}", r"\1\1", s)
    # normalize whitespace
    s = re.sub(r"\s+", " ", s)
    return s

class IntentPredictor:
    def __init__(
        self,
        models_dir: Path = MODELS,
        threshold: float = 0.45,
        nn_sim_threshold: float = 0.55,    # cosine sim for fallback
        top_k_return: int = 5
    ):
        meta_path = models_dir / "model_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Model meta not found at {meta_path}. Train the model first.")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        # load encoder + classifier + label encoder
        self.sbert = SentenceTransformer(meta["sbert_model_name"])
        self.clf = tf.keras.models.load_model(models_dir / "intent_classifier_keras")
        self.le  = joblib.load(models_dir / "label_encoder.joblib")
        self.classes = list(self.le.classes_)

        # load intents json (for responses + pattern bank)
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"intents.json not found at {DATA_PATH}")
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            intents_json = json.load(f)

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

        if not patterns:
            raise ValueError("No patterns found in intents.json to build semantic fallback.")

        # precompute embeddings (unit vectors) for pattern bank
        self.pattern_texts = patterns
        self.pattern_tags = np.array(pattern_tags)
        self.pattern_embs = self.sbert.encode(
            patterns, normalize_embeddings=True, convert_to_numpy=True, batch_size=128, show_progress_bar=False
        )  # shape: (N_patterns, dim)

        self.threshold = float(threshold)
        self.nn_sim_threshold = float(nn_sim_threshold)
        self.top_k_return = int(top_k_return)

    def _embed(self, texts: List[str]) -> np.ndarray:
        return self.sbert.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

    def predict_intent(self, text: str):
        """Primary classifier prediction with probabilities map (compatible with your old code)."""
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
        if k is None:
            k = self.top_k_return
        text_norm = _normalize_text(text)
        emb = self._embed([text_norm])
        probs = self.clf.predict(emb, verbose=0)[0]
        order = np.argsort(-probs)[:k]
        return [(self.le.classes_[i], float(probs[i])) for i in order]

    def _nn_fallback(self, text: str) -> Tuple[str, float]:
        """Nearest-neighbor over pattern embeddings (cosine sim via dot product)."""
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
        return "Sorry, I’m not sure about that. Could you rephrase?"
