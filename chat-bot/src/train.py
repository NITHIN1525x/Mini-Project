# src/train.py
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from sentence_transformers import SentenceTransformer
import joblib

# ---------------------------
# Paths & Config
# ---------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "intents.json"
OUT_DIR = ROOT / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SBERT_MODEL_NAME = "all-MiniLM-L6-v2"   # 384-dim, fast & accurate for intent tasks
VAL_SIZE = 0.10
TEST_SIZE = 0.10
RANDOM_SEED = 42
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-3
DROPOUT = 0.4
L2_REG = 0.001
MIN_PER_CLASS_FOR_STRATIFY = 2  # basic guard
AUGMENT_FACTOR = 3  # Generate 3x more data via augmentation

# ---------------------------
# Utils
# ---------------------------
def set_seeds(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def load_intents(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"intents.json not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def augment_text(text: str) -> List[str]:
    """Generate augmented variations of text for data augmentation."""
    augmented = [text]
    
    # Variation 1: Add question marks if not present
    if not text.endswith('?'):
        augmented.append(text + '?')
    
    # Variation 2: Remove punctuation
    cleaned = re.sub(r'[?!.,;:]', '', text).strip()
    if cleaned != text:
        augmented.append(cleaned)
    
    # Variation 3: Lowercase variation
    if text != text.lower():
        augmented.append(text.lower())
    
    # Variation 4: Add common prefixes
    prefixes = ['Can you tell me ', 'I want to know ', 'Please tell me ']
    for prefix in prefixes:
        if not text.lower().startswith(prefix.lower()):
            augmented.append(prefix + text.lower())
            if len(augmented) >= AUGMENT_FACTOR + 1:
                break
    
    return augmented[:AUGMENT_FACTOR + 1]

def expand_dataset(intents: Dict, augment: bool = True) -> Tuple[List[str], List[str]]:
    texts, labels = [], []
    for intent in intents.get("intents", []):
        tag = (intent.get("tag") or "").strip()
        for p in intent.get("patterns", []):
            t = (p or "").strip()
            if t and tag:
                if augment:
                    # Add original and augmented versions
                    augmented = augment_text(t)
                    for aug_text in augmented:
                        texts.append(aug_text)
                        labels.append(tag)
                else:
                    texts.append(t)
                    labels.append(tag)
    return texts, labels

def build_classifier(input_dim: int, num_classes: int) -> tf.keras.Model:
    """Build a regularized classifier to prevent overfitting."""
    regularizer = tf.keras.regularizers.l2(L2_REG)
    
    inputs = tf.keras.Input(shape=(input_dim,), name="sbert_embedding")
    x = tf.keras.layers.BatchNormalization()(inputs)
    x = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=regularizer)(x)
    x = tf.keras.layers.Dropout(DROPOUT)(x)
    x = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=regularizer)(x)
    x = tf.keras.layers.Dropout(DROPOUT)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", kernel_regularizer=regularizer)(x)

    model = tf.keras.Model(inputs, outputs, name="intent_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def can_stratify(y: np.ndarray, test_size: float) -> bool:
    """Return True if every class can contribute >=1 sample to BOTH splits."""
    if y.size == 0:
        return False
    counts = Counter(y)
    for cnt in counts.values():
        test_cnt = int(np.floor(cnt * test_size))
        train_cnt = cnt - test_cnt
        # need at least 1 on each side
        if test_cnt < 1 or train_cnt < 1:
            return False
    return True

def safe_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    random_state: int,
    try_stratify: bool = True
):
    """Stratify if feasible, else fall back to random split."""
    if try_stratify and can_stratify(y, test_size):
        return train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
    # fallback: non-stratified
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=None
    )

# ---------------------------
# Main
# ---------------------------
def main():
    set_seeds()

    # 1) Load & expand dataset
    data = load_intents(DATA_PATH)
    texts, labels = expand_dataset(data)
    if not texts:
        raise ValueError("No patterns found in intents.json. Please add patterns to your intents.")
    print(f"Loaded {len(texts)} patterns across {len(set(labels))} intents.")

    # 2) Label encode
    le = LabelEncoder()
    y = le.fit_transform(labels)
    classes = list(le.classes_)
    print("Classes:", classes)

    # 3) SBERT embeddings (frozen encoder)
    print(f"Loading SBERT model: {SBERT_MODEL_NAME}")
    sbert = SentenceTransformer(SBERT_MODEL_NAME)
    X = sbert.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    print("Embedding shape:", X.shape)  # (N, 384)

    # 4) Safe split with rare classes handled (kept in TRAIN)
    counts = Counter(y)
    rare_labels = {lbl for lbl, cnt in counts.items() if cnt < MIN_PER_CLASS_FOR_STRATIFY}
    idx_rare = np.array([i for i, lbl in enumerate(y) if lbl in rare_labels])
    idx_reg  = np.array([i for i, lbl in enumerate(y) if lbl not in rare_labels])

    rare_names = [classes[l] for l in sorted(list(rare_labels))] if rare_labels else []
    if rare_names:
        print(f"[INFO] Intents with <{MIN_PER_CLASS_FOR_STRATIFY} samples (kept only in TRAIN): {rare_names}")

    if len(idx_reg) > 0:
        X_reg, y_reg = X[idx_reg], y[idx_reg]
        # first split: train vs tmp (val+test)
        X_train, X_tmp, y_train, y_tmp = safe_split(
            X_reg, y_reg, test_size=VAL_SIZE + TEST_SIZE, random_state=RANDOM_SEED, try_stratify=True
        )
        # second split: tmp -> val vs test
        rel_test = TEST_SIZE / (VAL_SIZE + TEST_SIZE) if (VAL_SIZE + TEST_SIZE) > 0 else 0.0
        if rel_test > 0 and y_tmp.size > 0:
            X_val, X_test, y_val, y_test = safe_split(
                X_tmp, y_tmp, test_size=rel_test, random_state=RANDOM_SEED, try_stratify=True
            )
        else:
            X_val = np.empty((0, X.shape[1])); y_val = np.array([], dtype=int)
            X_test = np.empty((0, X.shape[1])); y_test = np.array([], dtype=int)
    else:
        # No regular classes; everything is rare (edge case) -> only train
        X_train = np.empty((0, X.shape[1])); y_train = np.array([], dtype=int)
        X_val = np.empty((0, X.shape[1])); y_val = np.array([], dtype=int)
        X_test = np.empty((0, X.shape[1])); y_test = np.array([], dtype=int)

    # Append all rare samples to TRAIN
    if len(idx_rare) > 0:
        X_train = np.vstack([X_train, X[idx_rare]]) if X_train.size else X[idx_rare]
        y_train = np.concatenate([y_train, y[idx_rare]]) if y_train.size else y[idx_rare]

    print(f"[SPLIT] Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")

    # 5) Build classifier
    num_classes = len(classes)
    if len(y_train) == 0:
        raise ValueError("Training split is empty. Please add more patterns to your intents.")
    model = build_classifier(input_dim=X.shape[1], num_classes=num_classes)
    model.summary()

    # 6) Class weights for imbalance (computed on TRAIN only)
    unique_train = np.unique(y_train)
    cw = compute_class_weight(class_weight="balanced", classes=unique_train, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(unique_train, cw)}
    print("[CLASS_WEIGHTS]", class_weight)

    # 7) Train with improved callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor="val_accuracy"),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, monitor="val_loss", min_lr=1e-6)
    ]
    has_val = len(y_val) > 0
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val) if has_val else None,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
        callbacks=callbacks,
        class_weight=class_weight
    )

    # 8) Evaluate (only if we have a test set)
    if len(y_test) > 0:
        test_probs = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
        y_pred = np.argmax(test_probs, axis=1)
        # sklearn's classification_report requires target_names length to match the
        # number of labels reported. The test split may not contain all classes
        # from `classes` (we have many intents), so compute the set of labels
        # actually present in y_test or y_pred and pass those explicitly.
        present_labels = np.unique(np.concatenate([y_test, y_pred]))
        present_target_names = [classes[i] for i in present_labels]
        print("\nTest report:\n",
              classification_report(y_test, y_pred, labels=present_labels, target_names=present_target_names, digits=4))
    else:
        print("\n[INFO] No test split (likely due to many rare classes). Skipping evaluation.")

    # 9) Save artifacts
    model.save(OUT_DIR / "intent_classifier_keras")
    joblib.dump(le, OUT_DIR / "label_encoder.joblib")
    meta = {
        "sbert_model_name": SBERT_MODEL_NAME,
        "classes": classes,
        "embedding_dim": int(X.shape[1]),
    }
    with open(OUT_DIR / "model_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nSaved model & artifacts to: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
