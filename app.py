"""
Cognitive Emotion Intelligence & Adaptive Lifestyle System
Single-file Streamlit application for major project demonstration.

Core capabilities:
- Multi-modal emotion inference (face image + emoji + text + optional voice text input)
- CNN with Global Average Pooling (GAP) training flow
- Grad-CAM explainability visualization
- Free API option via Hugging Face Inference API
- Optional Spotify integration
- Digital Emotional Twin logging
- Resource catalog generation and CSV export
- Ethical AI monitoring dashboard
- Lightweight RL-style adaptive recommendation update

Usage:
    streamlit run app.py
or:
    python app.py --export-catalog
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import math
import os
import random
import textwrap
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    import streamlit as st
except Exception:
    st = None

try:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
except Exception:
    spotipy = None
    SpotifyClientCredentials = None

try:
    import tensorflow as tf
    from tensorflow.keras import Model
    from tensorflow.keras.applications import EfficientNetB0
    try:
        from tensorflow.keras.applications import EfficientNetV2B0
    except Exception:
        EfficientNetV2B0 = None
    from tensorflow.keras.layers import (
        Conv2D,
        Dense,
        Dropout,
        GlobalAveragePooling2D,
        Input,
        MaxPooling2D,
    )
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
except Exception:
    tf = None
    Model = None
    EfficientNetB0 = None
    EfficientNetV2B0 = None
    Conv2D = None
    Dense = None
    Dropout = None
    GlobalAveragePooling2D = None
    Input = None
    MaxPooling2D = None
    ImageDataGenerator = None

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None


# ---------------------------------------------------------------------------
# Constants and paths
# ---------------------------------------------------------------------------

APP_TITLE = "Cognitive Emotion Intelligence & Adaptive Lifestyle System"
APP_SUBTITLE = (
    "Single-source major project app with CNN+GAP+Grad-CAM, "
    "Digital Emotional Twin, ethical monitoring, and free API option."
)

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "cnn_gap_emotion.keras"
MODEL_META_PATH = MODELS_DIR / "cnn_gap_emotion_metadata.json"
MANIFEST_PATH = DATASET_DIR / "dataset_manifest.json"
RESOURCE_CSV = ROOT / "resource_catalog.csv"
TWIN_LOG_CSV = ROOT / "cei_twin_log.csv"
REC_STATS_CSV = ROOT / "recommender_stats.csv"

RANDOM_SEED = 42
IMAGE_SIZE = (224, 224)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def ensure_dirs() -> None:
    for d in [DATASET_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR, MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def softmax(values: Dict[str, float]) -> Dict[str, float]:
    keys = list(values.keys())
    vals = np.array(list(values.values()), dtype=float)
    vals = vals - np.max(vals)
    exp = np.exp(vals)
    probs = exp / np.sum(exp)
    return {k: float(v) for k, v in zip(keys, probs)}


def normalize_label(label: str) -> str:
    label = str(label).strip().lower()
    mapping = {
        "happy": "joy",
        "happiness": "joy",
        "angry": "anger",
        "sad": "sadness",
        "neutral": "neutral",
        "fearful": "fear",
        "surprised": "surprise",
        "disgusted": "disgust",
    }
    return mapping.get(label, label)


def write_csv_row(path: Path, row: Dict[str, object]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Resource catalog
# ---------------------------------------------------------------------------

@dataclass
class ResourceItem:
    mood: str
    title: str
    url: str
    source: str
    resource_type: str
    offline_fallback: str


def build_resource_catalog() -> List[ResourceItem]:
    moods = [
        "joy",
        "sadness",
        "anger",
        "fear",
        "stress",
        "neutral",
        "focus",
        "relax",
        "motivation",
        "gratitude",
    ]
    base_entries: List[ResourceItem] = []
    for mood in moods:
        base_entries.extend(
            [
                ResourceItem(
                    mood=mood,
                    title=f"{mood.title()} Breathing Music Mix",
                    url=f"https://www.youtube.com/results?search_query={mood}+breathing+music",
                    source="YouTube",
                    resource_type="search",
                    offline_fallback="Use saved breathing playlist and 4-7-8 breathing timer.",
                ),
                ResourceItem(
                    mood=mood,
                    title=f"{mood.title()} Focus Playlist",
                    url=f"https://open.spotify.com/search/{mood}%20focus%20playlist",
                    source="Spotify",
                    resource_type="search",
                    offline_fallback="Play local instrumental tracks sorted by low tempo variance.",
                ),
                ResourceItem(
                    mood=mood,
                    title=f"{mood.title()} Reflection Prompt",
                    url=f"https://www.youtube.com/results?search_query={mood}+journaling+prompts",
                    source="YouTube",
                    resource_type="search",
                    offline_fallback="Write 5 lines: trigger, thought, feeling, action, reframe.",
                ),
                ResourceItem(
                    mood=mood,
                    title=f"{mood.title()} YouTube Music Set",
                    url=f"https://music.youtube.com/search?q={mood}+healing+playlist",
                    source="YouTube Music",
                    resource_type="search",
                    offline_fallback="Use offline playlist based on mood-energy matrix.",
                ),
            ]
        )

    extra = []
    for idx in range(1, 71):
        mood = moods[idx % len(moods)]
        extra.append(
            ResourceItem(
                mood=mood,
                title=f"{mood.title()} Adaptive Resource {idx}",
                url=f"https://www.youtube.com/results?search_query={mood}+adaptive+wellness+{idx}",
                source="YouTube",
                resource_type="search",
                offline_fallback="Use CBT card + 10-minute walking reset.",
            )
        )

    catalog = base_entries + extra
    return catalog


def export_catalog_csv(path: Path = RESOURCE_CSV) -> pd.DataFrame:
    items = build_resource_catalog()
    df = pd.DataFrame(
        [
            {
                "mood": i.mood,
                "title": i.title,
                "url": i.url,
                "source": i.source,
                "type": i.resource_type,
                "offline_fallback": i.offline_fallback,
            }
            for i in items
        ]
    )
    df.to_csv(path, index=False)
    return df


# ---------------------------------------------------------------------------
# Free API integrations (Hugging Face + optional OpenAI/Spotify)
# ---------------------------------------------------------------------------

def hf_headers() -> Dict[str, str]:
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN", "").strip() or os.getenv("HF_TOKEN", "").strip()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def validate_hf_token() -> Tuple[bool, str]:
    """
    Validates token by calling HF whoami endpoint.
    """
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN", "").strip() or os.getenv("HF_TOKEN", "").strip()
    if not token:
        return False, "HF token not found. Set HF_TOKEN or HUGGINGFACEHUB_API_TOKEN."
    try:
        resp = requests.get(
            "https://huggingface.co/api/whoami-v2",
            headers={"Authorization": f"Bearer {token}"},
            timeout=20,
        )
        if resp.status_code >= 400:
            return False, f"HF token check failed ({resp.status_code})."
        data = resp.json()
        user = data.get("name") or data.get("email") or "authenticated_user"
        return True, f"HF token is valid for user: {user}"
    except Exception:
        return False, "HF token validation request failed."


def call_hf_chat(prompt: str, model_id: str = "HuggingFaceH4/zephyr-7b-beta", timeout: int = 45) -> str:
    """
    Free-tier capable HF Inference API call (depends on user token quota).
    Falls back safely if unavailable.
    """
    api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.7,
            "return_full_text": False,
        },
    }
    try:
        resp = requests.post(api_url, headers=hf_headers(), json=payload, timeout=timeout)
        if resp.status_code >= 400:
            return (
                f"HF API unavailable ({resp.status_code}). "
                "Using safe local fallback guidance."
            )
        data = resp.json()
        if isinstance(data, list) and data and "generated_text" in data[0]:
            return data[0]["generated_text"].strip()
        if isinstance(data, dict) and "generated_text" in data:
            return str(data["generated_text"]).strip()
        return "Model responded in unexpected format. Using local fallback."
    except Exception:
        return "HF API timeout/error. Using local fallback."


def call_openai_chat(
    prompt: str,
    model: str = "gpt-4o-mini",
    timeout: int = 45,
) -> str:
    """
    Optional paid API path if OPENAI_API_KEY is configured.
    Gracefully falls back when missing/unavailable.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return "OpenAI key missing. Using free/local fallback."
    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a supportive lifestyle coach."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 220,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if resp.status_code >= 400:
            return f"OpenAI API unavailable ({resp.status_code}). Using free/local fallback."
        data = resp.json()
        return (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        ) or "OpenAI empty response. Using free/local fallback."
    except Exception:
        return "OpenAI API timeout/error. Using free/local fallback."


def parse_questions_output(raw_text: str) -> List[str]:
    raw_lines = [x.strip(" -0123456789.)") for x in raw_text.splitlines() if x.strip()]
    lines = [x for x in raw_lines if len(x) > 8][:5]
    return lines


def ask_mood_questions_with_provider(
    mood: str,
    context: str,
    provider: str = "auto",
) -> List[str]:
    prompt = textwrap.dedent(
        f"""
        You are a supportive lifestyle coach.
        User mood: {mood}
        User context: {context}
        Generate 5 concise reflective questions that are practical, safe, and non-clinical.
        """
    ).strip()

    output = ""
    p = provider.lower().strip()
    if p == "openai":
        output = call_openai_chat(prompt=prompt)
        lines = parse_questions_output(output)
        if lines:
            return lines
        output = call_hf_chat(prompt=prompt)
    elif p == "huggingface":
        output = call_hf_chat(prompt=prompt)
    elif p == "local":
        output = ""
    else:
        # auto: free-first strategy
        output = call_hf_chat(prompt=prompt)
        lines = parse_questions_output(output)
        if lines:
            return lines
        output = call_openai_chat(prompt=prompt)

    if "fallback" not in output.lower() and len(output) > 20:
        lines = parse_questions_output(output)
        if lines:
            return lines
    return [
        "What event in the last 24 hours influenced your mood the most?",
        "Which one action today can improve your emotional balance by 10%?",
        "What kind of music helps you feel calmer or more focused right now?",
        "What thought pattern would you like to replace with a healthier one?",
        "Who is one person you can connect with for positive support today?",
    ]


def get_spotify_recommendations_by_query(
    mood: str, limit: int = 10
) -> List[Dict[str, str]]:
    """
    Optional paid API path if user configures Spotify credentials.
    Has free public-search fallback links if keys missing.
    """
    cid = os.getenv("SPOTIPY_CLIENT_ID", "").strip()
    csecret = os.getenv("SPOTIPY_CLIENT_SECRET", "").strip()
    if not (cid and csecret and spotipy and SpotifyClientCredentials):
        return [
            {
                "track": f"{mood.title()} playlist search",
                "artist": "Open Search",
                "url": f"https://open.spotify.com/search/{mood}%20playlist",
            }
        ]

    try:
        auth = SpotifyClientCredentials(client_id=cid, client_secret=csecret)
        sp = spotipy.Spotify(auth_manager=auth)
        res = sp.search(q=f"{mood} mood playlist", type="track", limit=limit)
        items = []
        for t in res.get("tracks", {}).get("items", []):
            items.append(
                {
                    "track": t.get("name", "Unknown Track"),
                    "artist": ", ".join([a["name"] for a in t.get("artists", [])]) or "Unknown Artist",
                    "url": t.get("external_urls", {}).get("spotify", ""),
                }
            )
        return items or [
            {
                "track": f"{mood.title()} playlist search",
                "artist": "Open Search",
                "url": f"https://open.spotify.com/search/{mood}%20playlist",
            }
        ]
    except Exception:
        return [
            {
                "track": f"{mood.title()} playlist search",
                "artist": "Open Search",
                "url": f"https://open.spotify.com/search/{mood}%20playlist",
            }
        ]


# ---------------------------------------------------------------------------
# Emotion inference heuristics + fusion
# ---------------------------------------------------------------------------

EMOJI_MOOD_MAP = {
    "😀": "joy",
    "😄": "joy",
    "😊": "joy",
    "🙂": "neutral",
    "😌": "relax",
    "😢": "sadness",
    "😭": "sadness",
    "😡": "anger",
    "😠": "anger",
    "😨": "fear",
    "😰": "stress",
    "😐": "neutral",
    "🤩": "motivation",
    "😴": "relax",
}

TEXT_KEYWORDS = {
    "joy": ["happy", "joy", "great", "excited", "awesome", "good"],
    "sadness": ["sad", "down", "lonely", "cry", "upset", "hurt"],
    "anger": ["angry", "mad", "annoyed", "irritated", "frustrated"],
    "fear": ["afraid", "anxious", "scared", "fear", "panic"],
    "stress": ["stress", "overwhelmed", "pressure", "burnout", "tired"],
    "neutral": ["okay", "fine", "normal", "neutral"],
    "focus": ["focus", "study", "work", "deadline", "productive"],
    "relax": ["calm", "rest", "peace", "relax"],
    "motivation": ["motivate", "energy", "gym", "push", "achieve"],
}


def infer_from_text(text: str) -> Dict[str, float]:
    text = (text or "").lower()
    scores = {k: 0.0 for k in TEXT_KEYWORDS.keys()}
    for mood, kws in TEXT_KEYWORDS.items():
        for kw in kws:
            if kw in text:
                scores[mood] += 1.0
    if sum(scores.values()) == 0:
        scores["neutral"] = 1.0
    return softmax(scores)


def infer_from_emoji(emoji: str) -> Dict[str, float]:
    mood = EMOJI_MOOD_MAP.get(emoji, "neutral")
    base = {k: 0.01 for k in TEXT_KEYWORDS.keys()}
    base[mood] = 1.0
    return softmax(base)


def infer_from_image_basic(img: Image.Image) -> Dict[str, float]:
    """
    Lightweight heuristic fallback when no trained model is loaded.
    """
    img = img.convert("RGB").resize((64, 64))
    arr = np.array(img).astype(np.float32) / 255.0
    brightness = float(arr.mean())
    saturation = float(np.std(arr, axis=2).mean())
    scores = {k: 0.05 for k in TEXT_KEYWORDS.keys()}
    if brightness > 0.62:
        scores["joy"] += 0.8
        scores["motivation"] += 0.4
    elif brightness < 0.35:
        scores["sadness"] += 0.8
        scores["stress"] += 0.3
    else:
        scores["neutral"] += 0.6
        scores["focus"] += 0.2

    if saturation > 0.22:
        scores["motivation"] += 0.4
        scores["anger"] += 0.2
    else:
        scores["relax"] += 0.3
    return softmax(scores)


def fuse_modalities(
    text_scores: Dict[str, float],
    emoji_scores: Dict[str, float],
    image_scores: Dict[str, float],
    weights: Tuple[float, float, float] = (0.45, 0.2, 0.35),
) -> Dict[str, float]:
    wt, we, wi = weights
    keys = set(text_scores) | set(emoji_scores) | set(image_scores)
    fused = {}
    for k in keys:
        fused[k] = (
            wt * text_scores.get(k, 0.0)
            + we * emoji_scores.get(k, 0.0)
            + wi * image_scores.get(k, 0.0)
        )
    return softmax(fused)


# ---------------------------------------------------------------------------
# RL-style adaptive recommender (simple Q-table update)
# ---------------------------------------------------------------------------

def qtable_path() -> Path:
    return ROOT / "recommender_qtable.json"


def load_qtable() -> Dict[str, Dict[str, float]]:
    p = qtable_path()
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_qtable(qt: Dict[str, Dict[str, float]]) -> None:
    qtable_path().write_text(json.dumps(qt, indent=2), encoding="utf-8")


def choose_recommendation_strategy(mood: str) -> str:
    qt = load_qtable()
    actions = ["calming", "energizing", "focus", "reflective"]
    state = qt.get(mood, {a: 0.0 for a in actions})
    # epsilon-greedy
    if random.random() < 0.2:
        return random.choice(actions)
    return max(state, key=state.get)


def update_qtable(mood: str, action: str, reward: float, alpha: float = 0.25) -> None:
    qt = load_qtable()
    actions = ["calming", "energizing", "focus", "reflective"]
    state = qt.get(mood, {a: 0.0 for a in actions})
    old = state.get(action, 0.0)
    state[action] = old + alpha * (reward - old)
    qt[mood] = state
    save_qtable(qt)


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

HF_DATASET_OPTIONS = {
    "FER2025 (sample-ready)": "Piero2411/FER-2013",
    "EmoNet-Face-Big (sample-ready)": "motheecreator/Emotion_Detection_Images",
    "MER2024 (manual only)": None,
    "MER2023 (manual only)": None,
}


def detect_columns(example: dict) -> Tuple[Optional[str], Optional[str]]:
    image_col = None
    label_col = None
    for k, v in example.items():
        if image_col is None and ("image" in k.lower() or hasattr(v, "convert")):
            image_col = k
        if label_col is None and ("label" in k.lower() or "emotion" in k.lower()):
            label_col = k
    return image_col, label_col


def split_target(subset_size: int) -> Tuple[int, int, int]:
    train_n = int(subset_size * 0.7)
    val_n = int(subset_size * 0.15)
    test_n = subset_size - train_n - val_n
    return train_n, val_n, test_n


def save_image_to_class_dir(split_dir: Path, label: str, idx: int, image_obj) -> None:
    cls = normalize_label(label)
    d = split_dir / cls
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"{cls}_{idx:06d}.jpg"
    try:
        if hasattr(image_obj, "convert"):
            img = image_obj.convert("RGB")
        elif isinstance(image_obj, np.ndarray):
            img = Image.fromarray(image_obj).convert("RGB")
        elif isinstance(image_obj, dict) and "bytes" in image_obj:
            img = Image.open(io.BytesIO(image_obj["bytes"])).convert("RGB")
        else:
            return
        img = img.resize(IMAGE_SIZE)
        img.save(out, format="JPEG", quality=92)
    except Exception:
        return


def clear_split_dirs() -> None:
    for split in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        if split.exists():
            for cls in split.iterdir():
                if cls.is_dir():
                    for f in cls.iterdir():
                        try:
                            f.unlink()
                        except Exception:
                            pass


def prepare_sampled_dataset(
    dataset_name: str,
    subset_size: int = 800,
    seed: int = RANDOM_SEED,
) -> Dict[str, object]:
    """
    Downloads and prepares sampled dataset from HF datasets.
    """
    if load_dataset is None:
        raise RuntimeError("datasets library not installed.")
    ensure_dirs()
    clear_split_dirs()
    random.seed(seed)

    ds = load_dataset(dataset_name, split="train")
    if len(ds) == 0:
        raise RuntimeError("Dataset is empty.")

    subset_size = min(subset_size, len(ds))
    indices = list(range(len(ds)))
    random.shuffle(indices)
    indices = indices[:subset_size]

    sample = ds[indices[0]]
    image_col, label_col = detect_columns(sample)
    if image_col is None or label_col is None:
        raise RuntimeError("Could not auto-detect image/label columns.")

    train_n, val_n, test_n = split_target(subset_size)
    split_map = (
        ["train"] * train_n
        + ["val"] * val_n
        + ["test"] * test_n
    )
    random.shuffle(split_map)

    labels_seen = {}
    for i, (idx, split) in enumerate(zip(indices, split_map), start=1):
        row = ds[idx]
        label = row[label_col]
        if isinstance(label, int):
            try:
                label_name = ds.features[label_col].names[label]
            except Exception:
                label_name = str(label)
        else:
            label_name = str(label)
        img_obj = row[image_col]
        split_dir = {"train": TRAIN_DIR, "val": VAL_DIR, "test": TEST_DIR}[split]
        save_image_to_class_dir(split_dir, label_name, i, img_obj)
        labels_seen[normalize_label(label_name)] = labels_seen.get(normalize_label(label_name), 0) + 1

    manifest = {
        "created_at": now_iso(),
        "dataset_name": dataset_name,
        "subset_size": subset_size,
        "splits": {"train": train_n, "val": val_n, "test": test_n},
        "columns": {"image_col": image_col, "label_col": label_col},
        "labels": labels_seen,
        "image_size": IMAGE_SIZE,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


# ---------------------------------------------------------------------------
# CNN + GAP model, training, evaluation, Grad-CAM
# ---------------------------------------------------------------------------

def list_class_names(train_dir: Path = TRAIN_DIR) -> List[str]:
    if not train_dir.exists():
        return []
    return sorted([d.name for d in train_dir.iterdir() if d.is_dir()])


def build_cnn_gap_model(num_classes: int) -> "tf.keras.Model":
    if tf is None:
        raise RuntimeError("TensorFlow is not available.")

    inputs = Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

    # Prefer a newer EfficientNetV2 backbone; fallback to EfficientNetB0.
    backbone_cls = EfficientNetV2B0 if EfficientNetV2B0 is not None else EfficientNetB0
    base = backbone_cls(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
    )
    base.trainable = False

    x = base.output
    x = Conv2D(128, (3, 3), padding="same", activation="relu", name="conv_refine")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), padding="same", activation="relu", name="conv_last")(x)
    x = GlobalAveragePooling2D(name="gap")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax", name="pred")(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_generators(
    batch_size: int = 8,
) -> Tuple["ImageDataGenerator", "ImageDataGenerator", object, object, object]:
    if ImageDataGenerator is None:
        raise RuntimeError("TensorFlow image generator not available.")
    train_aug = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )
    eval_aug = ImageDataGenerator(rescale=1.0 / 255.0)
    train_gen = train_aug.flow_from_directory(
        TRAIN_DIR,
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
    )
    val_gen = eval_aug.flow_from_directory(
        VAL_DIR,
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )
    test_gen = eval_aug.flow_from_directory(
        TEST_DIR,
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )
    return train_aug, eval_aug, train_gen, val_gen, test_gen


def safe_roc_auc(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    try:
        if y_true.shape[1] > 2:
            return float(roc_auc_score(y_true, y_pred, multi_class="ovr", average="macro"))
        return float(roc_auc_score(y_true[:, 1], y_pred[:, 1]))
    except Exception:
        return None


def train_and_evaluate(
    epochs: int = 2,
    batch_size: int = 8,
) -> Dict[str, object]:
    if tf is None:
        raise RuntimeError("TensorFlow not available. Install tensorflow to train model.")
    ensure_dirs()
    classes = list_class_names(TRAIN_DIR)
    if len(classes) < 2:
        raise RuntimeError("Need at least 2 classes in dataset/train for training.")

    _, _, train_gen, val_gen, test_gen = build_generators(batch_size=batch_size)
    model = build_cnn_gap_model(num_classes=len(classes))
    hist = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        verbose=1,
    )

    y_prob = model.predict(test_gen, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = test_gen.classes

    precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, target_names=list(test_gen.class_indices.keys()), zero_division=0)

    y_true_onehot = tf.keras.utils.to_categorical(y_true, num_classes=len(classes))
    roc = safe_roc_auc(y_true_onehot, y_prob)

    model.save(MODEL_PATH)
    metadata = {
        "saved_at": now_iso(),
        "classes": list(test_gen.class_indices.keys()),
        "class_indices": test_gen.class_indices,
        "image_size": IMAGE_SIZE,
        "epochs": epochs,
        "batch_size": batch_size,
        "metrics": {
            "accuracy": acc,
            "precision_macro": precision,
            "recall_macro": recall,
            "f1_macro": f1,
            "roc_auc": roc,
        },
        "history": {k: [float(vv) for vv in vals] for k, vals in hist.history.items()},
    }
    MODEL_META_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "metrics": metadata["metrics"],
        "confusion_matrix": cm,
        "classification_report": report,
        "class_indices": test_gen.class_indices,
    }


def load_trained_model() -> Optional["tf.keras.Model"]:
    if tf is None:
        return None
    if MODEL_PATH.exists():
        try:
            return tf.keras.models.load_model(MODEL_PATH)
        except Exception:
            return None
    return None


def preprocess_for_model(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB").resize(IMAGE_SIZE)).astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict_emotion_with_model(img: Image.Image) -> Optional[Dict[str, float]]:
    model = load_trained_model()
    if model is None or not MODEL_META_PATH.exists():
        return None
    meta = json.loads(MODEL_META_PATH.read_text(encoding="utf-8"))
    classes = meta.get("classes", [])
    x = preprocess_for_model(img)
    probs = model.predict(x, verbose=0)[0]
    if len(classes) != len(probs):
        return None
    return {str(c): float(p) for c, p in zip(classes, probs)}


def make_gradcam_heatmap(
    img_array: np.ndarray, model: "tf.keras.Model", last_conv_layer_name: str = "conv_last"
) -> np.ndarray:
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def overlay_gradcam_on_image(pil_img: Image.Image, heatmap: np.ndarray, alpha: float = 0.4) -> Image.Image:
    import matplotlib.cm as cm

    img = pil_img.convert("RGB").resize(IMAGE_SIZE)
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = Image.fromarray((jet_heatmap * 255).astype(np.uint8)).resize(IMAGE_SIZE)
    overlay = Image.blend(img, jet_heatmap, alpha=alpha)
    return overlay


# ---------------------------------------------------------------------------
# Ethical AI monitoring + Digital Emotional Twin
# ---------------------------------------------------------------------------

def ethical_risk_assessment(
    confidence: float,
    text_len: int,
    used_image: bool,
    used_text: bool,
    used_emoji: bool,
) -> Dict[str, object]:
    risk_flags = []
    if confidence < 0.45:
        risk_flags.append("low_confidence_prediction")
    if text_len < 8 and not used_image:
        risk_flags.append("insufficient_context")
    if not (used_image and used_text and used_emoji):
        risk_flags.append("partial_modality_use")
    bias_note = (
        "Potential demographic bias if face data is imbalanced. "
        "Use diverse datasets and monitor per-class/per-group drift."
    )
    return {
        "risk_flags": risk_flags,
        "confidence_band": "high" if confidence > 0.75 else "medium" if confidence > 0.45 else "low",
        "action": "show_recommendation_with_disclaimer" if confidence > 0.45 else "ask_followup_questions",
        "bias_note": bias_note,
        "privacy_note": "Do not store raw images/audio unless user consent is explicit.",
    }


def log_emotional_twin(
    user_id: str,
    fused_scores: Dict[str, float],
    context: str,
    recommendation_action: str,
    confidence: float,
    ethical_action: str,
) -> None:
    top_mood = max(fused_scores, key=fused_scores.get)
    row = {
        "timestamp_utc": now_iso(),
        "user_id": user_id,
        "top_mood": top_mood,
        "confidence": round(confidence, 4),
        "fused_scores_json": json.dumps(fused_scores),
        "context": context[:500],
        "recommendation_action": recommendation_action,
        "ethical_action": ethical_action,
    }
    write_csv_row(TWIN_LOG_CSV, row)


def log_recommender_stats(user_id: str, mood: str, strategy: str, shown: int) -> None:
    row = {
        "timestamp_utc": now_iso(),
        "user_id": user_id,
        "mood": mood,
        "strategy": strategy,
        "num_items_shown": shown,
    }
    write_csv_row(REC_STATS_CSV, row)


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def section_header(title: str) -> None:
    st.markdown(f"### {title}")


def run_streamlit_app() -> None:
    ensure_dirs()
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)

    with st.sidebar:
        st.header("Project Controls")
        export_clicked = st.button("Export Resource Catalog CSV")
        if export_clicked:
            df = export_catalog_csv()
            st.success(f"Exported {len(df)} catalog items to {RESOURCE_CSV.name}")

        st.markdown("---")
        st.subheader("API setup")
        question_provider = st.selectbox(
            "Question generation provider",
            options=["auto", "huggingface", "openai", "local"],
            index=0,
            help=(
                "auto = Hugging Face free-first with fallback, "
                "openai = optional paid path, local = deterministic offline prompts."
            ),
        )
        st.write(
            "Set `HF_TOKEN` (or `HUGGINGFACEHUB_API_TOKEN`) in terminal for free-tier Hugging Face inference."
        )
        st.code(
            "set HF_TOKEN=your_token_here   # Windows CMD\n"
            "$env:HF_TOKEN='your_token_here'  # PowerShell",
            language="bash",
        )
        if st.button("Validate Hugging Face token"):
            ok, msg = validate_hf_token()
            if ok:
                st.success(msg)
            else:
                st.warning(msg)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Emotion Fusion", "Dataset Prep", "Training + Eval", "Grad-CAM", "Research & Viva"]
    )

    with tab1:
        section_header("Multi-Modal Emotion Fusion")
        col1, col2 = st.columns(2)
        with col1:
            user_id = st.text_input("User ID", value="demo_user")
            text_context = st.text_area(
                "Text context (voice-to-text fallback supported by manual input)",
                value="I feel a bit stressed because of deadlines but want to stay productive.",
                height=120,
            )
            emoji = st.text_input("Emoji input", value="😰")
            uploaded = st.file_uploader("Upload face image", type=["jpg", "jpeg", "png"])
        with col2:
            st.info(
                "Fusion uses weighted signals: text + emoji + image.\n"
                "If trained model is available, image emotion uses model inference; else heuristic fallback."
            )
            run_btn = st.button("Run Emotion Fusion & Recommendation", type="primary")

        if run_btn:
            if uploaded is not None:
                pil = Image.open(uploaded).convert("RGB")
                st.image(pil, caption="Input face image", width=260)
                model_scores = predict_emotion_with_model(pil)
                image_scores = model_scores if model_scores else infer_from_image_basic(pil)
            else:
                pil = None
                image_scores = softmax({k: 1.0 if k == "neutral" else 0.01 for k in TEXT_KEYWORDS})

            text_scores = infer_from_text(text_context)
            emoji_scores = infer_from_emoji(emoji.strip())
            fused = fuse_modalities(text_scores, emoji_scores, image_scores)
            top_mood = max(fused, key=fused.get)
            confidence = float(fused[top_mood])

            st.success(f"Detected mood: **{top_mood}** (confidence {confidence:.2f})")
            st.write("Fused mood distribution")
            st.json({k: round(v, 4) for k, v in sorted(fused.items(), key=lambda x: x[1], reverse=True)})

            questions = ask_mood_questions_with_provider(
                top_mood,
                text_context,
                provider=question_provider,
            )
            st.markdown("#### Important reflective questions (mindset-aware)")
            for i, q in enumerate(questions, start=1):
                st.write(f"{i}. {q}")

            strategy = choose_recommendation_strategy(top_mood)
            tracks = get_spotify_recommendations_by_query(top_mood, limit=8)
            st.markdown(f"#### Recommendations ({strategy} strategy)")
            for t in tracks[:8]:
                st.markdown(f"- **{t['track']}** — {t['artist']}  \n  {t['url']}")

            ethical = ethical_risk_assessment(
                confidence=confidence,
                text_len=len(text_context or ""),
                used_image=uploaded is not None,
                used_text=bool(text_context.strip()),
                used_emoji=bool(emoji.strip()),
            )
            st.markdown("#### Ethical AI monitor")
            st.json(ethical)

            log_emotional_twin(
                user_id=user_id.strip() or "unknown_user",
                fused_scores=fused,
                context=text_context,
                recommendation_action=strategy,
                confidence=confidence,
                ethical_action=ethical["action"],
            )
            log_recommender_stats(
                user_id=user_id.strip() or "unknown_user",
                mood=top_mood,
                strategy=strategy,
                shown=min(8, len(tracks)),
            )
            st.caption("Digital Emotional Twin and recommender stats logged to CSV.")

            feedback = st.slider("How useful were recommendations? (reward signal)", 1, 5, 3)
            if st.button("Submit feedback to adaptive recommender"):
                reward = (feedback - 3) / 2.0
                update_qtable(top_mood, strategy, reward)
                st.success("Adaptive policy updated (RL-style Q-table).")

    with tab2:
        section_header("Automatic dataset download and split preparation")
        st.write(
            "For 4GB RAM systems, keep sampled range around 600-1200 images, small batch size, and short epochs."
        )
        selected_label = st.selectbox("Select dataset source", list(HF_DATASET_OPTIONS.keys()), index=0)
        subset_size = st.slider("Sample size", min_value=300, max_value=1500, value=800, step=100)

        if st.button("Prepare sampled dataset"):
            dataset_name = HF_DATASET_OPTIONS[selected_label]
            if dataset_name is None:
                st.warning("This dataset is manual-only in this implementation.")
            else:
                with st.spinner("Preparing dataset..."):
                    try:
                        manifest = prepare_sampled_dataset(dataset_name=dataset_name, subset_size=subset_size)
                        st.success("Dataset prepared successfully.")
                        st.json(manifest)
                    except Exception as ex:
                        st.error(f"Dataset preparation failed: {ex}")

        if MANIFEST_PATH.exists():
            st.markdown("#### Current dataset manifest")
            st.code(MANIFEST_PATH.read_text(encoding="utf-8"), language="json")

    with tab3:
        section_header("CNN + GAP Training and Evaluation")
        st.write(
            "Upgraded architecture: EfficientNetV2B0 (fallback EfficientNetB0) "
            "+ Conv refinement + GAP + Dense softmax head."
        )
        epochs = st.slider("Epochs", 1, 5, 2)
        batch_size = st.selectbox("Batch size", [4, 8, 16], index=1)
        if st.button("Train and evaluate model"):
            with st.spinner("Training model..."):
                try:
                    result = train_and_evaluate(epochs=epochs, batch_size=batch_size)
                    st.success("Training completed.")
                    st.markdown("#### Metrics")
                    st.json(result["metrics"])
                    st.markdown("#### Confusion Matrix")
                    st.write(pd.DataFrame(result["confusion_matrix"]))
                    st.markdown("#### Classification Report")
                    st.code(result["classification_report"])
                except Exception as ex:
                    st.error(str(ex))

    with tab4:
        section_header("Grad-CAM Explainability")
        gimg = st.file_uploader("Upload image for Grad-CAM", type=["jpg", "jpeg", "png"], key="gcam_upload")
        if st.button("Generate Grad-CAM"):
            if tf is None:
                st.error("TensorFlow not installed.")
            elif gimg is None:
                st.warning("Please upload an image.")
            else:
                model = load_trained_model()
                if model is None:
                    st.error("No trained model found. Train model first.")
                else:
                    img = Image.open(gimg).convert("RGB")
                    x = preprocess_for_model(img)
                    try:
                        heatmap = make_gradcam_heatmap(x, model, last_conv_layer_name="conv_last")
                        overlay = overlay_gradcam_on_image(img, heatmap)
                        c1, c2 = st.columns(2)
                        c1.image(img, caption="Original", use_column_width=True)
                        c2.image(overlay, caption="Grad-CAM overlay", use_column_width=True)
                    except Exception as ex:
                        st.error(f"Grad-CAM generation failed: {ex}")

    with tab5:
        section_header("Research support, setup, and viva guidance")
        st.markdown(
            """
            #### Why Hugging Face for free API integration?
            - Free-tier token creation is straightforward via user account.
            - Good ecosystem for emotion/NLP models and inference endpoints.
            - Can be replaced with local models when offline or quota-limited.

            #### Windows 11 (i5, 4GB RAM) practical guidance
            - Use sampled datasets (600-1200 images).
            - Batch size 4 or 8.
            - Epochs 1-3 for demonstration.
            - Keep active browser tabs minimal during training.

            #### APK conversion (free trick)
            1. Host Streamlit app locally or on free cloud.
            2. Open URL in Android Chrome.
            3. Use `Add to Home Screen` to create installable PWA-like shortcut.
            4. For native APK packaging, use Kivy/Buildozer separately (outside this app).
            """
        )

        st.markdown("#### Top viva questions (quick set)")
        viva = [
            ("What is the novelty?", "Multi-modal emotion fusion + adaptive recommender + explainable AI + ethical monitoring in one deployable single-file system."),
            ("Why GAP in CNN?", "Global Average Pooling reduces parameters, lowers overfitting risk, and improves interpretability for Grad-CAM."),
            ("Why Grad-CAM?", "It validates whether model attention regions align with meaningful facial cues, improving trust and auditability."),
            ("How does RL logic help?", "Q-table updates recommendation strategy from user feedback, enabling personalized adaptation over sessions."),
            ("How does system address 2023-2025 gaps?", "It combines multimodal emotion signals, explainability, practical deployment, and ethical guardrails in one pipeline."),
        ]
        for q, a in viva:
            st.write(f"- **{q}** {a}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_cli_export_catalog() -> None:
    df = export_catalog_csv()
    print(f"Exported {len(df)} resources to {RESOURCE_CSV}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CEI major project app")
    parser.add_argument("--export-catalog", action="store_true", help="Export resource catalog CSV and exit")
    args, _ = parser.parse_known_args()

    ensure_dirs()
    if args.export_catalog:
        run_cli_export_catalog()
        return

    if st is None:
        print("Streamlit is not installed. Run: pip install -r requirements.txt")
        return
    run_streamlit_app()


if __name__ == "__main__":
    main()
