"""
CEI-ALOS Streamlit prototype (single-file):
- Multi-modal emotion pipeline (face, emoji, voice, text)
- Transfer Learning (MobileNetV2) training/evaluation with metrics
- Grad-CAM explainability
- RL-style no-repetition recommender
- Spotify + OpenAI integration (optional)
- Digital Emotional Twin logging
- Option A: automatic dataset acquisition + pre-split tooling
- Option B: 100+ structured resource entries + CSV export
"""

from __future__ import annotations

import io
import json
import math
import os
import random
import shutil
import subprocess
import tarfile
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import streamlit as st

# Optional / heavy imports
try:
    import cv2
except Exception:
    cv2 = None

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models

    TF_AVAILABLE = True
except Exception:
    tf = None
    layers = None
    models = None
    TF_AVAILABLE = False

try:
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    from sklearn.preprocessing import LabelBinarizer

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

try:
    from PIL import Image, ImageDraw

    PIL_AVAILABLE = True
except Exception:
    Image = None
    ImageDraw = None
    PIL_AVAILABLE = False

try:
    import speech_recognition as sr
except Exception:
    sr = None

try:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
except Exception:
    spotipy = None
    SpotifyClientCredentials = None

try:
    import openai
except Exception:
    openai = None

try:
    from datasets import load_dataset

    HF_DATASETS_AVAILABLE = True
except Exception:
    load_dataset = None
    HF_DATASETS_AVAILABLE = False

# ---------------- Settings ----------------
st.set_page_config(page_title="CEI-ALOS FINAL", layout="wide")

APP_ROOT = Path(".")
DATA_FILE = APP_ROOT / "cei_twin_log.csv"
RL_STATS_FILE = APP_ROOT / "recommender_stats.csv"
MODEL_DIR = APP_ROOT / "models"
DOWNLOAD_DIR = APP_ROOT / "downloads"
RAW_DATASET_DIR = APP_ROOT / "raw_datasets"
PREPARED_DATASET_DIR = APP_ROOT / "dataset"
CATALOG_EXPORT_FILE = APP_ROOT / "resource_catalog.csv"
CLASS_NAMES_FILE = MODEL_DIR / "class_names.json"

for folder in [MODEL_DIR, DOWNLOAD_DIR, RAW_DATASET_DIR, PREPARED_DATASET_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# ----------------- API setup -----------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
SPOTIPY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID", "")
SPOTIPY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET", "")

openai_client = None
if openai and OPENAI_API_KEY:
    try:
        if hasattr(openai, "OpenAI"):  # openai>=1.x
            openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        else:  # legacy API
            openai.api_key = OPENAI_API_KEY
    except Exception:
        openai_client = None

if spotipy and SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET:
    try:
        sp = spotipy.Spotify(
            auth_manager=SpotifyClientCredentials(
                client_id=SPOTIPY_CLIENT_ID,
                client_secret=SPOTIPY_CLIENT_SECRET,
            )
        )
    except Exception:
        sp = None
else:
    sp = None

# ----------------- Structured Resource Catalog (100+) -----------------
EMOJI_MAP = {
    "😊": "happy",
    "😢": "sad",
    "😡": "energetic",
    "😴": "calm",
    "😍": "happy",
    "😌": "calm",
}


def build_resource_catalog() -> dict[str, list[dict[str, str]]]:
    seed_links = {
        "HAPPY": [
            ("Pharrell Williams - Happy", "https://www.youtube.com/watch?v=ZbZSe6N_BXs"),
            ("Uptown Funk", "https://www.youtube.com/watch?v=OPf0YbXqDm0"),
            ("Can’t Stop The Feeling", "https://www.youtube.com/watch?v=ru0K8uYEZWw"),
            ("Best Day Of My Life", "https://www.youtube.com/watch?v=Y66j_BUCBMY"),
            ("On Top Of The World", "https://www.youtube.com/watch?v=w5tWYmIOWGk"),
        ],
        "SAD": [
            ("Guided Breathing 10 min", "https://www.youtube.com/watch?v=inpok4MKVLM"),
            ("Calm Anxiety Release", "https://www.youtube.com/watch?v=1vx8iUvfyCY"),
            ("Emotional Healing Music", "https://www.youtube.com/watch?v=2OEL4P1Rz04"),
            ("Mindful Pause", "https://www.youtube.com/watch?v=MIr3RsUWrdo"),
            ("Gentle Rain Ambience", "https://www.youtube.com/watch?v=mPZkdNFkNps"),
        ],
        "CALM": [
            ("Lofi hip hop radio", "https://www.youtube.com/watch?v=jfKfPfyJRdk"),
            ("Deep Focus Music", "https://www.youtube.com/watch?v=5qap5aO4i9A"),
            ("Box Breathing Tutorial", "https://www.youtube.com/watch?v=tEmt1Znux58"),
            ("Nature Relaxation", "https://www.youtube.com/watch?v=OdIJ2x3nxzQ"),
            ("Calm Piano", "https://www.youtube.com/watch?v=lFcSrYw-ARY"),
        ],
        "ENERGETIC": [
            ("Queen - Don’t Stop Me Now", "https://www.youtube.com/watch?v=HgzGwKwLmgM"),
            ("Eye of the Tiger", "https://www.youtube.com/watch?v=btPJPFnesV4"),
            ("Stronger Workout Mix", "https://www.youtube.com/watch?v=XqZsoesa55w"),
            ("Motivation Speech Mix", "https://www.youtube.com/watch?v=wnHW6o8WMas"),
            ("Power Focus Beats", "https://www.youtube.com/watch?v=QH2-TGUlwu4"),
        ],
    }

    query_terms = {
        "HAPPY": [
            "happy morning playlist",
            "feel good songs mix",
            "positive vibes music",
            "dance pop motivation songs",
            "joyful acoustic playlist",
            "sunshine playlist",
            "smile songs mix",
            "indie happy songs",
            "retro happy classics",
            "happy bollywood songs",
            "upbeat instrumental",
            "best mood booster songs",
            "road trip happy songs",
            "party warmup mix",
            "motivational happy playlist",
            "happy edm mix",
            "feel good chill playlist",
            "cheerful ukulele music",
            "good news playlist",
            "happy guitar songs",
            "weekend happy songs",
            "fun family playlist",
            "celebration songs",
            "sunset happy songs",
            "lighthearted music mix",
        ],
        "SAD": [
            "emotional healing playlist",
            "comfort songs when sad",
            "music for difficult days",
            "gentle piano for sadness",
            "heartbreak recovery playlist",
            "self compassion meditation",
            "grief support meditation",
            "slow emotional songs",
            "calming songs for anxiety",
            "soft rain with piano",
            "sad songs to release emotions",
            "mindful evening reflection music",
            "therapy journaling music",
            "relaxing prayer instrumental",
            "healing frequencies playlist",
            "late night calm songs",
            "lofi for emotional reset",
            "sleep and recover mix",
            "quiet vocals playlist",
            "restorative breathing music",
            "ambient healing soundtrack",
            "peaceful cello music",
            "slow acoustic ballads",
            "mental wellness songs",
            "gentle ocean sounds",
        ],
        "CALM": [
            "deep focus lofi",
            "calm coding music",
            "meditation with nature sounds",
            "study music no lyrics",
            "ambient concentration sounds",
            "mindfulness breathing session",
            "calm piano focus",
            "alpha waves concentration",
            "zen flute meditation",
            "binaural beats focus",
            "tea time jazz calm",
            "minimal ambient mix",
            "deep work playlist",
            "morning meditation calm",
            "sleep preparation music",
            "forest sounds relaxation",
            "rain sounds for reading",
            "spa relaxation music",
            "slow tempo chillhop",
            "calm instrumental guitar",
            "soft synth ambient",
            "evening unwind playlist",
            "mindful body scan audio",
            "focus music for students",
            "peaceful concentration sounds",
        ],
        "ENERGETIC": [
            "high intensity workout mix",
            "gym motivation songs",
            "running playlist 160 bpm",
            "powerlifting playlist",
            "sports hype music",
            "energetic edm drops",
            "drum and bass training mix",
            "boxing motivation songs",
            "pre exam motivation mix",
            "morning hustle playlist",
            "confidence boost songs",
            "hip hop pump up mix",
            "hardstyle training songs",
            "rock anthems workout",
            "focus and energy soundtrack",
            "dance cardio songs",
            "festival edm set",
            "high tempo motivation",
            "adrenaline songs playlist",
            "productivity power music",
            "driving energy mix",
            "crossfit playlist",
            "sprint training songs",
            "energetic percussion mix",
            "victory songs playlist",
        ],
    }

    catalog: dict[str, list[dict[str, str]]] = {}
    for category in ["HAPPY", "SAD", "CALM", "ENERGETIC"]:
        items: list[dict[str, str]] = []
        for idx, (title, url) in enumerate(seed_links[category], start=1):
            items.append(
                {
                    "id": f"{category[:3]}-seed-{idx:03d}",
                    "category": category,
                    "title": title,
                    "url": url,
                    "source_type": "youtube_video",
                    "offline_fallback": f"{category.lower()}_seed_{idx:03d}.mp3",
                }
            )
        for idx, query in enumerate(query_terms[category], start=1):
            items.append(
                {
                    "id": f"{category[:3]}-qry-{idx:03d}",
                    "category": category,
                    "title": query.title(),
                    "url": f"https://www.youtube.com/results?search_query={quote_plus(query)}",
                    "source_type": "youtube_search",
                    "offline_fallback": f"{category.lower()}_query_{idx:03d}.mp3",
                }
            )
        catalog[category] = items
    return catalog


RESOURCE_CATALOG = build_resource_catalog()


def flatten_catalog(catalog: dict[str, list[dict[str, str]]]) -> pd.DataFrame:
    rows = []
    for category, items in catalog.items():
        for item in items:
            row = dict(item)
            row["category"] = category
            rows.append(row)
    return pd.DataFrame(rows)


def persist_catalog_snapshot() -> None:
    df = flatten_catalog(RESOURCE_CATALOG)
    df.to_csv(CATALOG_EXPORT_FILE, index=False)


persist_catalog_snapshot()

# ----------------- Dataset source registry (A) -----------------
DATASET_SOURCE_REGISTRY = [
    {
        "key": "synthetic_2026",
        "name": "Synthetic Emotion Starter (low-RAM quick demo)",
        "year": 2026,
        "mode": "synthetic",
        "notes": "Generates simple face-like images locally for smoke testing.",
    },
    {
        "key": "fer2013_kaggle_2024",
        "name": "FER2013 via Kaggle mirror",
        "year": 2024,
        "mode": "kaggle",
        "dataset_ref": "msambare/fer2013",
        "notes": "Requires Kaggle API credentials configured on the machine.",
    },
    {
        "key": "hf_image_2025",
        "name": "HuggingFace image dataset (user-specified ID)",
        "year": 2025,
        "mode": "huggingface",
        "notes": "Works with image + label datasets if `datasets` package is installed.",
    },
    {
        "key": "custom_url_2026",
        "name": "Public ZIP/TAR URL (user-specified)",
        "year": 2026,
        "mode": "url",
        "notes": "Downloads and auto-prepares class folders or FER-style CSV layouts.",
    },
    {
        "key": "manual_upload_2023",
        "name": "Manual ZIP upload (RAF-DB / CK+ licensed packages)",
        "year": 2023,
        "mode": "upload",
        "notes": "Use when dataset URLs require authentication/approval.",
    },
]

# ----------------- Helpers -----------------
FER_LABEL_MAP = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def map_label_to_mood(label: str) -> str:
    t = (label or "").lower()
    if any(k in t for k in ["happy", "joy", "surprise", "smile", "excited"]):
        return "happy"
    if any(k in t for k in ["sad", "fear", "cry", "down"]):
        return "sad"
    if any(k in t for k in ["angry", "rage", "mad", "stress"]):
        return "energetic"
    return "calm"


def simple_text_to_mood(text: str) -> str:
    if not text:
        return "calm"
    t = text.lower()
    if any(w in t for w in ["sad", "unhappy", "depressed", "cry", "low"]):
        return "sad"
    if any(w in t for w in ["happy", "joy", "excited", "love", "great"]):
        return "happy"
    if any(w in t for w in ["angry", "mad", "irritat", "frustrat", "stress"]):
        return "energetic"
    return "calm"


def now_utc_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def init_recommender_stats() -> None:
    if not RL_STATS_FILE.exists():
        pd.DataFrame(
            columns=["ItemId", "ItemUrl", "Exposures", "Likes", "Skips", "LastRecommendedAt"]
        ).to_csv(RL_STATS_FILE, index=False)


def load_recommender_stats() -> pd.DataFrame:
    init_recommender_stats()
    try:
        return pd.read_csv(RL_STATS_FILE)
    except Exception:
        init_recommender_stats()
        return pd.read_csv(RL_STATS_FILE)


def upsert_recommender_stats(item: dict[str, str], liked: bool | None = None, skipped: bool | None = None) -> None:
    df = load_recommender_stats()
    item_id = item["id"]
    if item_id in df["ItemId"].values:
        df.loc[df["ItemId"] == item_id, "Exposures"] += 1
        if liked:
            df.loc[df["ItemId"] == item_id, "Likes"] += 1
        if skipped:
            df.loc[df["ItemId"] == item_id, "Skips"] += 1
        df.loc[df["ItemId"] == item_id, "LastRecommendedAt"] = now_utc_iso()
    else:
        row = {
            "ItemId": item_id,
            "ItemUrl": item["url"],
            "Exposures": 1,
            "Likes": int(bool(liked)),
            "Skips": int(bool(skipped)),
            "LastRecommendedAt": now_utc_iso(),
        }
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(RL_STATS_FILE, index=False)


def append_interaction_log(entry: dict[str, Any]) -> None:
    cols = [
        "User",
        "TimeUTC",
        "FaceLabel",
        "FaceMood",
        "EmojiMood",
        "VoiceMood",
        "TextMood",
        "FusedMood",
        "RecommendedId",
        "RecommendedUrl",
        "Feedback",
    ]
    row_df = pd.DataFrame([{k: entry.get(k, "") for k in cols}])
    if DATA_FILE.exists():
        old_df = pd.read_csv(DATA_FILE)
        out = pd.concat([old_df, row_df], ignore_index=True)
    else:
        out = row_df
    out.to_csv(DATA_FILE, index=False)


def read_user_recent_recommended_ids(user: str, last_n: int = 3) -> list[str]:
    if not DATA_FILE.exists():
        return []
    try:
        df = pd.read_csv(DATA_FILE)
        user_df = df[df["User"] == user]
        return user_df["RecommendedId"].dropna().astype(str).tolist()[-last_n:]
    except Exception:
        return []


def recommend_no_repeat_rl(user: str, mood: str, last_n: int = 3) -> dict[str, str]:
    candidates = RESOURCE_CATALOG.get(mood.upper(), [])
    if not candidates:
        candidates = [item for items in RESOURCE_CATALOG.values() for item in items]

    avoided = set(read_user_recent_recommended_ids(user=user, last_n=last_n))
    stats = load_recommender_stats()
    stats_idx = {str(r["ItemId"]): r for _, r in stats.iterrows()}
    total_exp = float(max(stats["Exposures"].sum(), 1)) if not stats.empty else 1.0

    best_item = None
    best_score = -10_000.0
    for item in candidates:
        row = stats_idx.get(item["id"])
        exposures = float(row["Exposures"]) if row is not None else 0.0
        likes = float(row["Likes"]) if row is not None else 0.0
        skips = float(row["Skips"]) if row is not None else 0.0

        base_reward = (likes + 1.0) / (exposures + 2.0) - 0.15 * (skips / (exposures + 1.0))
        exploration = math.sqrt(2.0 * math.log(total_exp + 2.0) / (exposures + 1.0))
        recent_penalty = -5.0 if item["id"] in avoided else 0.0
        jitter = random.random() * 0.03

        score = base_reward + 0.35 * exploration + recent_penalty + jitter
        if score > best_score:
            best_score = score
            best_item = item

    chosen = best_item if best_item is not None else random.choice(candidates)
    upsert_recommender_stats(chosen, liked=False, skipped=False)
    return chosen


def update_last_feedback(feedback: str) -> None:
    if not DATA_FILE.exists():
        return
    df = pd.read_csv(DATA_FILE)
    if not df.empty:
        df.loc[df.index[-1], "Feedback"] = feedback
        df.to_csv(DATA_FILE, index=False)


def record_voice_text(timeout: int = 5) -> str:
    if sr is None:
        return ""
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            audio = recognizer.listen(source, timeout=timeout)
        return recognizer.recognize_google(audio).lower()
    except Exception:
        return ""


def heuristic_face_mood_from_image_bytes(image_bytes: bytes | None) -> tuple[str, str]:
    if image_bytes is None:
        return "unknown", "calm"
    if cv2 is None:
        return "heuristic_unavailable", "calm"
    arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return "decode_failed", "calm"
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    if brightness < 80:
        return "dark_frame", "sad"
    if brightness > 155:
        return "bright_frame", "happy"
    return "neutral_brightness", "calm"


def build_transfer_model(num_classes: int, input_size: int = 160):
    if not TF_AVAILABLE:
        return None
    base = tf.keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(input_size, input_size, 3),
        alpha=0.35,
    )
    base.trainable = False
    inputs = layers.Input(shape=(input_size, input_size, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_image_for_model(image_bytes: bytes, input_size: int = 160) -> np.ndarray | None:
    if not PIL_AVAILABLE:
        return None
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((input_size, input_size))
        return np.array(image).astype("float32")
    except Exception:
        return None


def find_last_conv_layer_name(model) -> str | None:
    for layer in reversed(model.layers):
        if "conv" in layer.name.lower():
            return layer.name
    return None


def grad_cam_heatmap(img_array: np.ndarray, model, last_conv_layer_name: str | None = None) -> np.ndarray | None:
    if not TF_AVAILABLE:
        return None
    conv_name = last_conv_layer_name or find_last_conv_layer_name(model)
    if not conv_name:
        return None
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(conv_name).output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.array([img_array]))
        pred_idx = tf.argmax(predictions[0])
        loss = predictions[:, pred_idx]
    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        return None
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()
    if cv2 is None:
        return (heatmap * 255).astype(np.uint8)
    return cv2.resize((heatmap * 255).astype(np.uint8), (img_array.shape[1], img_array.shape[0]))


def overlay_heatmap_on_image(image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray | None:
    if cv2 is None or heatmap is None:
        return None
    try:
        img = image_rgb.astype(np.uint8)
        colored = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)
        return cv2.addWeighted(colored, alpha, img, 1.0 - alpha, 0)
    except Exception:
        return None


def evaluate_model_on_dataset(model, dataset, class_names: list[str]) -> dict[str, Any]:
    if not (TF_AVAILABLE and SKLEARN_AVAILABLE):
        return {"error": "tensorflow or sklearn unavailable"}
    y_true: list[int] = []
    y_pred: list[int] = []
    y_probs: list[list[float]] = []
    for x_batch, y_batch in dataset:
        preds = model.predict(x_batch, verbose=0)
        y_probs.extend(preds.tolist())
        y_pred.extend(np.argmax(preds, axis=1).tolist())
        y_true.extend(y_batch.numpy().tolist())

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    lb = LabelBinarizer()
    lb.fit(range(len(class_names)))
    y_true_bin = lb.transform(y_true)
    try:
        auc = roc_auc_score(y_true_bin, np.array(y_probs), average="macro", multi_class="ovr")
    except Exception:
        auc = None
    return {"report": report, "confusion_matrix": cm.tolist(), "roc_auc_macro": auc}


def spotify_search_tracks(mood: str, limit: int = 3) -> list[tuple[str, str]]:
    if sp is None:
        return []
    try:
        results = sp.search(q=f"{mood} mood", limit=limit, type="track")
        out = []
        for item in results["tracks"]["items"]:
            out.append((item["name"], item["external_urls"]["spotify"]))
        return out
    except Exception:
        return []


def openai_chat(prompt: str, system: str = "You are a helpful emotional AI assistant.") -> str:
    if not prompt.strip():
        return "Please enter a message."
    if not OPENAI_API_KEY or openai is None:
        return "OpenAI API not configured. Fallback tip: take 3 deep breaths and hydrate."
    try:
        if openai_client is not None:
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                max_tokens=300,
            )
            return (resp.choices[0].message.content or "").strip()
        if hasattr(openai, "ChatCompletion"):
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                max_tokens=300,
            )
            return resp.choices[0].message.content.strip()
    except Exception as exc:
        return f"OpenAI API error: {exc}"
    return "OpenAI package detected, but no compatible chat client path was found."


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def infer_dataset_root(root: Path) -> Path:
    current = root
    while True:
        subdirs = [p for p in current.iterdir() if p.is_dir()]
        files = [p for p in current.iterdir() if p.is_file()]
        if len(subdirs) == 1 and len(files) == 0:
            current = subdirs[0]
        else:
            return current


def discover_class_directories(root: Path) -> dict[str, list[Path]]:
    class_to_files: dict[str, list[Path]] = {}
    for child in root.iterdir():
        if child.is_dir():
            images = [p for p in child.rglob("*") if p.is_file() and is_image_file(p)]
            if images:
                class_to_files[child.name] = images
    return class_to_files


def split_imagefolder_dataset(
    class_to_files: dict[str, list[Path]],
    output_root: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, int]:
    random.seed(seed)
    counts = {"train": 0, "val": 0, "test": 0}
    for cls, files in class_to_files.items():
        files_copy = list(files)
        random.shuffle(files_copy)
        n = len(files_copy)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        if n_train + n_val >= n:
            n_val = max(1, n - n_train - 1)
        train_files = files_copy[:n_train]
        val_files = files_copy[n_train : n_train + n_val]
        test_files = files_copy[n_train + n_val :]
        if not test_files:
            test_files = val_files[-1:]
            val_files = val_files[:-1] or val_files

        for split_name, subset in [("train", train_files), ("val", val_files), ("test", test_files)]:
            split_dir = output_root / split_name / cls
            split_dir.mkdir(parents=True, exist_ok=True)
            for i, src in enumerate(subset, start=1):
                dst = split_dir / f"{src.stem}_{i:05d}{src.suffix.lower()}"
                shutil.copy2(src, dst)
                counts[split_name] += 1
    return counts


def extract_archive(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    lower_name = archive_path.name.lower()
    if lower_name.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(destination)
        return
    if lower_name.endswith(".tar.gz") or lower_name.endswith(".tgz") or lower_name.endswith(".tar"):
        with tarfile.open(archive_path, "r:*") as tfh:
            tfh.extractall(destination)
        return
    raise ValueError(f"Unsupported archive format: {archive_path.name}")


def download_file(url: str, target_path: Path) -> tuple[bool, str]:
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=90) as response, target_path.open("wb") as out_f:
            shutil.copyfileobj(response, out_f)
        return True, f"Downloaded to {target_path}"
    except Exception as exc:
        return False, str(exc)


def prepare_fer2013_csv(csv_path: Path, output_root: Path, max_rows: int = 12000) -> dict[str, int]:
    if not PIL_AVAILABLE:
        raise RuntimeError("Pillow is required for FER2013 CSV conversion.")
    df = pd.read_csv(csv_path)
    if "emotion" not in df.columns or "pixels" not in df.columns:
        raise ValueError("FER CSV must include at least: emotion, pixels.")

    has_usage = "Usage" in df.columns
    counts = {"train": 0, "val": 0, "test": 0}
    if max_rows > 0 and len(df) > max_rows:
        df = df.sample(max_rows, random_state=42).reset_index(drop=True)

    for idx, row in df.iterrows():
        label_id = int(row["emotion"])
        label_name = FER_LABEL_MAP.get(label_id, f"class_{label_id}")
        usage = str(row["Usage"]).lower() if has_usage else "training"

        if "train" in usage:
            split = "train"
        elif "public" in usage or "val" in usage:
            split = "val"
        else:
            split = "test"

        pixels = np.fromstring(str(row["pixels"]), sep=" ", dtype=np.uint8)
        if pixels.size != 48 * 48:
            continue
        image_array = pixels.reshape(48, 48)
        image = Image.fromarray(image_array).convert("RGB").resize((96, 96))

        out_dir = output_root / split / label_name
        out_dir.mkdir(parents=True, exist_ok=True)
        image.save(out_dir / f"fer_{idx:06d}.png")
        counts[split] += 1
    return counts


def auto_prepare_dataset(raw_root: Path, prepared_root: Path, fer_max_rows: int = 12000) -> tuple[bool, str]:
    ensure_clean_dir(prepared_root)
    candidate_root = infer_dataset_root(raw_root)

    # Case 1: FER-style CSV
    csv_candidates = [
        p
        for p in candidate_root.rglob("*.csv")
        if "fer" in p.name.lower() or {"emotion", "pixels"}.issubset(set(pd.read_csv(p, nrows=1).columns))
    ]
    for csv_path in csv_candidates:
        try:
            counts = prepare_fer2013_csv(csv_path, prepared_root, max_rows=fer_max_rows)
            return True, f"Prepared FER-style CSV dataset from {csv_path.name}: {counts}"
        except Exception:
            continue

    # Case 2: train/val or train/test already present
    train_dir = candidate_root / "train"
    val_dir = candidate_root / "val"
    test_dir = candidate_root / "test"
    if train_dir.exists() and train_dir.is_dir():
        if val_dir.exists():
            # Copy direct splits
            for split in ["train", "val", "test"]:
                src = candidate_root / split
                if src.exists():
                    shutil.copytree(src, prepared_root / split, dirs_exist_ok=True)
            return True, "Copied existing train/val/test splits."
        # split train and optionally merge test
        class_map = discover_class_directories(train_dir)
        if class_map:
            counts = split_imagefolder_dataset(class_map, prepared_root, train_ratio=0.85, val_ratio=0.15, seed=42)
            if test_dir.exists():
                test_map = discover_class_directories(test_dir)
                for cls, paths in test_map.items():
                    out = prepared_root / "test" / cls
                    out.mkdir(parents=True, exist_ok=True)
                    for i, src in enumerate(paths, start=1):
                        shutil.copy2(src, out / f"{src.stem}_{i:05d}{src.suffix.lower()}")
                        counts["test"] += 1
            return True, f"Prepared from train/(test) directories: {counts}"

    # Case 3: class folders at root
    root_class_map = discover_class_directories(candidate_root)
    if root_class_map and len(root_class_map) >= 2:
        counts = split_imagefolder_dataset(root_class_map, prepared_root, train_ratio=0.7, val_ratio=0.15, seed=42)
        return True, f"Prepared from class folders at dataset root: {counts}"

    # Case 4: nested class folders a few levels down
    for nested in candidate_root.rglob("*"):
        if nested.is_dir():
            depth = len(nested.relative_to(candidate_root).parts)
            if depth > 3:
                continue
            nested_map = discover_class_directories(nested)
            if nested_map and len(nested_map) >= 2:
                counts = split_imagefolder_dataset(nested_map, prepared_root, train_ratio=0.7, val_ratio=0.15, seed=42)
                return True, f"Prepared from nested folder {nested}: {counts}"

    return False, "Could not infer dataset structure. Provide class folders or FER-style CSV."


def generate_synthetic_emotion_dataset(
    output_root: Path, samples_per_class: int = 80, image_size: int = 96, seed: int = 42
) -> dict[str, int]:
    if not PIL_AVAILABLE:
        raise RuntimeError("Pillow is required for synthetic dataset generation.")
    random.seed(seed)
    np.random.seed(seed)
    ensure_clean_dir(output_root)
    counts = {"train": 0, "val": 0, "test": 0}
    classes = ["happy", "sad", "calm", "energetic"]

    def draw_face(label: str) -> Image.Image:
        img = Image.new("RGB", (image_size, image_size), (245, 245, 245))
        draw = ImageDraw.Draw(img)
        # face
        draw.ellipse((10, 10, image_size - 10, image_size - 10), fill=(255, 226, 180), outline=(30, 30, 30), width=2)
        # eyes
        draw.ellipse((30, 30, 38, 38), fill=(30, 30, 30))
        draw.ellipse((image_size - 38, 30, image_size - 30, 38), fill=(30, 30, 30))

        if label == "happy":
            draw.arc((28, 35, image_size - 28, image_size - 20), start=20, end=160, fill=(20, 20, 20), width=3)
        elif label == "sad":
            draw.arc((28, image_size - 40, image_size - 28, image_size - 15), start=200, end=340, fill=(20, 20, 20), width=3)
        elif label == "energetic":
            draw.line((28, 62, image_size - 28, 58), fill=(20, 20, 20), width=4)
            draw.polygon([(20, 20), (25, 10), (30, 20)], fill=(255, 60, 60))
            draw.polygon([(image_size - 20, 20), (image_size - 25, 10), (image_size - 30, 20)], fill=(255, 60, 60))
        else:  # calm
            draw.line((30, 60, image_size - 30, 60), fill=(20, 20, 20), width=3)

        # random noise for variability
        arr = np.array(img)
        noise = np.random.normal(0, 3.0, arr.shape).astype(np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    for cls in classes:
        for idx in range(samples_per_class):
            if idx < int(samples_per_class * 0.7):
                split = "train"
            elif idx < int(samples_per_class * 0.85):
                split = "val"
            else:
                split = "test"
            out_dir = output_root / split / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            draw_face(cls).save(out_dir / f"{cls}_{idx:04d}.png")
            counts[split] += 1
    return counts


def run_kaggle_download(dataset_ref: str, dest_root: Path) -> tuple[bool, str]:
    ensure_clean_dir(dest_root)
    cmd = ["kaggle", "datasets", "download", "-d", dataset_ref, "-p", str(dest_root), "--unzip"]
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True, proc.stdout or "Kaggle download completed."
    except Exception as exc:
        return False, str(exc)


def download_hf_image_dataset(
    dataset_id: str,
    output_root: Path,
    split_names: list[str] | None = None,
    max_images_per_split: int = 3000,
) -> tuple[bool, str]:
    if not HF_DATASETS_AVAILABLE:
        return False, "Hugging Face datasets package not installed. Install with: pip install datasets"
    if not PIL_AVAILABLE:
        return False, "Pillow is required to write images."

    ensure_clean_dir(output_root)
    split_names = split_names or ["train", "validation", "test"]
    try:
        ds = load_dataset(dataset_id)
    except Exception as exc:
        return False, f"load_dataset failed: {exc}"

    written = 0
    for split in split_names:
        if split not in ds:
            continue
        split_ds = ds[split]
        features = split_ds.features
        label_names = None
        if "label" in features and hasattr(features["label"], "names"):
            label_names = features["label"].names

        limit = min(max_images_per_split, len(split_ds))
        for idx in range(limit):
            row = split_ds[idx]
            if "image" not in row or "label" not in row:
                continue
            image_obj = row["image"]
            label = row["label"]
            label_name = (
                str(label_names[label])
                if label_names is not None and isinstance(label, int) and label < len(label_names)
                else f"class_{label}"
            )
            out_dir = output_root / split / label_name
            out_dir.mkdir(parents=True, exist_ok=True)
            image_obj.convert("RGB").save(out_dir / f"hf_{idx:06d}.jpg")
            written += 1
    if written == 0:
        return False, "No image/label rows were written from the selected dataset."
    return True, f"Wrote {written} images from Hugging Face dataset."


def dataset_split_summary(prepared_root: Path) -> dict[str, int]:
    out = {"train": 0, "val": 0, "test": 0}
    for split in out:
        split_dir = prepared_root / split
        if split_dir.exists():
            out[split] = sum(1 for p in split_dir.rglob("*") if p.is_file() and is_image_file(p))
    return out


def render_model_summary(model) -> str:
    lines: list[str] = []
    model.summary(print_fn=lines.append)
    return "\n".join(lines)


def load_class_names() -> list[str]:
    if CLASS_NAMES_FILE.exists():
        try:
            return json.loads(CLASS_NAMES_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def save_class_names(class_names: list[str]) -> None:
    CLASS_NAMES_FILE.write_text(json.dumps(class_names, indent=2), encoding="utf-8")


def maybe_render_confusion_matrix(cm: list[list[int]], class_names: list[str]) -> None:
    if not MATPLOTLIB_AVAILABLE:
        return
    arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(arr, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, str(arr[i, j]), ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(im)
    st.pyplot(fig)


def render_resource(item: dict[str, str]) -> None:
    url = item["url"]
    if "watch?v=" in url:
        st.video(url)
    else:
        st.markdown(f"[Open Resource: {item['title']}]({url})")


# ----------------- UI -----------------
st.title("🧠 CEI-ALOS — Cognitive Emotional Intelligence & Adaptive Lifestyle OS")
st.markdown(
    "Single-file Streamlit prototype with multi-modal fusion, transfer learning, explainability, "
    "adaptive recommendation, and dataset automation."
)

left, right = st.columns([1, 2])

with left:
    st.header("User & Inputs")
    user = st.text_input("Username", value="guest_user")
    emoji = st.select_slider("Select Emoji", options=list(EMOJI_MAP.keys()), value="😊")

    st.caption("Face input")
    camera_img = st.camera_input("Capture face (optional)")
    upload_img = st.file_uploader("Or upload image", type=["png", "jpg", "jpeg"], key="face_upload")

    face_image_bytes = None
    if upload_img is not None:
        face_image_bytes = upload_img.getvalue()
    elif camera_img is not None:
        face_image_bytes = camera_img.getvalue()

    st.write("Voice input (optional)")
    enable_mic = st.checkbox("Enable live microphone")
    voice_text = ""
    if enable_mic:
        st.info("Press record to capture a short sample.")
        if st.button("Record Voice (5s)"):
            with st.spinner("Recording..."):
                voice_text = record_voice_text(timeout=5)
            st.write(voice_text)
    else:
        voice_text = st.text_input("Or type voice/text signal", value="")

    user_text = st.text_area("Text context", value="")

with right:
    st.header("Option A — Dataset automation + Transfer Learning")
    st.markdown(
        "For low-RAM Windows systems, defaults are tuned for lighter usage "
        "(smaller image size, smaller batch, quick epochs)."
    )

    year_range = st.slider("Dataset year filter (latest editions)", 2023, 2026, (2023, 2026))
    available_sources = [
        s for s in DATASET_SOURCE_REGISTRY if year_range[0] <= int(s["year"]) <= year_range[1]
    ]
    source_labels = [f"{s['name']} [{s['year']}]" for s in available_sources]
    selected_label = st.selectbox("Select dataset source", source_labels)
    selected_source = available_sources[source_labels.index(selected_label)]
    st.caption(selected_source["notes"])

    fer_max_rows = st.slider("FER CSV conversion cap (rows)", 2000, 30000, 12000, step=1000)
    source_mode = selected_source["mode"]

    custom_url = ""
    hf_dataset_id = ""
    hf_max_images = 2000
    synthetic_per_class = 80
    upload_archive = None

    if source_mode == "url":
        custom_url = st.text_input("Public archive URL (.zip/.tar/.tar.gz/.tgz)", value="")
    elif source_mode == "huggingface":
        hf_dataset_id = st.text_input("Hugging Face dataset ID", value="")
        hf_max_images = st.slider("Max images per split", 500, 5000, 2000, step=250)
        if not HF_DATASETS_AVAILABLE:
            st.warning("`datasets` package not installed. Install: pip install datasets")
    elif source_mode == "synthetic":
        synthetic_per_class = st.slider("Synthetic samples per class", 40, 200, 80, step=10)
    elif source_mode == "upload":
        upload_archive = st.file_uploader(
            "Upload dataset archive (ZIP/TAR)", type=["zip", "tar", "gz", "tgz"], key="dataset_upload"
        )

    if st.button("Prepare Dataset (Auto)"):
        with st.spinner("Preparing dataset..."):
            raw_source_dir = RAW_DATASET_DIR / selected_source["key"]
            ensure_clean_dir(raw_source_dir)

            ok = False
            msg = ""
            if source_mode == "synthetic":
                counts = generate_synthetic_emotion_dataset(
                    PREPARED_DATASET_DIR, samples_per_class=synthetic_per_class, image_size=96
                )
                ok, msg = True, f"Synthetic dataset generated directly in prepared folder: {counts}"

            elif source_mode == "kaggle":
                success, out_msg = run_kaggle_download(selected_source["dataset_ref"], raw_source_dir)
                if success:
                    ok, msg = auto_prepare_dataset(raw_source_dir, PREPARED_DATASET_DIR, fer_max_rows=fer_max_rows)
                    msg = f"{out_msg}\n{msg}"
                else:
                    ok, msg = False, out_msg

            elif source_mode == "url":
                if not custom_url.strip():
                    ok, msg = False, "Please provide a public dataset URL."
                else:
                    suffix = ".zip"
                    lower_url = custom_url.lower()
                    if lower_url.endswith(".tar.gz"):
                        suffix = ".tar.gz"
                    elif lower_url.endswith(".tgz"):
                        suffix = ".tgz"
                    elif lower_url.endswith(".tar"):
                        suffix = ".tar"
                    archive_path = DOWNLOAD_DIR / f"{selected_source['key']}{suffix}"
                    success, out_msg = download_file(custom_url.strip(), archive_path)
                    if success:
                        try:
                            extract_archive(archive_path, raw_source_dir)
                            ok, msg = auto_prepare_dataset(
                                raw_source_dir, PREPARED_DATASET_DIR, fer_max_rows=fer_max_rows
                            )
                            msg = f"{out_msg}\n{msg}"
                        except Exception as exc:
                            ok, msg = False, f"Extraction/prepare failed: {exc}"
                    else:
                        ok, msg = False, out_msg

            elif source_mode == "huggingface":
                if not hf_dataset_id.strip():
                    ok, msg = False, "Please provide a Hugging Face dataset ID."
                else:
                    success, out_msg = download_hf_image_dataset(
                        dataset_id=hf_dataset_id.strip(),
                        output_root=raw_source_dir,
                        max_images_per_split=hf_max_images,
                    )
                    if success:
                        ok, msg = auto_prepare_dataset(raw_source_dir, PREPARED_DATASET_DIR, fer_max_rows=fer_max_rows)
                        msg = f"{out_msg}\n{msg}"
                    else:
                        ok, msg = False, out_msg

            elif source_mode == "upload":
                if upload_archive is None:
                    ok, msg = False, "Please upload an archive file first."
                else:
                    archive_path = DOWNLOAD_DIR / upload_archive.name
                    archive_path.write_bytes(upload_archive.getvalue())
                    try:
                        extract_archive(archive_path, raw_source_dir)
                        ok, msg = auto_prepare_dataset(raw_source_dir, PREPARED_DATASET_DIR, fer_max_rows=fer_max_rows)
                    except Exception as exc:
                        ok, msg = False, f"Upload extraction failed: {exc}"

            if ok:
                summary = dataset_split_summary(PREPARED_DATASET_DIR)
                st.success(msg)
                st.info(f"Prepared split summary: {summary}")
                metadata = {
                    "source_key": selected_source["key"],
                    "source_name": selected_source["name"],
                    "source_year": selected_source["year"],
                    "prepared_at": now_utc_iso(),
                    "split_summary": summary,
                }
                (PREPARED_DATASET_DIR / "metadata.json").write_text(
                    json.dumps(metadata, indent=2), encoding="utf-8"
                )
            else:
                st.error(msg)

    st.markdown("---")
    if TF_AVAILABLE:
        input_size = st.selectbox("Image size", [128, 160, 192, 224], index=1)
        batch_size = st.selectbox("Batch size", [4, 8, 16, 32], index=1)
        quick_epochs = st.slider("Quick training epochs", 1, 5, 2)
        num_classes_manual = st.number_input("Manual class count (Build only)", 2, 12, 4)

        if st.button("Build Model (MobileNetV2)"):
            model = build_transfer_model(num_classes=int(num_classes_manual), input_size=int(input_size))
            if model is not None:
                model_path = MODEL_DIR / "mobile_transfer_base.keras"
                model.save(model_path)
                st.success(f"Built and saved base model to {model_path}")
                st.text(render_model_summary(model))

        if st.button("Train + Evaluate (prepared dataset)"):
            train_dir = PREPARED_DATASET_DIR / "train"
            val_dir = PREPARED_DATASET_DIR / "val"
            if not train_dir.exists() or not val_dir.exists():
                st.error("Prepared dataset missing train/val folders. Run 'Prepare Dataset (Auto)' first.")
            else:
                try:
                    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
                        train_dir,
                        image_size=(int(input_size), int(input_size)),
                        batch_size=int(batch_size),
                        seed=42,
                    )
                    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
                        val_dir,
                        image_size=(int(input_size), int(input_size)),
                        batch_size=int(batch_size),
                        seed=42,
                    )
                    class_names = list(train_ds.class_names)
                    model = build_transfer_model(num_classes=len(class_names), input_size=int(input_size))
                    aug = models.Sequential(
                        [
                            layers.RandomFlip("horizontal"),
                            layers.RandomRotation(0.08),
                        ]
                    )
                    autotune = tf.data.AUTOTUNE
                    train_ds_aug = (
                        train_ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=autotune)
                        .prefetch(autotune)
                    )
                    val_ds_prefetch = val_ds.prefetch(autotune)

                    with st.spinner("Training in progress..."):
                        history = model.fit(train_ds_aug, validation_data=val_ds_prefetch, epochs=int(quick_epochs))
                    st.success("Training complete.")
                    st.line_chart(
                        pd.DataFrame(
                            {
                                "train_acc": history.history.get("accuracy", []),
                                "val_acc": history.history.get("val_accuracy", []),
                            }
                        )
                    )

                    trained_path = MODEL_DIR / "mobile_transfer_trained.keras"
                    model.save(trained_path)
                    save_class_names(class_names)
                    st.info(f"Saved trained model to {trained_path}")
                    st.info(f"Class names saved to {CLASS_NAMES_FILE}")

                    if SKLEARN_AVAILABLE:
                        eval_result = evaluate_model_on_dataset(model, val_ds_prefetch, class_names)
                        st.subheader("Evaluation metrics")
                        st.json(eval_result)
                        cm = eval_result.get("confusion_matrix")
                        if cm is not None:
                            maybe_render_confusion_matrix(cm, class_names)
                    else:
                        st.warning("scikit-learn unavailable; skipping detailed evaluation metrics.")
                except Exception as exc:
                    st.error(f"Training error: {exc}")
    else:
        st.warning(
            "TensorFlow unavailable. Install for transfer learning features: "
            "`pip install tensorflow` (or tensorflow-cpu)."
        )

st.markdown("---")
if st.button("🚀 Analyze & Recommend"):
    face_label = "none"
    face_mood = "calm"

    trained_model_path = MODEL_DIR / "mobile_transfer_trained.keras"
    class_names = load_class_names()
    if face_image_bytes:
        if TF_AVAILABLE and trained_model_path.exists():
            try:
                model = tf.keras.models.load_model(trained_model_path)
                arr = load_image_for_model(face_image_bytes, input_size=model.input_shape[1])
                if arr is not None:
                    pred = model.predict(np.array([arr]), verbose=0)[0]
                    pred_idx = int(np.argmax(pred))
                    if 0 <= pred_idx < len(class_names):
                        face_label = class_names[pred_idx]
                    else:
                        face_label = f"class_{pred_idx}"
                    face_mood = map_label_to_mood(face_label)

                    heatmap = grad_cam_heatmap(arr, model)
                    if heatmap is not None:
                        overlay = overlay_heatmap_on_image(arr.astype(np.uint8), heatmap, alpha=0.45)
                        if overlay is not None:
                            st.image(overlay, caption="Grad-CAM overlay", channels="BGR")
            except Exception:
                face_label, face_mood = heuristic_face_mood_from_image_bytes(face_image_bytes)
        else:
            face_label, face_mood = heuristic_face_mood_from_image_bytes(face_image_bytes)
    else:
        face_label, face_mood = "no_face_input", "calm"

    emoji_mood = EMOJI_MAP.get(emoji, "calm")
    voice_mood = simple_text_to_mood(voice_text)
    text_mood = simple_text_to_mood(user_text)

    # Weighted fusion
    scores = {"happy": 0.0, "sad": 0.0, "calm": 0.0, "energetic": 0.0}
    if face_mood in scores:
        scores[face_mood] += 0.40
    scores[emoji_mood] += 0.25
    scores[voice_mood] += 0.175
    scores[text_mood] += 0.175
    fused_mood = max(scores.items(), key=lambda kv: kv[1])[0]

    st.success(
        f"Face={face_label} ({face_mood}) | Emoji={emoji_mood} | Voice={voice_mood} | Text={text_mood}"
    )
    st.info(f"Fused mood: **{fused_mood}**")

    chosen_item = recommend_no_repeat_rl(user=user, mood=fused_mood, last_n=3)
    st.subheader("Recommended resource")
    st.caption(f"{chosen_item['title']} [{chosen_item['source_type']}]")
    render_resource(chosen_item)

    tracks = spotify_search_tracks(fused_mood, limit=3)
    if tracks:
        st.write("Spotify picks:")
        for name, url in tracks:
            st.markdown(f"- [{name}]({url})")
    else:
        st.info("Spotify unavailable or not configured.")

    append_interaction_log(
        {
            "User": user,
            "TimeUTC": now_utc_iso(),
            "FaceLabel": face_label,
            "FaceMood": face_mood,
            "EmojiMood": emoji_mood,
            "VoiceMood": voice_mood,
            "TextMood": text_mood,
            "FusedMood": fused_mood,
            "RecommendedId": chosen_item["id"],
            "RecommendedUrl": chosen_item["url"],
            "Feedback": "",
        }
    )
    st.success("Interaction logged to Digital Emotional Twin.")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("👍 I liked this", key="liked_btn"):
            upsert_recommender_stats(chosen_item, liked=True, skipped=False)
            update_last_feedback("liked")
            st.success("Preference updated.")
    with c2:
        if st.button("👎 Skip", key="skip_btn"):
            upsert_recommender_stats(chosen_item, liked=False, skipped=True)
            update_last_feedback("skipped")
            st.info("Skip logged.")

st.markdown("---")
st.header("Chat with Emotional AI")
chat_prompt = st.text_input("Message", value="")
if st.button("Send Chat"):
    st.write(openai_chat(chat_prompt))

st.markdown("---")
st.header("Dashboard & Analytics")
dash_c1, dash_c2, dash_c3 = st.columns(3)

with dash_c1:
    if st.button("Show Twin Logs"):
        if DATA_FILE.exists():
            df_logs = pd.read_csv(DATA_FILE)
            st.dataframe(df_logs, use_container_width=True)
            if not df_logs.empty and "FusedMood" in df_logs.columns:
                st.bar_chart(df_logs["FusedMood"].value_counts())
        else:
            st.info("No logs yet.")

with dash_c2:
    if st.button("Show Recommender Stats"):
        rdf = load_recommender_stats()
        st.dataframe(rdf, use_container_width=True)
        if not rdf.empty:
            rdf["ctr_like"] = rdf["Likes"] / rdf["Exposures"].clip(lower=1)
            st.bar_chart(rdf.set_index("ItemId")["ctr_like"])

with dash_c3:
    if st.button("Export Analytics Snapshot CSV"):
        if DATA_FILE.exists():
            df_logs = pd.read_csv(DATA_FILE)
        else:
            df_logs = pd.DataFrame()
        rdf = load_recommender_stats()
        out_dir = APP_ROOT / "analytics_exports"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path_logs = out_dir / f"twin_logs_{ts}.csv"
        path_rl = out_dir / f"recommender_stats_{ts}.csv"
        df_logs.to_csv(path_logs, index=False)
        rdf.to_csv(path_rl, index=False)
        st.success(f"Saved {path_logs.name} and {path_rl.name}")

st.markdown("---")
st.header("Option B — Structured Resource Catalog (100+)")
catalog_df = flatten_catalog(RESOURCE_CATALOG)
st.caption(f"Total resources: {len(catalog_df)} (target >=100)")
catalog_query = st.text_input("Filter resources (title/url/category contains)", value="")
if catalog_query.strip():
    q = catalog_query.strip().lower()
    mask = (
        catalog_df["title"].str.lower().str.contains(q, na=False)
        | catalog_df["url"].str.lower().str.contains(q, na=False)
        | catalog_df["category"].str.lower().str.contains(q, na=False)
    )
    filtered_df = catalog_df[mask]
else:
    filtered_df = catalog_df

st.dataframe(filtered_df, use_container_width=True, height=320)
catalog_csv = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download catalog CSV",
    data=catalog_csv,
    file_name="resource_catalog_filtered.csv",
    mime="text/csv",
)
st.caption(f"Full snapshot persisted to: {CATALOG_EXPORT_FILE.resolve()}")

for category, items in RESOURCE_CATALOG.items():
    with st.expander(f"{category} ({len(items)} resources)"):
        for item in items[:10]:
            st.markdown(f"- [{item['title']}]({item['url']})")
        if len(items) > 10:
            st.caption("Showing first 10 items here; full list available in the table/CSV.")

st.markdown("---")
st.header("Developer / API setup")
st.markdown(
    """
1) OpenAI:
   - Create key at https://platform.openai.com
   - Set environment variable:
     - Windows PowerShell: `$env:OPENAI_API_KEY="your_key"`
     - macOS/Linux: `export OPENAI_API_KEY="your_key"`

2) Spotify:
   - Create app at https://developer.spotify.com
   - Set:
     - `SPOTIPY_CLIENT_ID`
     - `SPOTIPY_CLIENT_SECRET`

3) Install dependencies:
   - `pip install streamlit opencv-python-headless numpy pandas tensorflow scikit-learn matplotlib pillow speechrecognition spotipy openai datasets`

4) Run:
   - `streamlit run app.py`
"""
)

st.markdown("---")
st.header("Android / Windows install recommendation (no paid wrapper)")
st.markdown(
    """
- Host the Streamlit app on a public URL (Streamlit Cloud, Render, Railway, etc.).
- Android: open URL in Chrome -> **Add to Home screen** (WebAPK-like install icon).
- Windows 11: open in Edge/Chrome -> **Install app** (PWA install).
- This is the most practical no-cost installable path without paid APK wrapper services.

If you need a true native APK, use Kivy/Buildozer or Chaquopy with a dedicated build pipeline.
"""
)

st.markdown("---")
st.header("Notes / limits")
st.markdown(
    """
- Full transfer-learning features require TensorFlow and labeled image data.
- Datasets like RAF-DB / CK+ may require manual license acceptance; use manual upload mode.
- CSV logging is for prototypes. For production, move to a secure DB and server-side key management.
- Keep batch size low on 4GB RAM systems for stable execution.
"""
)

st.write("CEI-ALOS prototype ready.")

