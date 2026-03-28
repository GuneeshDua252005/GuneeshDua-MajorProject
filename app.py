"""
CEI-ALOS unified Streamlit app.

Includes:
- Multi-modal fusion (face, voice, emoji, text)
- Transfer learning model training/evaluation (MobileNetV2)
- Grad-CAM explainability
- Adaptive no-repetition recommender with feedback stats
- Spotify + OpenAI integration (optional)
- Digital Emotional Twin logging
- Option A: automatic dataset preparation
  - Public no-auth downloader (GitHub facial-expressions dataset)
  - FER2013 CSV parser (upload or Kaggle-assisted download)
- Option B: 100+ structured resource catalog with CSV export/import
"""

from __future__ import annotations

import csv
import io
import json
import os
import random
import shutil
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote, quote_plus
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import streamlit as st

# Optional / heavy imports (guarded)
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
    from PIL import Image

    PIL_AVAILABLE = True
except Exception:
    Image = None
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

# ---------------- Settings ----------------
st.set_page_config(page_title="CEI-ALOS FINAL", layout="wide")

DATA_FILE = Path("cei_twin_log.csv")
RL_STATS_FILE = Path("recommender_stats.csv")
MODEL_DIR = Path("models")
CLASS_NAMES_FILE = MODEL_DIR / "class_names.json"
CATALOG_FILE = Path("resource_catalog.csv")
DATASET_DIR = Path("dataset")
TMP_DIR = Path("tmp")

MODEL_DIR.mkdir(exist_ok=True)
TMP_DIR.mkdir(exist_ok=True)

OPEN_FER_LEGEND_URL = "https://raw.githubusercontent.com/muxspace/facial_expressions/master/data/legend.csv"
OPEN_FER_IMAGE_BASE = "https://raw.githubusercontent.com/muxspace/facial_expressions/master/images/"

EMOJI_MAP = {"😊": "happy", "😢": "sad", "😡": "energetic", "😴": "calm", "😍": "happy"}
FER2013_LABELS = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}
OPEN_DATASET_MOOD_MAP = {
    "anger": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "happiness": "happy",
    "sadness": "sad",
    "surprise": "surprise",
    "neutral": "neutral",
}
MOOD_NORMALIZATION_MAP = {
    "angry": "energetic",
    "anger": "energetic",
    "mad": "energetic",
    "irritated": "energetic",
    "energetic": "energetic",
    "fear": "energetic",
    "surprise": "energetic",
    "happy": "happy",
    "joy": "happy",
    "calm": "calm",
    "neutral": "calm",
    "sad": "sad",
    "disgust": "sad",
}
MOOD_TO_CATEGORY = {
    "happy": "HAPPY",
    "sad": "SAD",
    "calm": "CALM",
    "energetic": "ENERGETIC",
}

# ---------------- API setup ----------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
SPOTIPY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID", "")
SPOTIPY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET", "")

OPENAI_CLIENT = None
if OPENAI_API_KEY:
    try:
        # New SDK path
        from openai import OpenAI

        OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        OPENAI_CLIENT = None
        if openai is not None:
            try:
                openai.api_key = OPENAI_API_KEY
            except Exception:
                pass

if spotipy and SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET:
    try:
        SPOTIFY_CLIENT = spotipy.Spotify(
            auth_manager=SpotifyClientCredentials(
                client_id=SPOTIPY_CLIENT_ID,
                client_secret=SPOTIPY_CLIENT_SECRET,
            )
        )
    except Exception:
        SPOTIFY_CLIENT = None
else:
    SPOTIFY_CLIENT = None


# ---------------- Utility helpers ----------------
def normalize_mood(label: str | None) -> str:
    if not label:
        return "calm"
    key = str(label).strip().lower()
    return MOOD_NORMALIZATION_MAP.get(key, "calm")


def read_csv_safe(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns or [])
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=columns or [])
    if columns:
        for c in columns:
            if c not in df.columns:
                df[c] = ""
        return df[columns]
    return df


def http_get_bytes(url: str, timeout: int = 30, retries: int = 2) -> bytes:
    headers = {"User-Agent": "Mozilla/5.0"}
    exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=timeout) as response:
                return response.read()
        except Exception as err:
            exc = err
            if attempt < retries:
                time.sleep(1.5 * (2**attempt))
    if exc is not None:
        raise exc
    raise RuntimeError("Unknown download error.")


def clear_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def split_list(
    items: list[Any],
    train_ratio: float,
    val_ratio: float,
) -> tuple[list[Any], list[Any], list[Any]]:
    n = len(items)
    if n == 0:
        return [], [], []
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    n_test = max(1, n - n_train - n_val)
    train = items[:n_train]
    val = items[n_train : n_train + n_val]
    test = items[n_train + n_val : n_train + n_val + n_test]
    return train, val, test


def maybe_resize_and_save_image(image_bytes: bytes, out_path: Path, image_size: tuple[int, int] = (224, 224)) -> bool:
    try:
        if PIL_AVAILABLE:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(image_size)
            img.save(out_path, format="JPEG", quality=90)
            return True
        if cv2 is not None:
            arr = np.frombuffer(image_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                return False
            frame = cv2.resize(frame, image_size)
            return bool(cv2.imwrite(str(out_path), frame))
        return False
    except Exception:
        return False


# ---------------- Option B: Resource catalog ----------------
def build_default_catalog_entries() -> list[dict[str, str]]:
    # 5 categories x 25 queries = 125 generated entries + direct links.
    query_bank = {
        "HAPPY": [
            "feel good pop playlist",
            "happy morning songs",
            "uplifting indie pop mix",
            "sunny day music playlist",
            "positive vibes music",
            "smile songs playlist",
            "good mood boost songs",
            "dance happy hits",
            "celebration songs mix",
            "happy acoustic covers",
            "joyful instrumental playlist",
            "feel better songs",
            "weekend happy mix",
            "good news music playlist",
            "retro feel good songs",
            "upbeat chill songs",
            "happy road trip playlist",
            "best mood lifting songs",
            "clean pop motivation",
            "happy lofi mix",
            "bright piano playlist",
            "happy guitar instrumentals",
            "mood reset songs",
            "spring vibes playlist",
            "warm happy jazz mix",
        ],
        "SAD": [
            "healing music for sadness",
            "comfort songs playlist",
            "soft piano for emotional release",
            "songs for difficult days",
            "gentle acoustic sad mix",
            "night rain and calm music",
            "emotional recovery playlist",
            "breakup healing songs",
            "mindful sadness playlist",
            "calm voice affirmations",
            "relaxing cello music",
            "heartbreak recovery mix",
            "soft songs to cry to",
            "slow emotional pop songs",
            "self compassion music",
            "peaceful guitar ballads",
            "lofi for rainy mood",
            "soothing instrumental mood",
            "deep breathing meditation music",
            "calming hymns instrumental",
            "sleepy emotional playlist",
            "slow piano jazz night",
            "sad to calm transition songs",
            "mental health support playlist",
            "quiet support songs",
        ],
        "CALM": [
            "meditation music for calm",
            "deep focus calm lofi",
            "ambient relaxation playlist",
            "nature sounds and piano",
            "morning mindfulness music",
            "stress relief breathing sounds",
            "spa calming music mix",
            "yoga background music",
            "calm instrumental guitar",
            "soft rain white noise",
            "theta waves relaxation",
            "sleep meditation soundscape",
            "study calm concentration music",
            "peaceful classical playlist",
            "calm ocean sounds",
            "forest birds relaxation",
            "zen music playlist",
            "healing frequency music",
            "calm jazz instrumental",
            "tranquil flute meditation",
            "quiet mind reset music",
            "grounding meditation audio",
            "slow ambient electronic",
            "focus and calm beats",
            "relaxing piano stream",
        ],
        "ENERGETIC": [
            "workout motivation songs",
            "gym power playlist",
            "high energy dance mix",
            "running playlist bpm",
            "electro hype music",
            "hip hop workout tracks",
            "focus and energy music",
            "morning energy boost songs",
            "pump up anthems",
            "rock motivation playlist",
            "sports training music",
            "edm festival mix",
            "cardio playlist nonstop",
            "drum and bass energy",
            "fast paced study beats",
            "confidence boost songs",
            "action cinematic music",
            "strong beats playlist",
            "power lifting music mix",
            "high tempo no lyrics",
            "pre workout hype songs",
            "dance cardio music",
            "adrenaline music playlist",
            "upbeat energetic pop",
            "high intensity interval tracks",
        ],
        "FOCUS": [
            "deep work lofi playlist",
            "coding music no lyrics",
            "pomodoro focus mix",
            "study beats concentration",
            "classical focus playlist",
            "brown noise for studying",
            "ambient coding music",
            "productive morning soundtrack",
            "minimal techno focus",
            "instrumental math study mix",
            "concentration binaural beats",
            "focus piano playlist",
            "work music for office",
            "long focus session music",
            "quiet background for reading",
            "brain power study playlist",
            "exam preparation music",
            "task flow instrumental",
            "attention boost music",
            "focus jazz instrumental",
            "ultra concentrate mix",
            "reading soundtrack ambient",
            "calm deep work sounds",
            "coding synthwave focus",
            "deep concentration soundscape",
        ],
    }

    direct_links = {
        "HAPPY": [
            "https://www.youtube.com/watch?v=ZbZSe6N_BXs",
            "https://www.youtube.com/watch?v=OPf0YbXqDm0",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
        ],
        "SAD": [
            "https://www.youtube.com/watch?v=2XU0oxnq2qU",
            "https://www.youtube.com/watch?v=inpok4MKVLM",
            "https://www.youtube.com/watch?v=1vx8iUvfyCY",
        ],
        "CALM": [
            "https://www.youtube.com/watch?v=5qap5aO4i9A",
            "https://www.youtube.com/watch?v=v7AYKMP6rOE",
            "https://www.youtube.com/watch?v=1ZYbU82GVz4",
        ],
        "ENERGETIC": [
            "https://www.youtube.com/watch?v=HgzGwKwLmgM",
            "https://www.youtube.com/watch?v=ml6cT4AZdqI",
            "https://www.youtube.com/watch?v=fLexgOxsZu0",
        ],
        "FOCUS": [
            "https://www.youtube.com/watch?v=jfKfPfyJRdk",
            "https://www.youtube.com/watch?v=lTRiuFIWV54",
            "https://www.youtube.com/watch?v=DWcJFNfaw9c",
        ],
    }

    entries: list[dict[str, str]] = []
    for category, queries in query_bank.items():
        for i, query in enumerate(queries, start=1):
            entries.append(
                {
                    "Category": category,
                    "Title": f"{category.title()} Query {i}",
                    "URL": f"https://www.youtube.com/results?search_query={quote_plus(query)}",
                    "Type": "youtube_search",
                    "Tags": query.replace(" ", ","),
                }
            )
    for category, links in direct_links.items():
        for i, link in enumerate(links, start=1):
            entries.append(
                {
                    "Category": category,
                    "Title": f"{category.title()} Direct {i}",
                    "URL": link,
                    "Type": "youtube_video",
                    "Tags": "direct,seed",
                }
            )
    return entries


def ensure_catalog_csv(path: Path = CATALOG_FILE) -> None:
    if path.exists():
        return
    df = pd.DataFrame(build_default_catalog_entries())
    df.to_csv(path, index=False)


def load_catalog_df(path: Path = CATALOG_FILE) -> pd.DataFrame:
    ensure_catalog_csv(path)
    expected = ["Category", "Title", "URL", "Type", "Tags"]
    df = read_csv_safe(path, expected)
    if df.empty:
        return pd.DataFrame(columns=expected)
    df["Category"] = df["Category"].astype(str).str.upper().fillna("MISC")
    df["Title"] = df["Title"].astype(str)
    df["URL"] = df["URL"].astype(str)
    df = df[df["URL"].str.startswith("http")]
    return df


def catalog_to_lookup(df: pd.DataFrame) -> dict[str, list[dict[str, str]]]:
    lookup: dict[str, list[dict[str, str]]] = {}
    for _, row in df.iterrows():
        category = str(row["Category"]).upper()
        lookup.setdefault(category, []).append(
            {
                "title": str(row["Title"]),
                "url": str(row["URL"]),
                "type": str(row["Type"]),
            }
        )
    return lookup


# ---------------- Digital Emotional Twin logging ----------------
def log_interaction(
    user: str,
    face_mood: str,
    emoji_mood: str,
    voice_mood: str,
    text_mood: str,
    fused_mood: str,
    recommended_url: str,
    recommended_title: str,
    feedback: str | None = None,
) -> None:
    cols = [
        "User",
        "TimeUTC",
        "Face",
        "Emoji",
        "Voice",
        "Text",
        "Fused",
        "RecommendedTitle",
        "RecommendedURL",
        "Feedback",
    ]
    old = read_csv_safe(DATA_FILE, cols)
    row = {
        "User": user,
        "TimeUTC": datetime.utcnow().isoformat(),
        "Face": face_mood,
        "Emoji": emoji_mood,
        "Voice": voice_mood,
        "Text": text_mood,
        "Fused": fused_mood,
        "RecommendedTitle": recommended_title,
        "RecommendedURL": recommended_url,
        "Feedback": feedback or "",
    }
    out = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
    out.to_csv(DATA_FILE, index=False)


def update_last_feedback(feedback: str) -> None:
    cols = [
        "User",
        "TimeUTC",
        "Face",
        "Emoji",
        "Voice",
        "Text",
        "Fused",
        "RecommendedTitle",
        "RecommendedURL",
        "Feedback",
    ]
    df = read_csv_safe(DATA_FILE, cols)
    if df.empty:
        return
    df.loc[df.index[-1], "Feedback"] = feedback
    df.to_csv(DATA_FILE, index=False)


# ---------------- RL-style recommender stats ----------------
def init_recommender_stats() -> None:
    if RL_STATS_FILE.exists():
        return
    pd.DataFrame(columns=["Item", "Exposures", "Likes"]).to_csv(RL_STATS_FILE, index=False)


def update_stats(item: str, liked: bool = False) -> None:
    init_recommender_stats()
    df = read_csv_safe(RL_STATS_FILE, ["Item", "Exposures", "Likes"])
    if item in df["Item"].values:
        df.loc[df["Item"] == item, "Exposures"] = df.loc[df["Item"] == item, "Exposures"].astype(int) + 1
        if liked:
            df.loc[df["Item"] == item, "Likes"] = df.loc[df["Item"] == item, "Likes"].astype(int) + 1
    else:
        df = pd.concat(
            [df, pd.DataFrame([{"Item": item, "Exposures": 1, "Likes": int(liked)}])],
            ignore_index=True,
        )
    df.to_csv(RL_STATS_FILE, index=False)


def recommend_no_repeat(
    user: str,
    mood: str,
    catalog_df: pd.DataFrame,
    last_n: int = 3,
) -> dict[str, str]:
    category = MOOD_TO_CATEGORY.get(mood, "CALM")
    mood_rows = catalog_df[catalog_df["Category"] == category]
    if mood_rows.empty:
        mood_rows = catalog_df

    # user history for no-repeat constraint
    log_df = read_csv_safe(
        DATA_FILE,
        [
            "User",
            "TimeUTC",
            "Face",
            "Emoji",
            "Voice",
            "Text",
            "Fused",
            "RecommendedTitle",
            "RecommendedURL",
            "Feedback",
        ],
    )
    if log_df.empty:
        recent = []
    else:
        recent = (
            log_df[log_df["User"] == user]["RecommendedURL"].dropna().astype(str).tolist()[-last_n:]
        )
    avoided = set(recent)

    candidates = []
    for _, row in mood_rows.iterrows():
        entry = {
            "title": str(row["Title"]),
            "url": str(row["URL"]),
            "category": str(row["Category"]),
        }
        if entry["url"] not in avoided:
            candidates.append(entry)
    if not candidates:
        candidates = [
            {
                "title": str(row["Title"]),
                "url": str(row["URL"]),
                "category": str(row["Category"]),
            }
            for _, row in mood_rows.iterrows()
        ]

    init_recommender_stats()
    stats = read_csv_safe(RL_STATS_FILE, ["Item", "Exposures", "Likes"])
    stats_map = {
        str(r["Item"]): (
            int(r["Exposures"]) if str(r["Exposures"]).strip() else 0,
            int(r["Likes"]) if str(r["Likes"]).strip() else 0,
        )
        for _, r in stats.iterrows()
    }

    # UCB-ish score: prefer less exposed items, but keep liked ones.
    scored: list[tuple[float, dict[str, str]]] = []
    for item in candidates:
        exp, likes = stats_map.get(item["url"], (0, 0))
        quality = (likes + 1.0) / (exp + 2.0)
        novelty = 1.0 / (1.0 + exp)
        score = 0.65 * quality + 0.35 * novelty + random.random() * 0.05
        scored.append((score, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    chosen = scored[0][1] if scored else random.choice(candidates)
    update_stats(chosen["url"], liked=False)
    return chosen


# ---------------- Audio/Text/Face helpers ----------------
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


def simple_text_to_mood(text: str) -> str:
    if not text:
        return "calm"
    t = text.lower()
    if any(word in t for word in ["sad", "unhappy", "depressed", "cry", "down"]):
        return "sad"
    if any(word in t for word in ["happy", "joy", "excited", "love", "great"]):
        return "happy"
    if any(word in t for word in ["angry", "mad", "irritat", "frustrat"]):
        return "energetic"
    return "calm"


def heuristic_face_mood_from_bytes(image_bytes: bytes | None) -> str:
    if cv2 is None or image_bytes is None:
        return "calm"
    try:
        arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return "calm"
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        if brightness < 85:
            return "sad"
        if brightness > 150:
            return "happy"
        return "calm"
    except Exception:
        return "calm"


# ---------------- Transfer Learning + Grad-CAM ----------------
def build_transfer_model(num_classes: int):
    if not TF_AVAILABLE:
        return None
    base = tf.keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
    )
    base.trainable = False
    inputs = layers.Input(shape=(224, 224, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_image_for_model(image_bytes: bytes):
    if not TF_AVAILABLE:
        return None
    try:
        if PIL_AVAILABLE:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
            return np.array(img, dtype=np.float32)
        if cv2 is not None:
            arr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                return None
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            return frame.astype(np.float32)
    except Exception:
        return None
    return None


def get_last_conv_layer_name(model) -> str | None:
    if not TF_AVAILABLE:
        return None
    for layer in reversed(model.layers):
        try:
            output_shape = getattr(layer, "output_shape", None)
            if output_shape is not None and len(output_shape) == 4:
                return layer.name
        except Exception:
            pass
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None


def grad_cam_heatmap(img_array: np.ndarray, model, last_conv_layer_name: str):
    if not TF_AVAILABLE:
        return None
    try:
        grad_model = tf.keras.models.Model(
            [model.inputs],
            [model.get_layer(last_conv_layer_name).output, model.output],
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(np.array([img_array]))
            pred_idx = tf.argmax(predictions[0])
            loss = predictions[:, pred_idx]
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()
        if cv2 is not None:
            heatmap = cv2.resize(heatmap, (224, 224))
        else:
            heatmap = np.resize(heatmap, (224, 224))
        heatmap = np.uint8(255 * heatmap)
        return heatmap
    except Exception:
        return None


def overlay_heatmap(image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4):
    if cv2 is None:
        return None
    try:
        colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(colored, alpha, image_rgb.astype(np.uint8), 1 - alpha, 0)
        return overlay
    except Exception:
        return None


def evaluate_model_on_dataset(model, dataset, class_names: list[str]) -> dict[str, Any]:
    if not (TF_AVAILABLE and SKLEARN_AVAILABLE):
        return {"error": "TensorFlow or scikit-learn not available."}

    y_true: list[int] = []
    y_pred: list[int] = []
    y_proba: list[list[float]] = []
    for x_batch, y_batch in dataset:
        probs = model.predict(x_batch, verbose=0)
        y_proba.extend(probs.tolist())
        y_pred.extend(np.argmax(probs, axis=1).tolist())
        y_true.extend(y_batch.numpy().tolist())

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_true, y_pred).tolist()
    lb = LabelBinarizer()
    lb.fit(range(len(class_names)))
    y_true_bin = lb.transform(y_true)
    try:
        roc_auc_macro = roc_auc_score(
            y_true_bin,
            np.array(y_proba),
            average="macro",
            multi_class="ovr",
        )
    except Exception:
        roc_auc_macro = None
    return {
        "classification_report": report,
        "confusion_matrix": cm,
        "roc_auc_macro": roc_auc_macro,
    }


# ---------------- Option A: dataset preparation ----------------
def prepare_open_dataset(
    output_root: Path,
    max_per_class: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    progress: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    legend_text = http_get_bytes(OPEN_FER_LEGEND_URL).decode("utf-8", errors="ignore")
    rows = list(csv.DictReader(io.StringIO(legend_text)))

    grouped: dict[str, list[str]] = {}
    for row in rows:
        emotion = str(row.get("emotion", "")).strip().lower()
        image_name = str(row.get("image", "")).strip()
        mapped = OPEN_DATASET_MOOD_MAP.get(emotion)
        if not mapped or not image_name:
            continue
        grouped.setdefault(mapped, []).append(image_name)

    rng = random.Random(seed)
    for k in grouped:
        dedup = list(dict.fromkeys(grouped[k]))
        rng.shuffle(dedup)
        grouped[k] = dedup[:max_per_class]

    total_target = max(1, sum(len(v) for v in grouped.values()))
    done = 0
    summary: dict[str, Any] = {"source": "open_facial_expressions", "classes": {}, "downloaded": 0, "failed": 0}

    for class_name, files in grouped.items():
        train_items, val_items, test_items = split_list(files, train_ratio, val_ratio)
        per_class = {"train": 0, "val": 0, "test": 0, "failed": 0}
        for split_name, split_items in (("train", train_items), ("val", val_items), ("test", test_items)):
            out_dir = output_root / split_name / class_name
            out_dir.mkdir(parents=True, exist_ok=True)
            for idx, filename in enumerate(split_items):
                out_path = out_dir / f"{Path(filename).stem}_{idx:05d}.jpg"
                try:
                    url = OPEN_FER_IMAGE_BASE + quote(filename)
                    bts = http_get_bytes(url, timeout=25, retries=2)
                    ok = maybe_resize_and_save_image(bts, out_path, (224, 224))
                    if ok:
                        per_class[split_name] += 1
                        summary["downloaded"] += 1
                    else:
                        per_class["failed"] += 1
                        summary["failed"] += 1
                except Exception:
                    per_class["failed"] += 1
                    summary["failed"] += 1
                done += 1
                if progress:
                    progress(min(done / total_target, 1.0))
        summary["classes"][class_name] = per_class

    return summary


def _row_get(row: dict[str, Any], candidates: list[str]) -> str:
    for key in candidates:
        if key in row and str(row[key]).strip():
            return str(row[key]).strip()
    return ""


def prepare_fer2013_csv_dataset(
    csv_bytes: bytes,
    output_root: Path,
    max_per_class: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    progress: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    if not PIL_AVAILABLE and cv2 is None:
        raise RuntimeError("PIL or OpenCV is required to write images from FER2013 CSV.")

    text = csv_bytes.decode("utf-8", errors="ignore")
    rows = list(csv.DictReader(io.StringIO(text)))
    rng = random.Random(seed)

    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        emotion_raw = _row_get(row, ["emotion", " Emotion", "label"])
        pixels = _row_get(row, ["pixels", " pixels"])
        usage = _row_get(row, ["Usage", " Usage", "usage"])
        if not emotion_raw or not pixels:
            continue
        try:
            emotion_id = int(emotion_raw)
        except ValueError:
            continue
        label = FER2013_LABELS.get(emotion_id)
        if not label:
            continue
        grouped.setdefault(label, []).append({"pixels": pixels, "usage": usage})

    for label in grouped:
        rng.shuffle(grouped[label])
        grouped[label] = grouped[label][:max_per_class]

    total = max(1, sum(len(v) for v in grouped.values()))
    processed = 0
    summary: dict[str, Any] = {"source": "fer2013_csv", "classes": {}, "written": 0, "failed": 0}

    for label, sample_rows in grouped.items():
        usage_buckets = {"train": [], "val": [], "test": []}
        without_usage = []
        for item in sample_rows:
            usage = item["usage"].strip().lower()
            if usage.startswith("train"):
                usage_buckets["train"].append(item)
            elif "public" in usage:
                usage_buckets["val"].append(item)
            elif "private" in usage or "test" in usage:
                usage_buckets["test"].append(item)
            else:
                without_usage.append(item)
        if without_usage:
            tr, va, te = split_list(without_usage, train_ratio, val_ratio)
            usage_buckets["train"].extend(tr)
            usage_buckets["val"].extend(va)
            usage_buckets["test"].extend(te)

        per_class = {"train": 0, "val": 0, "test": 0, "failed": 0}
        for split_name, bucket in usage_buckets.items():
            out_dir = output_root / split_name / label
            out_dir.mkdir(parents=True, exist_ok=True)
            for idx, item in enumerate(bucket):
                try:
                    values = [int(v) for v in item["pixels"].split()]
                    if len(values) != 48 * 48:
                        raise ValueError("invalid FER2013 pixels length")
                    arr = np.array(values, dtype=np.uint8).reshape(48, 48)
                    out_path = out_dir / f"{label}_{idx:05d}.jpg"
                    if PIL_AVAILABLE:
                        img = Image.fromarray(arr, mode="L").convert("RGB").resize((224, 224))
                        img.save(out_path, format="JPEG", quality=90)
                    else:
                        resized = cv2.resize(arr, (224, 224))
                        cv2.imwrite(str(out_path), resized)
                    per_class[split_name] += 1
                    summary["written"] += 1
                except Exception:
                    per_class["failed"] += 1
                    summary["failed"] += 1
                processed += 1
                if progress:
                    progress(min(processed / total, 1.0))
        summary["classes"][label] = per_class
    return summary


def download_fer2013_from_kaggle_to_csv_bytes() -> bytes:
    """
    Attempts dataset download using kaggle API.
    Requires kaggle package and valid credentials in ~/.kaggle/kaggle.json or env vars.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as exc:
        raise RuntimeError("kaggle package not installed. Run: pip install kaggle") from exc

    target_dir = TMP_DIR / "kaggle_fer2013"
    target_dir.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files("msambare/fer2013", path=str(target_dir), unzip=True)

    candidates: list[Path] = []
    for root, _, files in os.walk(target_dir):
        for name in files:
            lower = name.lower()
            if lower in {"fer2013.csv", "icml_face_data.csv", "train.csv"}:
                candidates.append(Path(root) / name)
    if not candidates:
        raise RuntimeError(
            "Downloaded dataset did not contain fer2013.csv/icml_face_data.csv/train.csv."
        )
    return candidates[0].read_bytes()


def extract_dataset_zip(uploaded_zip_bytes: bytes, output_root: Path) -> None:
    with zipfile.ZipFile(io.BytesIO(uploaded_zip_bytes)) as zf:
        zf.extractall(output_root)


# ---------------- Spotify + OpenAI helpers ----------------
def spotify_search_tracks(mood: str, limit: int = 3) -> list[tuple[str, str]]:
    if SPOTIFY_CLIENT is None:
        return []
    try:
        res = SPOTIFY_CLIENT.search(q=mood, limit=limit, type="track")
        return [(t["name"], t["external_urls"]["spotify"]) for t in res["tracks"]["items"]]
    except Exception:
        return []


def openai_chat(prompt: str, system_prompt: str = "You are a helpful emotional AI assistant.") -> str:
    if not prompt.strip():
        return "Please enter a message."
    if OPENAI_CLIENT is not None:
        try:
            resp = OPENAI_CLIENT.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=250,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            return f"OpenAI API error: {exc}"
    if openai is not None and OPENAI_API_KEY:
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=250,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            return f"OpenAI API error: {exc}"
    return "OpenAI not configured. Tip: set OPENAI_API_KEY."


# ---------------- UI ----------------
ensure_catalog_csv(CATALOG_FILE)
init_recommender_stats()

if "last_recommendation" not in st.session_state:
    st.session_state["last_recommendation"] = None

st.title("🧠 CEI-ALOS — Cognitive Emotional Intelligence & Adaptive Lifestyle OS")
st.markdown(
    "Single-file prototype with multi-modal emotion fusion, transfer learning, explainability, "
    "adaptive recommendations, and dataset/catalog tooling."
)

tab_analyze, tab_train, tab_catalog, tab_chat, tab_setup = st.tabs(
    [
        "Analyze & Recommend",
        "Option A: Dataset + Training",
        "Option B: Resource Catalog",
        "Chat + Analytics",
        "Setup + Distribution",
    ]
)

with tab_analyze:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("User Inputs")
        user = st.text_input("Username", value="guest_user")
        emoji = st.select_slider("Select Emoji", options=list(EMOJI_MAP.keys()))
        image_capture = st.camera_input("Capture face (optional)")
        upload_image = st.file_uploader("Or upload image", type=["png", "jpg", "jpeg"], key="face_upload")
        face_bytes = None
        if upload_image is not None:
            face_bytes = upload_image.read()
        elif image_capture is not None:
            face_bytes = image_capture.getvalue()

        voice_mode = st.checkbox("Enable microphone capture")
        if "voice_text" not in st.session_state:
            st.session_state["voice_text"] = ""
        if voice_mode:
            if st.button("Record Voice (5s)"):
                with st.spinner("Recording..."):
                    st.session_state["voice_text"] = record_voice_text(timeout=5)
        voice_text = st.text_input("Voice or typed short text", value=st.session_state["voice_text"])
        user_text = st.text_area("Context text", value="")

        run_analysis = st.button("🚀 Analyze & Recommend")

    with col2:
        st.subheader("Inference Output")
        if run_analysis:
            face_mood = "calm"
            face_label_raw = "calm"
            if face_bytes:
                used_model = False
                model_path = MODEL_DIR / "mobile_transfer_trained.keras"
                if TF_AVAILABLE and model_path.exists():
                    try:
                        model = tf.keras.models.load_model(model_path)
                        arr = load_image_for_model(face_bytes)
                        if arr is not None:
                            probs = model.predict(np.array([arr]), verbose=0)
                            idx = int(np.argmax(probs[0]))
                            class_names = []
                            if CLASS_NAMES_FILE.exists():
                                class_names = json.loads(CLASS_NAMES_FILE.read_text())
                            face_label_raw = class_names[idx] if idx < len(class_names) else f"class_{idx}"
                            face_mood = normalize_mood(face_label_raw)
                            used_model = True

                            conv_name = get_last_conv_layer_name(model)
                            if conv_name:
                                heatmap = grad_cam_heatmap(arr, model, conv_name)
                                if heatmap is not None:
                                    over = overlay_heatmap(arr.astype(np.uint8), heatmap)
                                    if over is not None:
                                        st.image(over, caption=f"Grad-CAM ({conv_name})")
                    except Exception:
                        used_model = False
                if not used_model:
                    face_mood = heuristic_face_mood_from_bytes(face_bytes)
                    face_label_raw = face_mood

            emoji_mood = normalize_mood(EMOJI_MAP.get(emoji, "calm"))
            voice_mood = simple_text_to_mood(voice_text)
            text_mood = simple_text_to_mood(user_text)

            scores = {"happy": 0.0, "sad": 0.0, "calm": 0.0, "energetic": 0.0}
            for mood, weight in [
                (normalize_mood(face_mood), 0.40),
                (emoji_mood, 0.25),
                (voice_mood, 0.20),
                (text_mood, 0.15),
            ]:
                scores[mood] = scores.get(mood, 0.0) + weight
            fused = max(scores.items(), key=lambda x: x[1])[0]

            catalog_df = load_catalog_df(CATALOG_FILE)
            rec = recommend_no_repeat(user, fused, catalog_df, last_n=3)
            st.session_state["last_recommendation"] = rec

            log_interaction(
                user=user,
                face_mood=str(face_label_raw),
                emoji_mood=emoji_mood,
                voice_mood=voice_mood,
                text_mood=text_mood,
                fused_mood=fused,
                recommended_url=rec["url"],
                recommended_title=rec["title"],
                feedback=None,
            )

            st.success(
                f"Face: {face_label_raw} | Emoji: {emoji_mood} | Voice: {voice_mood} | Text: {text_mood}"
            )
            st.info(f"Fused mood: **{fused}**")
            st.markdown(f"### Recommended: [{rec['title']}]({rec['url']})")
            if "youtube.com/watch" in rec["url"]:
                st.video(rec["url"])

            tracks = spotify_search_tracks(fused, limit=3)
            if tracks:
                st.markdown("**Spotify picks**")
                for name, url in tracks:
                    st.markdown(f"- [{name}]({url})")
            else:
                st.caption("Spotify not available or credentials missing.")

        last_rec = st.session_state.get("last_recommendation")
        if last_rec:
            c1, c2 = st.columns(2)
            with c1:
                if st.button("👍 I liked this", key="like_btn"):
                    update_stats(last_rec["url"], liked=True)
                    update_last_feedback("liked")
                    st.success("Preference recorded.")
            with c2:
                if st.button("👎 Skip / Not relevant", key="skip_btn"):
                    update_stats(last_rec["url"], liked=False)
                    update_last_feedback("skipped")
                    st.info("Feedback recorded.")

with tab_train:
    st.subheader("Option A — Automatic dataset download + pre-split")
    st.markdown(
        "Choose a source and prepare `dataset/train`, `dataset/val`, `dataset/test` automatically."
    )
    source = st.selectbox(
        "Dataset source",
        [
            "Public Facial Expressions (GitHub, no auth)",
            "FER2013 via Kaggle API (requires credentials)",
            "FER2013 CSV upload",
        ],
    )
    max_per_class = st.slider("Max samples per class", min_value=40, max_value=500, value=140, step=20)
    train_ratio = st.slider("Train ratio", min_value=0.5, max_value=0.9, value=0.7, step=0.05)
    val_ratio = st.slider("Validation ratio", min_value=0.05, max_value=0.3, value=0.15, step=0.05)
    random_seed = st.number_input("Random seed", min_value=1, max_value=99999, value=42)
    clear_existing = st.checkbox("Clear existing ./dataset before preparing", value=True)

    fer_csv_upload = None
    if source == "FER2013 CSV upload":
        fer_csv_upload = st.file_uploader("Upload FER2013 CSV", type=["csv"], key="fer_csv_upload")

    if st.button("Prepare dataset now"):
        if clear_existing:
            clear_dir(DATASET_DIR)
        else:
            DATASET_DIR.mkdir(parents=True, exist_ok=True)
        pbar = st.progress(0.0)

        def cb(v: float) -> None:
            pbar.progress(max(0.0, min(v, 1.0)))

        try:
            if source == "Public Facial Expressions (GitHub, no auth)":
                summary = prepare_open_dataset(
                    output_root=DATASET_DIR,
                    max_per_class=max_per_class,
                    train_ratio=train_ratio,
                    val_ratio=val_ratio,
                    seed=int(random_seed),
                    progress=cb,
                )
            elif source == "FER2013 via Kaggle API (requires credentials)":
                csv_bytes = download_fer2013_from_kaggle_to_csv_bytes()
                summary = prepare_fer2013_csv_dataset(
                    csv_bytes=csv_bytes,
                    output_root=DATASET_DIR,
                    max_per_class=max_per_class,
                    train_ratio=train_ratio,
                    val_ratio=val_ratio,
                    seed=int(random_seed),
                    progress=cb,
                )
            else:
                if fer_csv_upload is None:
                    raise RuntimeError("Please upload a FER2013 CSV first.")
                summary = prepare_fer2013_csv_dataset(
                    csv_bytes=fer_csv_upload.read(),
                    output_root=DATASET_DIR,
                    max_per_class=max_per_class,
                    train_ratio=train_ratio,
                    val_ratio=val_ratio,
                    seed=int(random_seed),
                    progress=cb,
                )
            pbar.progress(1.0)
            st.success("Dataset prepared.")
            st.json(summary)
        except Exception as exc:
            st.error(f"Dataset preparation error: {exc}")

    st.markdown("---")
    st.markdown("Optional: upload a pre-structured ZIP containing `train/val[/test]` class folders.")
    dataset_zip = st.file_uploader("Upload dataset ZIP", type=["zip"], key="dataset_zip_upload")
    if dataset_zip and st.button("Extract uploaded ZIP to ./dataset"):
        if clear_existing:
            clear_dir(DATASET_DIR)
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        extract_dataset_zip(dataset_zip.read(), DATASET_DIR)
        st.success("ZIP extracted to ./dataset")

    st.markdown("---")
    st.subheader("Transfer Learning (MobileNetV2)")
    if TF_AVAILABLE:
        if st.button("Build & save untrained model"):
            try:
                n_classes = st.session_state.get("n_classes_ui", 3)
                model = build_transfer_model(int(n_classes))
                if model is None:
                    raise RuntimeError("Model could not be created.")
                model.save(MODEL_DIR / "mobile_transfer.keras")
                st.success("Saved model to models/mobile_transfer.keras")
            except Exception as exc:
                st.error(f"Build error: {exc}")

        n_classes_ui = st.number_input("Expected classes (for build only)", min_value=2, max_value=12, value=3, key="n_classes_ui")
        batch_size = st.number_input("Batch size", min_value=4, max_value=64, value=16)
        epochs = st.number_input("Epochs (demo)", min_value=1, max_value=20, value=3)

        if st.button("Train model on ./dataset/train + ./dataset/val"):
            train_dir = DATASET_DIR / "train"
            val_dir = DATASET_DIR / "val"
            if not train_dir.exists() or not val_dir.exists():
                st.error("Missing dataset/train or dataset/val. Run dataset preparation first.")
            else:
                try:
                    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
                        train_dir, image_size=(224, 224), batch_size=int(batch_size)
                    )
                    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
                        val_dir, image_size=(224, 224), batch_size=int(batch_size)
                    )
                    class_names = list(train_ds.class_names)
                    model = build_transfer_model(len(class_names))
                    if model is None:
                        raise RuntimeError("Model build failed.")

                    aug = models.Sequential([layers.RandomFlip("horizontal"), layers.RandomRotation(0.1)])
                    autotune = tf.data.AUTOTUNE
                    train_aug = train_ds.map(lambda x, y: (aug(x, training=True), y)).prefetch(autotune)
                    val_prefetch = val_ds.prefetch(autotune)

                    with st.spinner("Training..."):
                        model.fit(train_aug, validation_data=val_prefetch, epochs=int(epochs), verbose=1)
                    model.save(MODEL_DIR / "mobile_transfer_trained.keras")
                    CLASS_NAMES_FILE.write_text(json.dumps(class_names))
                    st.success("Training complete. Saved model + class names.")

                    if SKLEARN_AVAILABLE:
                        metrics = evaluate_model_on_dataset(model, val_prefetch, class_names)
                        st.markdown("#### Validation metrics")
                        st.json(metrics)
                except Exception as exc:
                    st.error(f"Training error: {exc}")

        if st.button("Evaluate saved trained model"):
            model_path = MODEL_DIR / "mobile_transfer_trained.keras"
            eval_dir = DATASET_DIR / "test"
            if not model_path.exists():
                st.error("No trained model found. Train first.")
            elif not eval_dir.exists():
                st.error("No dataset/test found. Prepare dataset with test split first.")
            else:
                try:
                    model = tf.keras.models.load_model(model_path)
                    class_names = json.loads(CLASS_NAMES_FILE.read_text()) if CLASS_NAMES_FILE.exists() else None
                    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
                        eval_dir, image_size=(224, 224), batch_size=16
                    )
                    if class_names is None:
                        class_names = list(test_ds.class_names)
                    metrics = evaluate_model_on_dataset(model, test_ds, class_names)
                    st.json(metrics)
                except Exception as exc:
                    st.error(f"Evaluation error: {exc}")
    else:
        st.warning("TensorFlow not available. Install tensorflow for training and Grad-CAM.")

with tab_catalog:
    st.subheader("Option B — Resource catalog (100+ entries) with CSV export/import")
    df_catalog = load_catalog_df(CATALOG_FILE)
    st.info(
        f"Catalog entries: {len(df_catalog)} | Categories: {', '.join(sorted(df_catalog['Category'].unique()))}"
    )
    if len(df_catalog) < 100:
        st.warning("Catalog has less than 100 entries; regenerate default to restore 100+ items.")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Regenerate default 100+ catalog"):
            pd.DataFrame(build_default_catalog_entries()).to_csv(CATALOG_FILE, index=False)
            st.success("Default catalog regenerated.")
            st.rerun()
    with col_b:
        st.download_button(
            "Download catalog CSV",
            data=load_catalog_df(CATALOG_FILE).to_csv(index=False).encode("utf-8"),
            file_name="resource_catalog.csv",
            mime="text/csv",
        )

    st.markdown("Upload your own catalog CSV (`Category,Title,URL,Type,Tags`).")
    catalog_upload = st.file_uploader("Upload catalog CSV", type=["csv"], key="catalog_upload")
    merge_uploaded = st.checkbox("Merge uploaded catalog with existing", value=True)
    if st.button("Apply uploaded catalog"):
        if catalog_upload is None:
            st.error("Please upload a CSV first.")
        else:
            try:
                incoming = pd.read_csv(catalog_upload)
                needed = {"Category", "Title", "URL"}
                if not needed.issubset(set(incoming.columns)):
                    raise RuntimeError("CSV must include at least Category, Title, URL columns.")
                for col in ["Type", "Tags"]:
                    if col not in incoming.columns:
                        incoming[col] = ""
                incoming = incoming[["Category", "Title", "URL", "Type", "Tags"]]
                if merge_uploaded:
                    combined = pd.concat([load_catalog_df(CATALOG_FILE), incoming], ignore_index=True)
                    combined.drop_duplicates(subset=["Category", "Title", "URL"], inplace=True)
                    combined.to_csv(CATALOG_FILE, index=False)
                else:
                    incoming.to_csv(CATALOG_FILE, index=False)
                st.success("Catalog updated.")
                st.rerun()
            except Exception as exc:
                st.error(f"Catalog update error: {exc}")

    st.markdown("---")
    display_df = load_catalog_df(CATALOG_FILE)
    for category in sorted(display_df["Category"].unique()):
        with st.expander(f"{category} ({int((display_df['Category'] == category).sum())})"):
            subset = display_df[display_df["Category"] == category].head(25)
            for _, row in subset.iterrows():
                st.markdown(f"- [{row['Title']}]({row['URL']})")
            if (display_df["Category"] == category).sum() > 25:
                st.caption("Showing first 25 entries for this category.")

with tab_chat:
    st.subheader("Chat with Emotional AI")
    chat_input = st.text_input("Say something", "")
    if st.button("Send Chat"):
        st.write(openai_chat(chat_input))

    st.markdown("---")
    st.subheader("Digital Emotional Twin Dashboard")
    if st.button("Show logs"):
        logs = read_csv_safe(
            DATA_FILE,
            [
                "User",
                "TimeUTC",
                "Face",
                "Emoji",
                "Voice",
                "Text",
                "Fused",
                "RecommendedTitle",
                "RecommendedURL",
                "Feedback",
            ],
        )
        if logs.empty:
            st.info("No logs yet.")
        else:
            st.dataframe(logs)
            st.bar_chart(logs["Fused"].value_counts())

    if st.button("Show recommender stats"):
        stats = read_csv_safe(RL_STATS_FILE, ["Item", "Exposures", "Likes"])
        if stats.empty:
            st.info("No recommender stats yet.")
        else:
            stats["score"] = stats["Likes"] / (stats["Exposures"] + 1e-6)
            st.dataframe(stats.sort_values("score", ascending=False))
            st.bar_chart(stats.set_index("Item")["score"])

with tab_setup:
    st.subheader("Developer setup")
    st.markdown(
        """
1) Install dependencies:
   - `pip install streamlit numpy pandas pillow opencv-python-headless tensorflow scikit-learn speechrecognition spotipy openai kaggle`

2) OpenAI setup:
   - `export OPENAI_API_KEY="your_key"`

3) Spotify setup:
   - `export SPOTIPY_CLIENT_ID="your_id"`
   - `export SPOTIPY_CLIENT_SECRET="your_secret"`

4) Optional Kaggle setup for FER2013 auto-download:
   - Install `kaggle` package and configure credentials (`~/.kaggle/kaggle.json`)
   - Or set `KAGGLE_USERNAME` and `KAGGLE_KEY`.

5) Run app:
   - `streamlit run app.py`
"""
    )
    st.subheader("Android / Windows installable experience (no paid wrappers)")
    st.markdown(
        """
- Host this Streamlit app (Streamlit Cloud / Render / Railway / VM).
- Android Chrome: open URL -> **Add to Home screen**.
- Windows Edge/Chrome: open URL -> **Install app**.

This gives an installable PWA-like experience without paid APK wrapper services.
For a true native APK, use Kivy/Buildozer or Chaquopy (extra build toolchain required).
"""
    )
    st.subheader("Notes")
    st.markdown(
        """
- Full training + Grad-CAM requires TensorFlow.
- FER2013 Kaggle route may require accepted terms and authenticated API credentials.
- For production, replace CSV logs with a database and move API usage server-side.
"""
    )

st.caption(
    "CEI-ALOS prototype: logs/models/catalog are persisted as local files in the project directory."
)
