"""CEI-ALOS single-file Streamlit prototype.

This app bundles:
- Multi-modal mood intake (face, emoji, voice transcript, text)
- Transfer-learning training/evaluation with MobileNetV2
- Automatic public dataset download + train/val folder export
- Grad-CAM explainability
- History-aware no-repetition recommender
- Spotify + OpenAI hooks
- CSV-based Digital Emotional Twin logging
- A structured 100+ item resource catalog with CSV export

Run:
    streamlit run app.py
"""

from __future__ import annotations

import io
import json
import os
import random
import re
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from urllib.parse import quote, quote_plus
from uuid import uuid4

import numpy as np
import pandas as pd
import streamlit as st

try:
    from PIL import Image

    PIL_AVAILABLE = True
except Exception:
    Image = None
    PIL_AVAILABLE = False

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
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        precision_recall_fscore_support,
        roc_auc_score,
    )
    from sklearn.preprocessing import LabelBinarizer

    SKLEARN_AVAILABLE = True
except Exception:
    classification_report = None
    confusion_matrix = None
    precision_recall_fscore_support = None
    roc_auc_score = None
    LabelBinarizer = None
    SKLEARN_AVAILABLE = False

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
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    import openai as legacy_openai
except Exception:
    legacy_openai = None

try:
    from datasets import load_dataset

    DATASETS_AVAILABLE = True
except Exception:
    load_dataset = None
    DATASETS_AVAILABLE = False


APP_ROOT = Path(__file__).resolve().parent
DATASET_BASE_DIR = APP_ROOT / "dataset"
MODEL_DIR = APP_ROOT / "models"

DATA_FILE = APP_ROOT / "cei_twin_log.csv"
RL_STATS_FILE = APP_ROOT / "recommender_stats.csv"
CATALOG_CSV_FILE = APP_ROOT / "resource_catalog.csv"

MODEL_PATH = MODEL_DIR / "mobile_transfer_trained.keras"
LABELS_PATH = MODEL_DIR / "mobile_transfer_labels.json"
DATASET_INFO_PATH = MODEL_DIR / "last_dataset_info.json"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
SPOTIPY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID", "").strip()
SPOTIPY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET", "").strip()

LOG_COLUMNS = [
    "LogId",
    "User",
    "Time",
    "FaceLabel",
    "FaceMood",
    "EmojiMood",
    "VoiceMood",
    "TextMood",
    "FusedMood",
    "RecommendedItemId",
    "RecommendedTitle",
    "RecommendedKind",
    "RecommendedURL",
    "Feedback",
]

STATS_COLUMNS = [
    "ItemId",
    "Title",
    "Mood",
    "Kind",
    "Exposures",
    "Likes",
    "Skips",
    "LastServedAt",
    "LastFeedbackAt",
]

EMOJI_MAP = {
    "😊": "happy",
    "😍": "happy",
    "😢": "sad",
    "😞": "sad",
    "😌": "calm",
    "😴": "calm",
    "😡": "energetic",
    "🤩": "energetic",
}

NORMALIZED_MOOD_MAP = {
    "happy": "happy",
    "joy": "happy",
    "love": "happy",
    "excited": "happy",
    "surprise": "energetic",
    "angry": "energetic",
    "anger": "energetic",
    "irritated": "energetic",
    "mad": "energetic",
    "neutral": "calm",
    "calm": "calm",
    "relaxed": "calm",
    "peaceful": "calm",
    "sleepy": "calm",
    "sad": "sad",
    "fear": "sad",
    "afraid": "sad",
    "disgust": "sad",
    "depressed": "sad",
}

PUBLIC_DATASET_PRESETS = {
    "FER2013 Cleaned (public Hugging Face)": "mehmet-3emin/fer2013-cleaned",
    "Teen-Different / Facial-Expression (experimental public Hugging Face)": "Teen-Different/Facial-Expression",
    "Custom public Hugging Face image dataset": "__custom__",
}

SOUNDHELIX_AUDIO_URLS = [
    f"https://www.soundhelix.com/examples/mp3/SoundHelix-Song-{index}.mp3"
    for index in range(1, 13)
]

RESOURCE_QUERY_LIBRARY = {
    "HAPPY": {
        "youtube": [
            "feel good pop playlist",
            "happy dance workout at home",
            "gratitude meditation morning",
            "positive affirmations for joy",
            "upbeat acoustic session",
            "sunshine road trip songs",
            "laughter yoga exercise",
            "quick mood lift breathing",
            "celebration songs mix",
            "dopamine menu ideas",
            "confidence boost visualization",
            "happy chill instrumental",
        ],
        "spotify": [
            "happy hits",
            "good vibes mix",
            "feel good friday",
            "dance pop energy",
            "summer upbeat playlist",
            "acoustic happy songs",
            "morning motivation music",
            "sunshine indie playlist",
            "cheerful piano mix",
            "boost your mood",
        ],
        "audio": [
            ("Uplifting Instrumental Session 1", SOUNDHELIX_AUDIO_URLS[0]),
            ("Uplifting Instrumental Session 2", SOUNDHELIX_AUDIO_URLS[1]),
            ("Uplifting Instrumental Session 3", SOUNDHELIX_AUDIO_URLS[2]),
            ("Bright Reset Track 1", SOUNDHELIX_AUDIO_URLS[3]),
            ("Bright Reset Track 2", SOUNDHELIX_AUDIO_URLS[4]),
        ],
    },
    "SAD": {
        "youtube": [
            "comforting acoustic songs",
            "self compassion meditation",
            "gentle breathing for sadness",
            "grief support journaling",
            "calming piano when sad",
            "emotional release stretching",
            "rain sounds relaxation",
            "soothing bedtime stories adults",
            "healing after a hard day",
            "guided meditation for sadness",
            "slow lo fi comfort mix",
            "gentle nature walk sounds",
        ],
        "spotify": [
            "sad songs but comforting",
            "acoustic chill playlist",
            "healing piano playlist",
            "calm rainy day music",
            "emotional release music",
            "sleep and recovery music",
            "deep focus calm",
            "mellow indie comfort",
            "self care sunday music",
            "soft instrumental healing",
        ],
        "audio": [
            ("Gentle Recovery Track 1", SOUNDHELIX_AUDIO_URLS[5]),
            ("Gentle Recovery Track 2", SOUNDHELIX_AUDIO_URLS[6]),
            ("Gentle Recovery Track 3", SOUNDHELIX_AUDIO_URLS[7]),
            ("Soft Reflection Session 1", SOUNDHELIX_AUDIO_URLS[0]),
            ("Soft Reflection Session 2", SOUNDHELIX_AUDIO_URLS[1]),
        ],
    },
    "CALM": {
        "youtube": [
            "box breathing exercise",
            "mindfulness meditation beginners",
            "ocean waves relaxation",
            "lo fi calm study mix",
            "gentle yoga unwind",
            "progressive muscle relaxation",
            "rain sounds sleep",
            "forest ambience meditation",
            "soothing instrumental piano",
            "stress relief breathing 5 minutes",
            "mindful journaling prompts",
            "sound bath relaxation",
        ],
        "spotify": [
            "calm vibes playlist",
            "peaceful piano",
            "nature sounds relaxation",
            "ambient focus music",
            "meditation music",
            "deep sleep music",
            "spa chill playlist",
            "lo fi calm beats",
            "soft rain sounds",
            "evening unwind mix",
        ],
        "audio": [
            ("Ambient Reset Track 1", SOUNDHELIX_AUDIO_URLS[2]),
            ("Ambient Reset Track 2", SOUNDHELIX_AUDIO_URLS[3]),
            ("Ambient Reset Track 3", SOUNDHELIX_AUDIO_URLS[4]),
            ("Quiet Focus Session 1", SOUNDHELIX_AUDIO_URLS[5]),
            ("Quiet Focus Session 2", SOUNDHELIX_AUDIO_URLS[6]),
        ],
    },
    "ENERGETIC": {
        "youtube": [
            "hiit cardio music mix",
            "power workout playlist",
            "confidence pump up songs",
            "morning energy routine",
            "focus sprint music",
            "motivational speech workout",
            "dance cardio at home",
            "rock gym mix",
            "fast lo fi coding music",
            "energetic stretch activation",
            "pre exam motivation music",
            "upbeat instrumental training",
        ],
        "spotify": [
            "beast mode workout",
            "energy booster playlist",
            "run faster music",
            "gym motivation mix",
            "pump up songs",
            "coding focus energy",
            "power rock playlist",
            "cardio dance hits",
            "high intensity training music",
            "motivation rap playlist",
        ],
        "audio": [
            ("Activation Instrumental 1", SOUNDHELIX_AUDIO_URLS[7]),
            ("Activation Instrumental 2", SOUNDHELIX_AUDIO_URLS[8]),
            ("Activation Instrumental 3", SOUNDHELIX_AUDIO_URLS[9]),
            ("Sprint Session 1", SOUNDHELIX_AUDIO_URLS[10]),
            ("Sprint Session 2", SOUNDHELIX_AUDIO_URLS[11]),
        ],
    },
}


def ensure_runtime_dirs() -> None:
    DATASET_BASE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def ensure_csv(path: Path, columns: list[str]) -> None:
    if not path.exists():
        pd.DataFrame(columns=columns).to_csv(path, index=False)


def read_csv_or_empty(path: Path, columns: list[str]) -> pd.DataFrame:
    ensure_csv(path, columns)
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=columns)


def initialize_runtime_files() -> None:
    ensure_runtime_dirs()
    ensure_csv(DATA_FILE, LOG_COLUMNS)
    ensure_csv(RL_STATS_FILE, STATS_COLUMNS)


def init_session_state() -> None:
    defaults = {
        "voice_text": "",
        "last_analysis": None,
        "last_recommendation": None,
        "last_ranked_choices": [],
        "last_log_id": None,
        "active_dataset_root": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-")
    return cleaned or "item"


def safe_link_button(label: str, url: str) -> None:
    if hasattr(st, "link_button"):
        st.link_button(label, url)
    else:
        st.markdown(f"[{label}]({url})")


@st.cache_data(show_spinner=False)
def build_resource_catalog() -> pd.DataFrame:
    items: list[dict[str, str]] = []

    for mood, source_map in RESOURCE_QUERY_LIBRARY.items():
        for index, query in enumerate(source_map["youtube"], start=1):
            items.append(
                {
                    "ItemId": f"{slugify(mood)}-yt-{index:02d}",
                    "Mood": mood,
                    "Kind": "youtube_search",
                    "Source": "YouTube",
                    "Title": f"{mood.title()} | {query.title()}",
                    "Query": query,
                    "URL": f"https://www.youtube.com/results?search_query={quote_plus(query)}",
                    "Tags": f"{mood.lower()}, youtube, guided",
                }
            )

        for index, query in enumerate(source_map["spotify"], start=1):
            items.append(
                {
                    "ItemId": f"{slugify(mood)}-sp-{index:02d}",
                    "Mood": mood,
                    "Kind": "spotify_search",
                    "Source": "Spotify",
                    "Title": f"{mood.title()} | {query.title()}",
                    "Query": query,
                    "URL": f"https://open.spotify.com/search/{quote(query, safe='')}",
                    "Tags": f"{mood.lower()}, spotify, playlist",
                }
            )

        for index, (title, url) in enumerate(source_map["audio"], start=1):
            items.append(
                {
                    "ItemId": f"{slugify(mood)}-audio-{index:02d}",
                    "Mood": mood,
                    "Kind": "direct_audio",
                    "Source": "Direct audio",
                    "Title": f"{mood.title()} | {title}",
                    "Query": title.lower(),
                    "URL": url,
                    "Tags": f"{mood.lower()}, audio, fallback",
                }
            )

    return pd.DataFrame(items)


def ensure_catalog_export(catalog_df: pd.DataFrame) -> None:
    catalog_df.to_csv(CATALOG_CSV_FILE, index=False)


def load_logs() -> pd.DataFrame:
    return read_csv_or_empty(DATA_FILE, LOG_COLUMNS)


def save_logs(df: pd.DataFrame) -> None:
    df.to_csv(DATA_FILE, index=False)


def load_recommender_stats() -> pd.DataFrame:
    return read_csv_or_empty(RL_STATS_FILE, STATS_COLUMNS)


def save_recommender_stats(df: pd.DataFrame) -> None:
    df.to_csv(RL_STATS_FILE, index=False)


def log_interaction(
    user: str,
    face_label: str,
    face_mood: str,
    emoji_mood: str,
    voice_mood: str,
    text_mood: str,
    fused_mood: str,
    recommended_item: dict[str, str],
) -> str:
    log_id = str(uuid4())
    entry = pd.DataFrame(
        [
            {
                "LogId": log_id,
                "User": user,
                "Time": datetime.utcnow().isoformat(),
                "FaceLabel": face_label,
                "FaceMood": face_mood,
                "EmojiMood": emoji_mood,
                "VoiceMood": voice_mood,
                "TextMood": text_mood,
                "FusedMood": fused_mood,
                "RecommendedItemId": recommended_item["ItemId"],
                "RecommendedTitle": recommended_item["Title"],
                "RecommendedKind": recommended_item["Kind"],
                "RecommendedURL": recommended_item["URL"],
                "Feedback": "",
            }
        ]
    )
    logs = load_logs()
    logs = pd.concat([logs, entry], ignore_index=True)
    save_logs(logs)
    return log_id


def update_log_feedback(log_id: str | None, feedback: str) -> None:
    if not log_id:
        return
    logs = load_logs()
    if logs.empty:
        return
    mask = logs["LogId"] == log_id
    if mask.any():
        logs.loc[mask, "Feedback"] = feedback
        save_logs(logs)


def record_recommendation(item: dict[str, str]) -> None:
    stats = load_recommender_stats()
    mask = stats["ItemId"] == item["ItemId"]

    if mask.any():
        stats.loc[mask, "Exposures"] = stats.loc[mask, "Exposures"].fillna(0).astype(int) + 1
        stats.loc[mask, "Title"] = item["Title"]
        stats.loc[mask, "Mood"] = item["Mood"]
        stats.loc[mask, "Kind"] = item["Kind"]
        stats.loc[mask, "LastServedAt"] = datetime.utcnow().isoformat()
    else:
        new_row = pd.DataFrame(
            [
                {
                    "ItemId": item["ItemId"],
                    "Title": item["Title"],
                    "Mood": item["Mood"],
                    "Kind": item["Kind"],
                    "Exposures": 1,
                    "Likes": 0,
                    "Skips": 0,
                    "LastServedAt": datetime.utcnow().isoformat(),
                    "LastFeedbackAt": "",
                }
            ]
        )
        stats = pd.concat([stats, new_row], ignore_index=True)

    save_recommender_stats(stats)


def record_feedback(item_id: str, liked: bool) -> None:
    stats = load_recommender_stats()
    mask = stats["ItemId"] == item_id
    if not mask.any():
        return

    if liked:
        stats.loc[mask, "Likes"] = stats.loc[mask, "Likes"].fillna(0).astype(int) + 1
    else:
        stats.loc[mask, "Skips"] = stats.loc[mask, "Skips"].fillna(0).astype(int) + 1

    stats.loc[mask, "LastFeedbackAt"] = datetime.utcnow().isoformat()
    save_recommender_stats(stats)


def get_recent_recommendation_ids(user: str, last_n: int = 5) -> list[str]:
    logs = load_logs()
    if logs.empty:
        return []
    user_history = logs[logs["User"] == user]
    if user_history.empty:
        return []
    return user_history["RecommendedItemId"].dropna().astype(str).tolist()[-last_n:]


def recommend_resource(
    user: str,
    mood: str,
    catalog_df: pd.DataFrame,
    last_n: int = 5,
) -> tuple[dict[str, str], list[dict[str, str]]]:
    candidates = catalog_df[catalog_df["Mood"] == mood.upper()].copy()
    if candidates.empty:
        candidates = catalog_df.copy()

    stats = load_recommender_stats()
    stats_lookup = {
        row["ItemId"]: row
        for _, row in stats.iterrows()
    }
    recent_ids = set(get_recent_recommendation_ids(user, last_n=last_n))

    scored_rows: list[tuple[float, float, dict[str, str]]] = []
    for _, row in candidates.iterrows():
        item = row.to_dict()
        item_id = item["ItemId"]
        stat_row = stats_lookup.get(item_id)

        exposures = int(stat_row["Exposures"]) if stat_row is not None and pd.notna(stat_row["Exposures"]) else 0
        likes = int(stat_row["Likes"]) if stat_row is not None and pd.notna(stat_row["Likes"]) else 0
        skips = int(stat_row["Skips"]) if stat_row is not None and pd.notna(stat_row["Skips"]) else 0

        novelty_score = 1.0 / (1 + exposures)
        feedback_score = ((likes - 0.75 * skips) / exposures) if exposures else 0.15
        repeat_penalty = 1.5 if item_id in recent_ids else 0.0
        kind_bonus = 0.05 if item["Kind"] == "direct_audio" and mood.upper() == "CALM" else 0.0

        score = 0.55 * novelty_score + 0.45 * (0.5 + feedback_score) + kind_bonus - repeat_penalty
        scored_rows.append((score, random.random(), item))

    scored_rows.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
    top_choices = [entry[2] for entry in scored_rows[:5]] or [candidates.iloc[0].to_dict()]
    chosen_item = random.choice(top_choices)

    record_recommendation(chosen_item)
    return chosen_item, top_choices


def simple_text_to_mood(text: str) -> str:
    if not text:
        return "calm"

    lowered = text.lower()
    if any(token in lowered for token in ["sad", "unhappy", "depressed", "cry", "lonely", "grief", "hurt"]):
        return "sad"
    if any(token in lowered for token in ["happy", "joy", "excited", "love", "grateful", "great", "amazing"]):
        return "happy"
    if any(token in lowered for token in ["angry", "mad", "irritat", "furious", "energized", "pumped"]):
        return "energetic"
    if any(token in lowered for token in ["calm", "peace", "relax", "okay", "fine", "tired", "sleepy"]):
        return "calm"
    return "calm"


def normalize_emotion_label(label: str) -> str:
    lowered = str(label).strip().lower()
    for token, mood in NORMALIZED_MOOD_MAP.items():
        if token in lowered:
            return mood
    return "calm"


def heuristic_face_mood_from_image(image_bytes: bytes | None) -> str:
    if cv2 is None or image_bytes is None:
        return "calm"

    arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return "calm"

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    if brightness > 150 and contrast > 35:
        return "happy"
    if brightness < 85:
        return "sad"
    if contrast > 60:
        return "energetic"
    return "calm"


def transcribe_audio_file(audio_input) -> str:
    if sr is None or audio_input is None:
        return ""

    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(io.BytesIO(audio_input.getvalue())) as source:
            audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data).lower()
    except Exception:
        return ""


def summarize_model(model) -> str:
    lines: list[str] = []
    model.summary(print_fn=lines.append)
    return "\n".join(lines)


def save_class_names(class_names: list[str]) -> None:
    LABELS_PATH.write_text(json.dumps(class_names, indent=2), encoding="utf-8")


def load_class_names() -> list[str]:
    if not LABELS_PATH.exists():
        return []
    try:
        return json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def load_image_for_model(image_bytes: bytes) -> np.ndarray | None:
    if not PIL_AVAILABLE:
        return None

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
        return np.array(image, dtype=np.float32)
    except Exception:
        return None


def build_transfer_model(num_classes: int):
    if not TF_AVAILABLE:
        return None

    try:
        base_model = tf.keras.applications.MobileNetV2(
            weights="imagenet",
            include_top=False,
            input_shape=(224, 224, 3),
        )
    except Exception:
        base_model = tf.keras.applications.MobileNetV2(
            weights=None,
            include_top=False,
            input_shape=(224, 224, 3),
        )

    base_model.trainable = False

    inputs = layers.Input(shape=(224, 224, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="cei_alos_mobilenetv2")
    model.base_model = base_model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


@st.cache_resource(show_spinner=False)
def load_trained_model_cached(model_path: str):
    if not TF_AVAILABLE:
        return None
    return tf.keras.models.load_model(model_path)


def find_last_conv_layer_name(model) -> str | None:
    for layer in reversed(model.layers):
        try:
            if len(layer.output.shape) == 4:
                return layer.name
        except Exception:
            continue
    return None


def grad_cam_heatmap(image_array: np.ndarray, model, last_conv_layer_name: str | None = None):
    if not TF_AVAILABLE:
        return None

    last_conv_layer_name = last_conv_layer_name or find_last_conv_layer_name(model)
    if not last_conv_layer_name:
        return None

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output],
    )

    image_batch = np.expand_dims(image_array, axis=0)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_batch)
        predicted_index = tf.argmax(predictions[0])
        class_channel = predictions[:, predicted_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    if tf.math.reduce_max(heatmap) > 0:
        heatmap /= tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def overlay_heatmap(image_array: np.ndarray, heatmap):
    if cv2 is None or heatmap is None:
        return None

    base = np.clip(image_array, 0, 255).astype(np.uint8)
    resized_heatmap = cv2.resize(heatmap, (base.shape[1], base.shape[0]))
    colored = cv2.applyColorMap(np.uint8(255 * resized_heatmap), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(base, 0.65, colored, 0.35, 0)


def evaluate_model_on_dataset(model, dataset, class_names: list[str]) -> dict[str, object]:
    if not SKLEARN_AVAILABLE:
        return {"error": "scikit-learn is not installed."}

    y_true: list[int] = []
    y_pred: list[int] = []
    y_proba: list[list[float]] = []

    for x_batch, y_batch in dataset:
        predictions = model.predict(x_batch, verbose=0)
        y_proba.extend(predictions.tolist())
        y_pred.extend(np.argmax(predictions, axis=1).tolist())
        y_true.extend(y_batch.numpy().tolist())

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred).tolist()

    auc_score = None
    try:
        lb = LabelBinarizer()
        lb.fit(range(len(class_names)))
        y_true_bin = lb.transform(y_true)
        y_proba_array = np.array(y_proba)
        if y_true_bin.ndim == 1:
            y_true_bin = np.column_stack([1 - y_true_bin, y_true_bin])
        auc_score = roc_auc_score(
            y_true_bin,
            y_proba_array,
            average="macro",
            multi_class="ovr",
        )
    except Exception:
        auc_score = None

    return {
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "roc_auc_macro": float(auc_score) if auc_score is not None else None,
        "classification_report": report,
        "confusion_matrix": matrix,
    }


def get_validation_dir(dataset_root: Path) -> Path | None:
    for candidate_name in ("val", "validation", "test"):
        candidate = dataset_root / candidate_name
        if candidate.exists():
            return candidate
    return None


def train_model_on_directory(
    dataset_root: Path,
    epochs: int,
    batch_size: int,
    fine_tune: bool,
) -> dict[str, object]:
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is not installed.")

    train_dir = dataset_root / "train"
    val_dir = get_validation_dir(dataset_root)

    if not train_dir.exists() or val_dir is None:
        raise FileNotFoundError("Expected dataset/train and dataset/val (or dataset/test).")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(224, 224),
        batch_size=batch_size,
        label_mode="int",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=(224, 224),
        batch_size=batch_size,
        label_mode="int",
    )

    class_names = list(train_ds.class_names)
    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.08),
        ],
        name="augmentation",
    )
    autotune = tf.data.AUTOTUNE

    train_ds_augmented = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=autotune,
    ).prefetch(autotune)
    val_ds_prefetched = val_ds.prefetch(autotune)

    model = build_transfer_model(len(class_names))
    history = model.fit(
        train_ds_augmented,
        validation_data=val_ds_prefetched,
        epochs=epochs,
        verbose=0,
    )

    if fine_tune:
        base_model = getattr(model, "base_model", None)
        if base_model is not None:
            base_model.trainable = True
            for layer in base_model.layers[:-20]:
                layer.trainable = False
            model.compile(
                optimizer=tf.keras.optimizers.Adam(1e-5),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            fine_tune_history = model.fit(
                train_ds_augmented,
                validation_data=val_ds_prefetched,
                epochs=max(1, min(2, epochs)),
                verbose=0,
            )
            for key, values in fine_tune_history.history.items():
                history.history.setdefault(key, []).extend(values)

    model.save(MODEL_PATH)
    save_class_names(class_names)
    DATASET_INFO_PATH.write_text(
        json.dumps(
            {
                "dataset_root": str(dataset_root),
                "class_names": class_names,
                "saved_at": datetime.utcnow().isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    load_trained_model_cached.clear()

    evaluation = evaluate_model_on_dataset(model, val_ds_prefetched, class_names)
    return {
        "class_names": class_names,
        "history": history.history,
        "evaluation": evaluation,
    }


def predict_face_with_best_available_method(image_bytes: bytes | None):
    if image_bytes is None:
        return "calm", "calm", None, None

    if TF_AVAILABLE and MODEL_PATH.exists():
        try:
            model = load_trained_model_cached(str(MODEL_PATH))
            image_array = load_image_for_model(image_bytes)
            if image_array is not None:
                predictions = model.predict(np.expand_dims(image_array, axis=0), verbose=0)[0]
                class_names = load_class_names()
                best_index = int(np.argmax(predictions))
                raw_label = class_names[best_index] if best_index < len(class_names) else f"class_{best_index}"
                normalized = normalize_emotion_label(raw_label)
                heatmap = grad_cam_heatmap(image_array, model)
                overlay = overlay_heatmap(image_array, heatmap)
                return raw_label, normalized, predictions.tolist(), overlay
        except Exception:
            pass

    heuristic_label = heuristic_face_mood_from_image(image_bytes)
    return heuristic_label, heuristic_label, None, None


def detect_image_column(dataset_split) -> str | None:
    candidate_names = ["image", "img", "pixel_values"]
    for candidate in candidate_names:
        if candidate in dataset_split.column_names:
            return candidate

    for column_name, feature in dataset_split.features.items():
        if "image" in feature.__class__.__name__.lower():
            return column_name
    return None


def detect_label_column(dataset_split, image_column: str | None) -> str | None:
    candidate_names = ["label", "emotion", "labels", "class", "category", "target"]
    for candidate in candidate_names:
        if candidate in dataset_split.column_names and candidate != image_column:
            return candidate

    for column_name, feature in dataset_split.features.items():
        if column_name == image_column:
            continue
        feature_name = feature.__class__.__name__.lower()
        if "classlabel" in feature_name:
            return column_name

    for column_name in dataset_split.column_names:
        if column_name != image_column:
            return column_name
    return None


def hf_image_to_pil(image_value):
    if not PIL_AVAILABLE:
        raise RuntimeError("Pillow is not installed.")

    if hasattr(image_value, "convert"):
        return image_value.convert("RGB")
    if isinstance(image_value, (bytes, bytearray)):
        return Image.open(io.BytesIO(image_value)).convert("RGB")
    if isinstance(image_value, str):
        return Image.open(image_value).convert("RGB")
    if isinstance(image_value, dict):
        if image_value.get("bytes"):
            return Image.open(io.BytesIO(image_value["bytes"])).convert("RGB")
        if image_value.get("path"):
            return Image.open(image_value["path"]).convert("RGB")
    raise ValueError("Unsupported image format in dataset row.")


def resolve_label_name(label_value, label_feature) -> str:
    if isinstance(label_value, list):
        if len(label_value) != 1:
            raise ValueError("Multi-label datasets are not supported by the CNN training flow.")
        label_value = label_value[0]

    if hasattr(label_feature, "names") and isinstance(label_value, (int, np.integer)):
        return str(label_feature.names[int(label_value)])
    return str(label_value)


def prepare_public_image_dataset(
    dataset_id: str,
    output_dir: Path,
    max_samples_per_class: int,
    image_size: int,
    seed: int,
    validation_ratio: float,
) -> dict[str, object]:
    if not DATASETS_AVAILABLE:
        raise RuntimeError("The datasets package is not installed.")
    if not PIL_AVAILABLE:
        raise RuntimeError("Pillow is required for dataset preparation.")

    dataset_bundle = load_dataset(dataset_id)

    if hasattr(dataset_bundle, "keys"):
        split_names = list(dataset_bundle.keys())
        reference_split = dataset_bundle[split_names[0]]

        train_split = dataset_bundle["train"] if "train" in dataset_bundle else reference_split
        if "validation" in dataset_bundle:
            val_split = dataset_bundle["validation"]
        elif "val" in dataset_bundle:
            val_split = dataset_bundle["val"]
        elif "test" in dataset_bundle:
            val_split = dataset_bundle["test"]
        else:
            split_dict = train_split.train_test_split(test_size=validation_ratio, seed=seed)
            train_split = split_dict["train"]
            val_split = split_dict["test"]
    else:
        split_dict = dataset_bundle.train_test_split(test_size=validation_ratio, seed=seed)
        train_split = split_dict["train"]
        val_split = split_dict["test"]
        reference_split = train_split

    image_column = detect_image_column(reference_split)
    label_column = detect_label_column(reference_split, image_column)
    if image_column is None or label_column is None:
        raise ValueError("Could not detect image/label columns in the selected dataset.")

    label_feature = reference_split.features.get(label_column)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, dict[str, int]] = {"train": {}, "val": {}}
    class_names: set[str] = set()

    for split_name, split_dataset in {"train": train_split, "val": val_split}.items():
        label_counts: dict[str, int] = {}
        shuffled_split = split_dataset.shuffle(seed=seed)
        for row in shuffled_split:
            label_name = slugify(resolve_label_name(row[label_column], label_feature))
            current_count = label_counts.get(label_name, 0)
            if current_count >= max_samples_per_class:
                continue

            image = hf_image_to_pil(row[image_column]).resize((image_size, image_size))
            destination = output_dir / split_name / label_name
            destination.mkdir(parents=True, exist_ok=True)
            image.save(destination / f"{split_name}_{current_count:05d}.jpg", format="JPEG", quality=95)

            label_counts[label_name] = current_count + 1
            class_names.add(label_name)

        counts[split_name] = label_counts

    if not class_names:
        raise ValueError("No samples were exported. Try a different public dataset or lower the filters.")

    metadata = {
        "dataset_id": dataset_id,
        "output_dir": str(output_dir),
        "class_names": sorted(class_names),
        "counts": counts,
        "exported_at": datetime.utcnow().isoformat(),
    }
    (output_dir / "dataset_info.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def locate_dataset_root(search_dir: Path) -> Path | None:
    candidates = [search_dir] + [path for path in search_dir.rglob("*") if path.is_dir()]
    for candidate in candidates:
        if (candidate / "train").exists() and get_validation_dir(candidate) is not None:
            return candidate
    return None


def extract_uploaded_zip(uploaded_file) -> Path:
    destination = DATASET_BASE_DIR / f"uploaded-{slugify(Path(uploaded_file.name).stem)}"
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(uploaded_file.getvalue())) as zip_file:
        zip_file.extractall(destination)

    return locate_dataset_root(destination) or destination


def discover_local_datasets() -> list[Path]:
    roots: set[Path] = set()

    if DATASET_BASE_DIR.exists():
        direct_root = locate_dataset_root(DATASET_BASE_DIR)
        if direct_root is not None:
            roots.add(direct_root)

        for child in DATASET_BASE_DIR.iterdir():
            if child.is_dir():
                located = locate_dataset_root(child)
                if located is not None:
                    roots.add(located)

    return sorted(roots, key=lambda path: str(path))


@st.cache_resource(show_spinner=False)
def get_spotify_client():
    if not (spotipy and SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET):
        return None

    try:
        return spotipy.Spotify(
            auth_manager=SpotifyClientCredentials(
                client_id=SPOTIPY_CLIENT_ID,
                client_secret=SPOTIPY_CLIENT_SECRET,
            )
        )
    except Exception:
        return None


def spotify_search_tracks(mood: str, limit: int = 3) -> list[tuple[str, str]]:
    client = get_spotify_client()
    if client is None:
        return []

    search_map = {
        "happy": "happy upbeat feel good",
        "sad": "comforting acoustic calm",
        "calm": "calm peaceful ambient",
        "energetic": "energy workout pump up",
    }
    query = search_map.get(mood.lower(), mood)

    try:
        response = client.search(q=query, limit=limit, type="track")
        return [
            (item["name"], item["external_urls"]["spotify"])
            for item in response["tracks"]["items"]
        ]
    except Exception:
        return []


def openai_chat(prompt: str, system_prompt: str = "You are a helpful emotional AI assistant.") -> str:
    if not prompt.strip():
        return "Enter a message first."
    if not OPENAI_API_KEY:
        return "OpenAI is not configured. Set OPENAI_API_KEY to enable chat."

    try:
        if OpenAI is not None:
            client = OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=300,
            )
            content = response.choices[0].message.content
            return content.strip() if content else "No response returned."

        if legacy_openai is not None:
            legacy_openai.api_key = OPENAI_API_KEY
            response = legacy_openai.ChatCompletion.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=300,
            )
            return response["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        return f"OpenAI API error: {exc}"

    return "OpenAI client library is not installed."


def render_recommendation(item: dict[str, str]) -> None:
    st.subheader("Recommended resource")
    st.write(f"**{item['Title']}**")
    st.caption(f"{item['Mood']} • {item['Kind'].replace('_', ' ')} • {item['Source']}")

    if item["Kind"] == "direct_audio":
        try:
            st.audio(item["URL"])
        except Exception:
            pass

    safe_link_button("Open recommended resource", item["URL"])
    st.code(item["URL"])


def render_analysis_panel(catalog_df: pd.DataFrame) -> None:
    col_inputs, col_results = st.columns([1, 1.35])

    with col_inputs:
        st.subheader("User input")
        user = st.text_input("Username", value="guest_user")
        emoji = st.select_slider("Select emoji", options=list(EMOJI_MAP.keys()), value="😊")

        st.caption("Add a face image from your camera or upload one manually.")
        camera_file = st.camera_input("Capture face (optional)")
        uploaded_image = st.file_uploader("Or upload image", type=["png", "jpg", "jpeg"], key="face_upload")

        face_bytes = None
        if uploaded_image is not None:
            face_bytes = uploaded_image.getvalue()
        elif camera_file is not None:
            face_bytes = camera_file.getvalue()

        st.write("Voice / transcript input")
        if hasattr(st, "audio_input"):
            audio_clip = st.audio_input("Record a short voice sample (optional)")
            if st.button("Transcribe recorded audio", key="transcribe_audio"):
                transcript = transcribe_audio_file(audio_clip) if audio_clip is not None else ""
                st.session_state["voice_text"] = transcript
                if transcript:
                    st.success("Voice sample transcribed.")
                else:
                    st.info("No transcript produced. You can still type the transcript manually.")

        voice_text = st.text_input(
            "Voice transcript / mood cue",
            key="voice_text",
            placeholder="I feel overwhelmed but want to calm down",
        )
        user_text = st.text_area(
            "Tell me more",
            value="",
            placeholder="Share a little more context for the fusion engine.",
        )

        analyze_clicked = st.button("Analyze & Recommend", type="primary")

    with col_results:
        if analyze_clicked:
            raw_face_label, face_mood, probabilities, grad_cam_overlay = predict_face_with_best_available_method(face_bytes)
            emoji_mood = EMOJI_MAP.get(emoji, "calm")
            voice_mood = simple_text_to_mood(voice_text)
            text_mood = simple_text_to_mood(user_text)

            scores = {"happy": 0.0, "sad": 0.0, "calm": 0.0, "energetic": 0.0}
            if face_mood in scores:
                scores[face_mood] += 0.40
            scores[emoji_mood] += 0.25
            scores[voice_mood] += 0.15
            scores[text_mood] += 0.20

            fused_mood = max(scores.items(), key=lambda item: item[1])[0]
            recommended_item, top_choices = recommend_resource(user, fused_mood, catalog_df)
            log_id = log_interaction(
                user=user,
                face_label=raw_face_label,
                face_mood=face_mood,
                emoji_mood=emoji_mood,
                voice_mood=voice_mood,
                text_mood=text_mood,
                fused_mood=fused_mood,
                recommended_item=recommended_item,
            )

            st.session_state["last_analysis"] = {
                "user": user,
                "raw_face_label": raw_face_label,
                "face_mood": face_mood,
                "emoji_mood": emoji_mood,
                "voice_mood": voice_mood,
                "text_mood": text_mood,
                "fused_mood": fused_mood,
                "scores": scores,
                "probabilities": probabilities,
            }
            st.session_state["last_recommendation"] = recommended_item
            st.session_state["last_ranked_choices"] = top_choices
            st.session_state["last_log_id"] = log_id

            if grad_cam_overlay is not None:
                st.image(grad_cam_overlay, caption="Grad-CAM overlay (approximate attention map)")

        analysis = st.session_state.get("last_analysis")
        if analysis:
            st.subheader("Detected mood summary")
            st.success(
                " • ".join(
                    [
                        f"Face: {analysis['raw_face_label']} -> {analysis['face_mood']}",
                        f"Emoji: {analysis['emoji_mood']}",
                        f"Voice: {analysis['voice_mood']}",
                        f"Text: {analysis['text_mood']}",
                    ]
                )
            )
            st.info(f"Fused mood: **{analysis['fused_mood']}**")
            st.json(analysis["scores"])

            if analysis.get("probabilities") is not None:
                class_names = load_class_names()
                pred_df = pd.DataFrame(
                    {
                        "Class": class_names[: len(analysis["probabilities"])]
                        if class_names
                        else [f"class_{idx}" for idx in range(len(analysis["probabilities"]))],
                        "Probability": analysis["probabilities"],
                    }
                ).sort_values("Probability", ascending=False)
                st.dataframe(pred_df, use_container_width=True)

            render_recommendation(st.session_state["last_recommendation"])

            tracks = spotify_search_tracks(analysis["fused_mood"], limit=3)
            if tracks:
                st.write("Spotify picks")
                for track_name, url in tracks:
                    st.markdown(f"- [{track_name}]({url})")
            else:
                st.caption("Spotify suggestions are unavailable until SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET are set.")

            if st.session_state.get("last_ranked_choices"):
                with st.expander("Alternative picks considered"):
                    for candidate in st.session_state["last_ranked_choices"]:
                        st.markdown(f"- **{candidate['Title']}** ({candidate['Kind']})")

        last_item = st.session_state.get("last_recommendation")
        if last_item:
            st.subheader("Feedback")
            like_col, skip_col = st.columns(2)
            with like_col:
                if st.button("👍 I liked this", key="feedback_like"):
                    record_feedback(last_item["ItemId"], liked=True)
                    update_log_feedback(st.session_state.get("last_log_id"), "liked")
                    st.success("Preference updated.")
            with skip_col:
                if st.button("👎 Skip / not relevant", key="feedback_skip"):
                    record_feedback(last_item["ItemId"], liked=False)
                    update_log_feedback(st.session_state.get("last_log_id"), "skipped")
                    st.info("Noted. Future repeats will be reduced.")

    st.markdown("---")
    st.subheader("Chat with Emotional AI")
    chat_prompt = st.text_input("Say something to the emotional AI", key="chat_prompt")
    if st.button("Send chat", key="send_chat"):
        fused_mood = (
            st.session_state.get("last_analysis", {}).get("fused_mood", "calm")
            if st.session_state.get("last_analysis")
            else "calm"
        )
        reply = openai_chat(
            prompt=f"Current inferred mood: {fused_mood}. User message: {chat_prompt}",
            system_prompt="You are a calm, supportive, emotionally aware assistant.",
        )
        st.write(reply)


def render_training_panel() -> None:
    st.subheader("Automatic dataset setup + training")
    left_col, right_col = st.columns([1, 1.15])

    with left_col:
        st.write("**Public dataset automation**")
        if DATASETS_AVAILABLE:
            dataset_choice = st.selectbox(
                "Public source",
                options=list(PUBLIC_DATASET_PRESETS.keys()),
            )
            dataset_id = PUBLIC_DATASET_PRESETS[dataset_choice]
            custom_dataset_id = ""
            if dataset_id == "__custom__":
                custom_dataset_id = st.text_input(
                    "Custom public Hugging Face dataset ID",
                    placeholder="example-user/public-emotion-dataset",
                )
                dataset_id = custom_dataset_id.strip()

            max_samples_per_class = st.slider(
                "Max images per class per split",
                min_value=20,
                max_value=300,
                value=120,
                step=10,
            )
            image_size = st.select_slider(
                "Export image size",
                options=[48, 96, 128, 160, 224],
                value=224,
            )
            validation_ratio = st.slider(
                "Validation ratio (used when the dataset has no validation/test split)",
                min_value=0.1,
                max_value=0.4,
                value=0.2,
                step=0.05,
            )

            if st.button("Download and prepare public dataset", key="prepare_public_dataset"):
                if not dataset_id:
                    st.error("Enter a custom public dataset ID first.")
                else:
                    target_dir = DATASET_BASE_DIR / slugify(dataset_id.replace("/", "-"))
                    with st.spinner("Downloading, reformatting, and exporting the dataset..."):
                        try:
                            metadata = prepare_public_image_dataset(
                                dataset_id=dataset_id,
                                output_dir=target_dir,
                                max_samples_per_class=max_samples_per_class,
                                image_size=image_size,
                                seed=42,
                                validation_ratio=validation_ratio,
                            )
                            st.session_state["active_dataset_root"] = metadata["output_dir"]
                            st.success(f"Prepared dataset at {metadata['output_dir']}")
                            st.json(metadata)
                        except Exception as exc:
                            st.error(f"Dataset preparation failed: {exc}")
        else:
            st.warning("Install the `datasets` package to enable automatic public dataset download.")

        st.caption(
            "The built-in no-auth flow targets openly accessible public Hugging Face datasets. "
            "Common academic sets such as RAF-DB or CK+ often require manual access or license steps."
        )

        st.markdown("---")
        st.write("**Upload your own dataset ZIP**")
        dataset_zip = st.file_uploader(
            "Upload a ZIP with train/val or train/test folders",
            type=["zip"],
            key="dataset_zip",
        )
        if dataset_zip is not None and st.button("Extract uploaded ZIP", key="extract_uploaded_zip"):
            with st.spinner("Extracting dataset ZIP..."):
                try:
                    dataset_root = extract_uploaded_zip(dataset_zip)
                    st.session_state["active_dataset_root"] = str(dataset_root)
                    st.success(f"Dataset extracted to {dataset_root}")
                except Exception as exc:
                    st.error(f"ZIP extraction failed: {exc}")

    with right_col:
        st.write("**Transfer-learning model**")

        local_datasets = discover_local_datasets()
        dataset_options = [str(path) for path in local_datasets]

        if dataset_options:
            default_index = 0
            active_root = st.session_state.get("active_dataset_root")
            if active_root in dataset_options:
                default_index = dataset_options.index(active_root)

            selected_dataset = st.selectbox(
                "Local dataset root",
                options=dataset_options,
                index=default_index,
            )
            st.session_state["active_dataset_root"] = selected_dataset
            st.caption(f"Training will use: {selected_dataset}")
        else:
            selected_dataset = ""
            st.info("No local dataset found yet. Prepare a public dataset or upload your own ZIP.")

        preview_classes = st.number_input(
            "Number of classes for architecture preview",
            min_value=2,
            max_value=12,
            value=4,
        )

        if TF_AVAILABLE:
            preview_col, train_col = st.columns(2)
            with preview_col:
                if st.button("Preview MobileNetV2 model", key="preview_model"):
                    preview_model = build_transfer_model(int(preview_classes))
                    st.code(summarize_model(preview_model))

            with train_col:
                st.write(" ")

            epochs = st.slider("Epochs", min_value=1, max_value=8, value=3)
            batch_size = st.select_slider("Batch size", options=[8, 16, 24, 32], value=16)
            fine_tune = st.checkbox("Fine-tune the last MobileNetV2 layers after warm-up", value=True)

            if st.button("Train model", key="train_model"):
                if not selected_dataset:
                    st.error("Select or prepare a dataset first.")
                else:
                    with st.spinner("Training MobileNetV2 transfer model..."):
                        try:
                            training_result = train_model_on_directory(
                                dataset_root=Path(selected_dataset),
                                epochs=epochs,
                                batch_size=batch_size,
                                fine_tune=fine_tune,
                            )
                            st.success(f"Training complete. Saved model to {MODEL_PATH.name}")

                            history_df = pd.DataFrame(training_result["history"])
                            if not history_df.empty:
                                st.line_chart(history_df)

                            evaluation = training_result["evaluation"]
                            if isinstance(evaluation, dict) and "error" not in evaluation:
                                metrics_df = pd.DataFrame(
                                    [
                                        {
                                            "macro_precision": evaluation["macro_precision"],
                                            "macro_recall": evaluation["macro_recall"],
                                            "macro_f1": evaluation["macro_f1"],
                                            "roc_auc_macro": evaluation["roc_auc_macro"],
                                        }
                                    ]
                                )
                                st.dataframe(metrics_df, use_container_width=True)

                                report_df = pd.DataFrame(evaluation["classification_report"]).T
                                st.write("Classification report")
                                st.dataframe(report_df, use_container_width=True)

                                confusion_df = pd.DataFrame(
                                    evaluation["confusion_matrix"],
                                    index=training_result["class_names"],
                                    columns=training_result["class_names"],
                                )
                                st.write("Confusion matrix")
                                st.dataframe(confusion_df, use_container_width=True)
                            else:
                                st.info("Evaluation metrics are unavailable because scikit-learn is not installed.")
                        except Exception as exc:
                            st.error(f"Training failed: {exc}")
        else:
            st.warning("TensorFlow is not installed, so the app will fall back to heuristic face mood detection.")


def render_catalog_and_dashboard(catalog_df: pd.DataFrame) -> None:
    st.subheader("Expanded resource catalog")
    st.caption(
        f"The catalog contains {len(catalog_df)} structured entries across four moods "
        "and is exported to resource_catalog.csv on every run."
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Catalog items", len(catalog_df))
    metric_cols[1].metric("Happy items", int((catalog_df["Mood"] == "HAPPY").sum()))
    metric_cols[2].metric("Calm items", int((catalog_df["Mood"] == "CALM").sum()))
    metric_cols[3].metric("Direct audio fallbacks", int((catalog_df["Kind"] == "direct_audio").sum()))

    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        selected_mood = st.selectbox("Filter by mood", ["ALL"] + sorted(catalog_df["Mood"].unique().tolist()))
    with filter_col2:
        selected_kind = st.selectbox("Filter by kind", ["ALL"] + sorted(catalog_df["Kind"].unique().tolist()))

    filtered_df = catalog_df.copy()
    if selected_mood != "ALL":
        filtered_df = filtered_df[filtered_df["Mood"] == selected_mood]
    if selected_kind != "ALL":
        filtered_df = filtered_df[filtered_df["Kind"] == selected_kind]

    st.dataframe(filtered_df, use_container_width=True)
    st.download_button(
        "Download resource_catalog.csv",
        data=catalog_df.to_csv(index=False).encode("utf-8"),
        file_name="resource_catalog.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.subheader("Digital Emotional Twin logs")
    logs = load_logs()
    if logs.empty:
        st.info("No log entries yet. Analyze a mood to generate history.")
    else:
        st.dataframe(logs.sort_values("Time", ascending=False), use_container_width=True)
        count_cols = st.columns(2)
        with count_cols[0]:
            st.write("Fused mood distribution")
            st.bar_chart(logs["FusedMood"].value_counts())
        with count_cols[1]:
            st.write("Feedback distribution")
            feedback_series = logs["Feedback"].replace("", "pending").fillna("pending")
            st.bar_chart(feedback_series.value_counts())

    st.markdown("---")
    st.subheader("Recommender stats")
    stats = load_recommender_stats()
    if stats.empty:
        st.info("No recommender activity yet.")
    else:
        stats_display = stats.copy()
        exposures = stats_display["Exposures"].replace(0, np.nan)
        stats_display["LikeRate"] = (stats_display["Likes"] / exposures).fillna(0.0)
        stats_display["SkipRate"] = (stats_display["Skips"] / exposures).fillna(0.0)
        stats_display = stats_display.sort_values(["LikeRate", "Exposures"], ascending=[False, False])
        st.dataframe(stats_display, use_container_width=True)
        st.bar_chart(stats_display.set_index("Title")[["LikeRate", "SkipRate"]].head(15))


def render_setup_notes() -> None:
    st.subheader("Developer setup")
    st.markdown(
        """
1. Install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Configure environment variables:
   - `OPENAI_API_KEY`
   - `SPOTIPY_CLIENT_ID`
   - `SPOTIPY_CLIENT_SECRET`

3. Run:
   ```bash
   streamlit run app.py
   ```

4. Optional dataset automation:
   - Install `datasets` to enable the no-auth public dataset downloader.
   - Use the built-in FER2013 preset or a custom public Hugging Face image dataset ID.
        """
    )

    st.markdown("---")
    st.subheader("Mobile / Android recommendation")
    st.markdown(
        """
- Best low-cost path: host the Streamlit app and let users install it from the browser as a home-screen shortcut / browser app.
- On Android Chrome, use **Add to Home screen**.
- On desktop Chrome/Edge, use **Install app**.
- This gives an installable experience, but Streamlit still depends on a running backend, so it is more accurate to call it *PWA-like* than a fully offline PWA.
- If you need a true native APK, use Kivy/Buildozer or a mobile-specific frontend that talks to a Python backend.
        """
    )

    st.markdown("---")
    st.subheader("Notes and limitations")
    st.markdown(
        """
- The face pipeline gracefully falls back to a heuristic when TensorFlow or a trained model is unavailable.
- The public dataset downloader focuses on openly accessible datasets; not every academic emotion dataset can be fetched without manual access steps.
- CSV persistence is good for prototypes. For production, move logs and recommender stats to a database.
- Emotion detection can be useful for supportive experiences, but it should not be used for high-stakes decisions without careful validation and ethical review.
        """
    )


def main() -> None:
    st.set_page_config(page_title="CEI-ALOS FINAL", layout="wide")
    initialize_runtime_files()
    init_session_state()

    catalog_df = build_resource_catalog()
    ensure_catalog_export(catalog_df)

    st.title("🧠 CEI-ALOS — Cognitive Emotional Intelligence & Adaptive Lifestyle OS")
    st.markdown(
        "Single-file Streamlit prototype with multi-modal fusion, transfer learning, "
        "explainability, a history-aware recommender, automatic public dataset preparation, "
        "and a 100+ item structured resource catalog."
    )

    status_cols = st.columns(4)
    status_cols[0].metric("Catalog size", len(catalog_df))
    status_cols[1].metric("Datasets ready", len(discover_local_datasets()))
    status_cols[2].metric("Face model", "Ready" if MODEL_PATH.exists() else "Not trained")
    configured_api_count = int(bool(OPENAI_API_KEY)) + int(bool(SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET))
    status_cols[3].metric("APIs configured", f"{configured_api_count}/2")

    tab_experience, tab_training, tab_catalog, tab_notes = st.tabs(
        ["Experience", "Training + Dataset", "Catalog + Analytics", "Setup + Notes"]
    )

    with tab_experience:
        render_analysis_panel(catalog_df)

    with tab_training:
        render_training_panel()

    with tab_catalog:
        render_catalog_and_dashboard(catalog_df)

    with tab_notes:
        render_setup_notes()

    st.markdown("---")
    st.caption(
        "CEI-ALOS persists log files, recommender stats, trained models, and prepared datasets in the working directory."
    )


if __name__ == "__main__":
    main()
