from __future__ import annotations

import argparse
import io
import json
import os
import random
import re
import shutil
import textwrap
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote_plus

import numpy as np
import pandas as pd

try:
    import streamlit as st
except Exception:
    st = None

try:
    from PIL import Image
except Exception:
    Image = None

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
    classification_report = None
    confusion_matrix = None
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
    from datasets import get_dataset_split_names, load_dataset

    HF_DATASETS_AVAILABLE = True
except Exception:
    get_dataset_split_names = None
    load_dataset = None
    HF_DATASETS_AVAILABLE = False


APP_TITLE = "CEI-ALOS FINAL"
APP_SUBTITLE = "Cognitive Emotional Intelligence + Adaptive Lifestyle OS"
BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DATA_FILE = BASE_DIR / "cei_twin_log.csv"
RL_STATS_FILE = BASE_DIR / "recommender_stats.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "mobile_transfer.keras"
MODEL_META_PATH = MODEL_DIR / "mobile_transfer_metadata.json"
DATASET_DIR = BASE_DIR / "dataset"
CATALOG_EXPORT_PATH = BASE_DIR / "resource_catalog.csv"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

MODEL_DIR.mkdir(exist_ok=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
SPOTIPY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID", "").strip()
SPOTIPY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET", "").strip()

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OpenAI and OPENAI_API_KEY else None
spotify_client = None
if spotipy and SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET:
    try:
        spotify_client = spotipy.Spotify(
            auth_manager=SpotifyClientCredentials(
                client_id=SPOTIPY_CLIENT_ID,
                client_secret=SPOTIPY_CLIENT_SECRET,
            )
        )
    except Exception:
        spotify_client = None


EMOJI_MAP = {
    "😊": "happy",
    "😄": "happy",
    "😢": "sad",
    "😭": "sad",
    "😡": "energetic",
    "😴": "calm",
    "😌": "calm",
    "😍": "happy",
    "😬": "energetic",
}

MOOD_CHOICES = ["happy", "sad", "calm", "energetic"]
EMOTION_TO_MOOD = {
    "happy": "happy",
    "joy": "happy",
    "love": "happy",
    "surprise": "energetic",
    "excited": "energetic",
    "angry": "energetic",
    "anger": "energetic",
    "disgust": "energetic",
    "fear": "sad",
    "afraid": "sad",
    "sad": "sad",
    "grief": "sad",
    "lonely": "sad",
    "neutral": "calm",
    "calm": "calm",
    "relaxed": "calm",
    "peace": "calm",
}

FER2025_LABELS = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise",
}

DATASET_NUMERIC_LABEL_MAPS = {
    "FER2025": FER2025_LABELS,
}

FALLBACK_ACTIONS = {
    "happy": "Offline fallback: play a saved upbeat playlist or write down 3 wins from today.",
    "sad": "Offline fallback: play a downloaded comfort track and do a 2-minute breathing exercise.",
    "calm": "Offline fallback: use a saved lo-fi or rain-sound file and keep a low-distraction routine.",
    "energetic": "Offline fallback: play a downloaded workout mix and take a short movement break.",
}


@dataclass(frozen=True)
class DatasetSource:
    key: str
    label: str
    year: str
    dataset_id: str
    description: str
    license_name: str
    auto_download: bool
    recommended_sample: int
    notes: str
    suggested_max_classes: int


@dataclass(frozen=True)
class CatalogEntry:
    id: str
    mood: str
    title: str
    url: str
    resource_type: str
    source: str
    playable: bool
    tags: str
    offline_fallback: str


DATASET_SOURCES = {
    "FER2025": DatasetSource(
        key="FER2025",
        label="FER2025 (Hugging Face, 2025)",
        year="2025",
        dataset_id="imadhavan/FER2025",
        description=(
            "Recent balanced 7-class facial emotion dataset. Use sample-mode in this app "
            "for Windows 11 machines with 4 GB RAM."
        ),
        license_name="CC BY-NC 4.0",
        auto_download=True,
        recommended_sample=900,
        notes=(
            "Best fit for this project. Start with 600-1200 sampled images and 3 classes or "
            "7 classes depending on your memory budget."
        ),
        suggested_max_classes=7,
    ),
    "EmoNet-Face-Big": DatasetSource(
        key="EmoNet-Face-Big",
        label="EmoNet-Face-Big (Hugging Face, 2025 synthetic)",
        year="2025",
        dataset_id="laion/EmoNet-Face-Big",
        description=(
            "Large synthetic emotion dataset with a richer taxonomy than classic FER datasets. "
            "Useful for experiments or pretraining-style sampling."
        ),
        license_name="See dataset card on Hugging Face",
        auto_download=True,
        recommended_sample=700,
        notes=(
            "High class count. Limit to 6-8 classes for lightweight MobileNetV2 experiments "
            "on low-RAM laptops."
        ),
        suggested_max_classes=8,
    ),
    "MER2024": DatasetSource(
        key="MER2024",
        label="MER2024 (recent multimodal research dataset)",
        year="2024",
        dataset_id="MERChallenge/MER2024",
        description=(
            "Recent multimodal emotion benchmark. Included for project planning, but auto-download "
            "is disabled because challenge datasets often require manual terms review."
        ),
        license_name="Challenge-specific terms",
        auto_download=False,
        recommended_sample=0,
        notes="Use the ZIP uploader if you already have approved access and local copies.",
        suggested_max_classes=7,
    ),
    "MER2023": DatasetSource(
        key="MER2023",
        label="MER2023 (recent multimodal research dataset)",
        year="2023",
        dataset_id="MERChallenge/MER2023",
        description=(
            "Another recent multimodal benchmark. Listed so the project can document current "
            "2023-2026 dataset options, but the app does not auto-download restricted sources."
        ),
        license_name="Challenge-specific terms",
        auto_download=False,
        recommended_sample=0,
        notes="For older datasets like RAF-DB or CK+, use the manual ZIP import path.",
        suggested_max_classes=7,
    ),
}


DIRECT_VIDEO_URLS = {
    "happy": [
        "https://www.youtube.com/watch?v=d-diB65scQU",
        "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
        "https://www.youtube.com/watch?v=ZMO_XC9w7Lw",
        "https://www.youtube.com/watch?v=OPf0YbXqDm0",
    ],
    "sad": [
        "https://www.youtube.com/watch?v=2XU0oxnq2qU",
        "https://www.youtube.com/watch?v=inpok4MKVLM",
        "https://www.youtube.com/watch?v=1vx8iUvfyCY",
        "https://www.youtube.com/watch?v=6p_yaNFSYao",
    ],
    "calm": [
        "https://www.youtube.com/watch?v=5qap5aO4i9A",
        "https://www.youtube.com/watch?v=v7AYKMP6rOE",
        "https://www.youtube.com/watch?v=ausxoXBrmWs",
        "https://www.youtube.com/watch?v=lFcSrYw-ARY",
    ],
    "energetic": [
        "https://www.youtube.com/watch?v=HgzGwKwLmgM",
        "https://www.youtube.com/watch?v=ml6cT4AZdqI",
        "https://www.youtube.com/watch?v=btPJPFnesV4",
        "https://www.youtube.com/watch?v=fLexgOxsZu0",
    ],
}

YOUTUBE_SEARCH_QUERIES = {
    "happy": [
        "feel good morning playlist",
        "gratitude meditation for joy",
        "smile reset breathing",
        "upbeat study break songs",
        "confidence booster playlist",
        "dance break positivity mix",
        "sunny day walk music",
        "celebration songs clean mix",
    ],
    "sad": [
        "comfort songs for hard days",
        "self compassion meditation",
        "gentle piano for sadness",
        "journaling prompts for grief",
        "calm talk for emotional overwhelm",
        "how to regulate emotions breathing",
        "soft acoustic healing songs",
        "mindfulness for low mood",
    ],
    "calm": [
        "lofi focus session",
        "rain sounds for studying",
        "deep breathing box breathing",
        "ambient nature soundscape",
        "yoga stretch for focus",
        "mindful minute reset",
        "peaceful instrumental for reading",
        "desk meditation for students",
    ],
    "energetic": [
        "power workout mix clean",
        "motivation speech short",
        "focus sprint music",
        "confidence gym playlist",
        "high energy coding playlist",
        "quick movement break",
        "pump up songs clean mix",
        "productive deep work soundtrack",
    ],
}

SPOTIFY_SEARCH_QUERIES = {
    "happy": [
        "happy playlist",
        "feel good hits",
        "dance pop mood boost",
        "sunshine acoustic",
        "happy workday mix",
        "good vibes only",
        "joyful study playlist",
        "positive energy songs",
    ],
    "sad": [
        "comfort songs",
        "gentle piano",
        "soft acoustic sad",
        "emotional reset playlist",
        "self care songs",
        "calm down playlist",
        "late night reflection",
        "healing ambient music",
    ],
    "calm": [
        "lofi beats",
        "peaceful piano",
        "deep focus",
        "ambient calm",
        "sleepy acoustic",
        "rainy day jazz",
        "meditation music",
        "reading soundtrack",
    ],
    "energetic": [
        "workout hits",
        "high energy mix",
        "running playlist",
        "focus power playlist",
        "motivation songs",
        "hype coding mix",
        "gym boost",
        "productivity bangers",
    ],
}

YTMUSIC_SEARCH_QUERIES = {
    "happy": [
        "cheerful bollywood playlist",
        "happy indie songs",
        "feel good retro hits",
        "weekend road trip music",
        "bright cafe playlist",
        "smiling study mix",
        "joyful k-pop clean mix",
        "dance around the room songs",
    ],
    "sad": [
        "soft ghazal comfort songs",
        "slow rainy evening music",
        "healing instrumental violin",
        "mindful pause audio",
        "gentle sleep prep playlist",
        "quiet reflection songs",
        "self soothing instrumental",
        "reset after bad day mix",
    ],
    "calm": [
        "brown noise focus",
        "deep work lofi",
        "calm nature piano",
        "peaceful coding soundtrack",
        "meditation bell sounds",
        "morning stillness music",
        "slow ambient reading music",
        "evening tea playlist",
    ],
    "energetic": [
        "high bpm study music",
        "confidence rap clean",
        "afrobeats energy mix",
        "sports warmup songs",
        "power walk music",
        "focus hype soundtrack",
        "quick cardio playlist",
        "energetic morning mix",
    ],
}


def ensure_runtime_files() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    if not RL_STATS_FILE.exists():
        pd.DataFrame(
            columns=["ItemId", "Exposures", "Likes", "Skips", "LastShown", "LastFeedback"]
        ).to_csv(RL_STATS_FILE, index=False)


def require_streamlit() -> None:
    if st is None:
        raise RuntimeError(
            "Streamlit is not installed. Install the dependencies and run `streamlit run app.py`."
        )


def sanitize_slug(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower())
    text = text.strip("_")
    return text or "unknown"


def normalize_label_to_mood(label: str) -> str:
    token = sanitize_slug(label).replace("_", " ")
    for key, mood in EMOTION_TO_MOOD.items():
        if key in token:
            return mood
    return "calm"


@lru_cache(maxsize=1)
def build_resource_catalog() -> tuple[CatalogEntry, ...]:
    entries: list[CatalogEntry] = []
    for mood in MOOD_CHOICES:
        fallback = FALLBACK_ACTIONS[mood]
        for index, url in enumerate(DIRECT_VIDEO_URLS[mood], start=1):
            entries.append(
                CatalogEntry(
                    id=f"{mood}_direct_{index}",
                    mood=mood,
                    title=f"{mood.title()} direct video {index}",
                    url=url,
                    resource_type="youtube_video",
                    source="YouTube",
                    playable=True,
                    tags=f"{mood},direct,video",
                    offline_fallback=fallback,
                )
            )
        for index, query in enumerate(YOUTUBE_SEARCH_QUERIES[mood], start=1):
            entries.append(
                CatalogEntry(
                    id=f"{mood}_yt_search_{index}",
                    mood=mood,
                    title=f"{mood.title()} YouTube search: {query.title()}",
                    url=f"https://www.youtube.com/results?search_query={quote_plus(query)}",
                    resource_type="youtube_search",
                    source="YouTube",
                    playable=False,
                    tags=f"{mood},youtube,search,{sanitize_slug(query)}",
                    offline_fallback=fallback,
                )
            )
        for index, query in enumerate(SPOTIFY_SEARCH_QUERIES[mood], start=1):
            entries.append(
                CatalogEntry(
                    id=f"{mood}_spotify_search_{index}",
                    mood=mood,
                    title=f"{mood.title()} Spotify search: {query.title()}",
                    url=f"https://open.spotify.com/search/{quote_plus(query)}",
                    resource_type="spotify_search",
                    source="Spotify",
                    playable=False,
                    tags=f"{mood},spotify,search,{sanitize_slug(query)}",
                    offline_fallback=fallback,
                )
            )
        for index, query in enumerate(YTMUSIC_SEARCH_QUERIES[mood], start=1):
            entries.append(
                CatalogEntry(
                    id=f"{mood}_ytmusic_search_{index}",
                    mood=mood,
                    title=f"{mood.title()} YouTube Music search: {query.title()}",
                    url=f"https://music.youtube.com/search?q={quote_plus(query)}",
                    resource_type="ytmusic_search",
                    source="YouTube Music",
                    playable=False,
                    tags=f"{mood},ytmusic,search,{sanitize_slug(query)}",
                    offline_fallback=fallback,
                )
            )
    return tuple(entries)


def catalog_dataframe() -> pd.DataFrame:
    return pd.DataFrame([asdict(entry) for entry in build_resource_catalog()])


def export_resource_catalog_csv(path: Path = CATALOG_EXPORT_PATH) -> Path:
    df = catalog_dataframe()
    df.to_csv(path, index=False)
    return path


def init_recommender_stats() -> pd.DataFrame:
    ensure_runtime_files()
    return pd.read_csv(RL_STATS_FILE)


def write_recommender_stats(df: pd.DataFrame) -> None:
    df.to_csv(RL_STATS_FILE, index=False)


def update_stats(item_id: str, feedback: str = "shown") -> None:
    df = init_recommender_stats()
    if item_id not in df["ItemId"].astype(str).values:
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        {
                            "ItemId": item_id,
                            "Exposures": 0,
                            "Likes": 0,
                            "Skips": 0,
                            "LastShown": "",
                            "LastFeedback": "",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    row_mask = df["ItemId"].astype(str) == str(item_id)
    now_text = datetime.utcnow().isoformat()
    if feedback == "shown":
        df.loc[row_mask, "Exposures"] = df.loc[row_mask, "Exposures"].fillna(0).astype(int) + 1
        df.loc[row_mask, "LastShown"] = now_text
    elif feedback == "liked":
        df.loc[row_mask, "Likes"] = df.loc[row_mask, "Likes"].fillna(0).astype(int) + 1
        df.loc[row_mask, "LastFeedback"] = "liked"
    elif feedback == "skipped":
        df.loc[row_mask, "Skips"] = df.loc[row_mask, "Skips"].fillna(0).astype(int) + 1
        df.loc[row_mask, "LastFeedback"] = "skipped"
    write_recommender_stats(df)


def log_interaction(
    user: str,
    face_label: str,
    face_mood: str,
    emoji_mood: str,
    voice_mood: str,
    text_mood: str,
    fused_mood: str,
    recommendation: dict[str, Any],
    feedback: str = "",
) -> None:
    entry = {
        "User": user,
        "TimeUTC": datetime.utcnow().isoformat(),
        "FaceLabel": face_label,
        "FaceMood": face_mood,
        "EmojiMood": emoji_mood,
        "VoiceMood": voice_mood,
        "TextMood": text_mood,
        "FusedMood": fused_mood,
        "RecommendedId": recommendation.get("id", ""),
        "RecommendedTitle": recommendation.get("title", ""),
        "RecommendedUrl": recommendation.get("url", ""),
        "RecommendedSource": recommendation.get("source", ""),
        "Feedback": feedback,
    }
    df_new = pd.DataFrame([entry])
    if DATA_FILE.exists():
        df = pd.read_csv(DATA_FILE)
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(DATA_FILE, index=False)


def update_last_feedback(user: str, item_id: str, feedback: str) -> None:
    if not DATA_FILE.exists():
        return
    df = pd.read_csv(DATA_FILE)
    matches = (df["User"].astype(str) == str(user)) & (df["RecommendedId"].astype(str) == str(item_id))
    if not matches.any():
        return
    index = df[matches].index[-1]
    df.loc[index, "Feedback"] = feedback
    df.to_csv(DATA_FILE, index=False)


def simple_text_to_mood(text: str) -> str:
    if not text:
        return "calm"
    lowered = text.lower()
    if any(token in lowered for token in ["sad", "unhappy", "depressed", "cry", "lonely", "grief"]):
        return "sad"
    if any(token in lowered for token in ["happy", "joy", "excited", "love", "great", "awesome"]):
        return "happy"
    if any(token in lowered for token in ["angry", "mad", "irritat", "frustrat", "rage", "stressed"]):
        return "energetic"
    return "calm"


def heuristic_face_mood_from_bytes(image_bytes: bytes | None) -> str:
    if not image_bytes or Image is None:
        return "calm"
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("L")
        brightness = float(np.asarray(image).mean())
    except Exception:
        return "calm"
    if brightness < 85:
        return "sad"
    if brightness > 165:
        return "happy"
    return "calm"


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


def build_transfer_model(num_classes: int):
    if not TF_AVAILABLE:
        return None
    try:
        base = tf.keras.applications.MobileNetV2(
            weights="imagenet",
            include_top=False,
            input_shape=(224, 224, 3),
        )
    except Exception:
        base = tf.keras.applications.MobileNetV2(
            weights=None,
            include_top=False,
            input_shape=(224, 224, 3),
        )
    base.trainable = False
    inputs = layers.Input(shape=(224, 224, 3), name="image")
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="emotion")(x)
    model = models.Model(inputs, outputs, name="cei_alos_mobilenetv2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def format_model_summary(model: Any) -> str:
    lines: list[str] = []
    model.summary(print_fn=lines.append)
    return "\n".join(lines)


def load_image_for_model(image_bytes: bytes | None) -> np.ndarray | None:
    if not image_bytes or Image is None:
        return None
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
        return np.asarray(image, dtype=np.float32)
    except Exception:
        return None


@lru_cache(maxsize=1)
def load_saved_model():
    if not (TF_AVAILABLE and MODEL_PATH.exists()):
        return None, {}
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    metadata = {}
    if MODEL_META_PATH.exists():
        try:
            metadata = json.loads(MODEL_META_PATH.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}
    return model, metadata


def save_model_metadata(class_names: list[str], extra: dict[str, Any] | None = None) -> None:
    payload = {
        "class_names": class_names,
        "saved_at_utc": datetime.utcnow().isoformat(),
    }
    if extra:
        payload.update(extra)
    MODEL_META_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def find_last_conv_layer_name(model: Any) -> str | None:
    if not TF_AVAILABLE:
        return None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None


def grad_cam_heatmap(image_array: np.ndarray, model: Any) -> np.ndarray | None:
    if not TF_AVAILABLE or cv2 is None:
        return None
    last_conv_layer_name = find_last_conv_layer_name(model)
    if not last_conv_layer_name:
        return None
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output],
    )
    input_batch = np.expand_dims(image_array.astype(np.float32), axis=0)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_batch)
        index = tf.argmax(predictions[0])
        loss = predictions[:, index]
    gradients = tape.gradient(loss, conv_outputs)
    if gradients is None:
        return None
    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_gradients, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    max_value = tf.math.reduce_max(heatmap)
    if float(max_value) <= 0:
        return None
    heatmap = (heatmap / max_value).numpy()
    heatmap = cv2.resize(heatmap, (224, 224))
    return np.uint8(255 * heatmap)


def overlay_heatmap(image_array: np.ndarray, heatmap: np.ndarray | None, alpha: float = 0.35) -> np.ndarray | None:
    if cv2 is None or heatmap is None:
        return None
    base_image = np.clip(image_array, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(colored, alpha, base_image, 1 - alpha, 0)
    return overlay


def predict_face_emotion(image_bytes: bytes | None) -> dict[str, Any]:
    if not image_bytes:
        return {
            "label": "calm",
            "mood": "calm",
            "confidence": None,
            "overlay": None,
            "method": "no_image",
        }
    if TF_AVAILABLE and MODEL_PATH.exists():
        try:
            model, metadata = load_saved_model()
            image_array = load_image_for_model(image_bytes)
            if model is not None and image_array is not None:
                predictions = model.predict(np.expand_dims(image_array, axis=0), verbose=0)[0]
                index = int(np.argmax(predictions))
                class_names = metadata.get("class_names") or [f"class_{i}" for i in range(len(predictions))]
                label = class_names[index] if index < len(class_names) else f"class_{index}"
                mood = normalize_label_to_mood(label)
                heatmap = grad_cam_heatmap(image_array, model)
                overlay = overlay_heatmap(image_array, heatmap)
                return {
                    "label": str(label),
                    "mood": mood,
                    "confidence": float(predictions[index]),
                    "overlay": overlay,
                    "method": "trained_model",
                }
        except Exception:
            pass
    heuristic_mood = heuristic_face_mood_from_bytes(image_bytes)
    return {
        "label": heuristic_mood,
        "mood": heuristic_mood,
        "confidence": None,
        "overlay": None,
        "method": "heuristic",
    }


def evaluate_model_on_dataset(model: Any, dataset: Any, class_names: list[str]) -> dict[str, Any]:
    if not (TF_AVAILABLE and SKLEARN_AVAILABLE):
        return {"error": "tensorflow or scikit-learn is not available"}
    true_labels: list[int] = []
    predicted_labels: list[int] = []
    predicted_probabilities: list[list[float]] = []
    for images, labels in dataset:
        probabilities = model.predict(images, verbose=0)
        predicted_probabilities.extend(probabilities.tolist())
        predicted_labels.extend(np.argmax(probabilities, axis=1).tolist())
        true_labels.extend(labels.numpy().tolist())
    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(true_labels, predicted_labels).tolist()
    roc_auc_macro = None
    try:
        binarizer = LabelBinarizer()
        binarizer.fit(range(len(class_names)))
        y_true_bin = binarizer.transform(true_labels)
        roc_auc_macro = roc_auc_score(
            y_true_bin,
            np.asarray(predicted_probabilities),
            average="macro",
            multi_class="ovr",
        )
    except Exception:
        roc_auc_macro = None
    return {
        "report": report,
        "confusion_matrix": matrix,
        "roc_auc_macro": roc_auc_macro,
    }


def dataset_has_images(split_dir: Path) -> bool:
    if not split_dir.exists():
        return False
    for path in split_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            return True
    return False


def build_training_datasets(dataset_root: Path, batch_size: int = 8):
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is not installed.")
    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"
    test_dir = dataset_root / "test"
    if not dataset_has_images(train_dir):
        raise RuntimeError("No training images found under dataset/train.")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(224, 224),
        batch_size=batch_size,
        shuffle=True,
    )
    class_names = list(train_ds.class_names)
    val_ds = None
    test_ds = None
    if dataset_has_images(val_dir):
        val_ds = tf.keras.utils.image_dataset_from_directory(
            val_dir,
            image_size=(224, 224),
            batch_size=batch_size,
            shuffle=False,
        )
    if dataset_has_images(test_dir):
        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            image_size=(224, 224),
            batch_size=batch_size,
            shuffle=False,
        )
    augmentation = models.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.05),
        ],
        name="augmentation",
    )
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.map(
        lambda x, y: (augmentation(x, training=True), y),
        num_parallel_calls=autotune,
    ).prefetch(autotune)
    if val_ds is not None:
        val_ds = val_ds.prefetch(autotune)
    if test_ds is not None:
        test_ds = test_ds.prefetch(autotune)
    return train_ds, val_ds, test_ds, class_names


def emotion_fusion(face_mood: str, emoji_mood: str, voice_mood: str, text_mood: str) -> tuple[str, dict[str, float]]:
    scores = {mood: 0.0 for mood in MOOD_CHOICES}
    if face_mood in scores:
        scores[face_mood] += 0.40
    if emoji_mood in scores:
        scores[emoji_mood] += 0.25
    if voice_mood in scores:
        scores[voice_mood] += 0.20
    if text_mood in scores:
        scores[text_mood] += 0.15
    fused = max(scores.items(), key=lambda item: item[1])[0]
    return fused, scores


def spotify_search_tracks(mood: str, limit: int = 3) -> list[tuple[str, str]]:
    if spotify_client is None:
        return []
    try:
        result = spotify_client.search(q=f"{mood} playlist", limit=limit, type="track")
        items = result.get("tracks", {}).get("items", [])
        return [(item["name"], item["external_urls"]["spotify"]) for item in items]
    except Exception:
        return []


def openai_chat(prompt: str, system: str = "You are a helpful emotional AI assistant.") -> str:
    if not prompt.strip():
        return "Type a message first."
    if openai_client is None:
        return "GPT is not configured. Add OPENAI_API_KEY to enable chat responses."
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=250,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        return f"OpenAI API error: {exc}"


def get_user_history(user: str) -> list[str]:
    if not DATA_FILE.exists():
        return []
    df = pd.read_csv(DATA_FILE)
    history = df[df["User"].astype(str) == str(user)]["RecommendedId"].dropna().astype(str).tolist()
    return history


def recommend_no_repeat(user: str, mood: str, last_n: int = 5) -> dict[str, Any]:
    catalog = catalog_dataframe()
    candidates = catalog[catalog["mood"] == mood].copy()
    if candidates.empty:
        candidates = catalog.copy()
    recent_ids = set(get_user_history(user)[-last_n:])
    stats = init_recommender_stats()
    merged = candidates.merge(stats, how="left", left_on="id", right_on="ItemId")
    merged["Exposures"] = merged["Exposures"].fillna(0)
    merged["Likes"] = merged["Likes"].fillna(0)
    merged["Skips"] = merged["Skips"].fillna(0)
    merged["like_ratio"] = merged["Likes"] / merged["Exposures"].clip(lower=1)
    merged["skip_ratio"] = merged["Skips"] / merged["Exposures"].clip(lower=1)
    merged["novelty_bonus"] = 1.0 / (merged["Exposures"] + 1.0)
    merged["playable_bonus"] = merged["playable"].astype(float) * 0.12
    merged["recent_penalty"] = merged["id"].isin(recent_ids).astype(float) * 0.50
    merged["score"] = (
        0.45 * merged["like_ratio"]
        + 0.35 * merged["novelty_bonus"]
        + merged["playable_bonus"]
        - 0.20 * merged["skip_ratio"]
        - merged["recent_penalty"]
        + np.random.uniform(0.0, 0.05, size=len(merged))
    )
    pool = merged[~merged["id"].isin(recent_ids)].copy()
    if pool.empty:
        pool = merged.copy()
    top_pool = pool.sort_values("score", ascending=False).head(min(8, len(pool)))
    if top_pool.empty:
        chosen = catalog.sample(1).iloc[0].to_dict()
    else:
        weights = np.maximum(top_pool["score"].to_numpy(dtype=float), 0.01)
        chosen = top_pool.sample(1, weights=weights).iloc[0].to_dict()
    update_stats(str(chosen["id"]), feedback="shown")
    return {column: chosen[column] for column in catalog.columns}


def coerce_to_pil_image(value: Any) -> Any:
    if Image is None or value is None:
        return None
    try:
        if isinstance(value, Image.Image):
            return value.convert("RGB")
        if isinstance(value, np.ndarray):
            array = np.asarray(value)
            if array.ndim == 2:
                return Image.fromarray(array.astype(np.uint8)).convert("RGB")
            return Image.fromarray(array.astype(np.uint8)).convert("RGB")
        if isinstance(value, (bytes, bytearray)):
            return Image.open(io.BytesIO(value)).convert("RGB")
        if isinstance(value, str) and Path(value).exists():
            return Image.open(value).convert("RGB")
        if isinstance(value, dict):
            if value.get("bytes"):
                return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
            path_value = value.get("path")
            if path_value and Path(path_value).exists():
                return Image.open(path_value).convert("RGB")
    except Exception:
        return None
    return None


def looks_like_label(value: Any) -> bool:
    return isinstance(value, (str, int, np.integer))


def looks_like_annotation_label(value: Any) -> bool:
    if isinstance(value, dict):
        return True
    if isinstance(value, (list, tuple)):
        return any(isinstance(item, (dict, str, int, np.integer, float)) for item in value[:5])
    return False


def detect_dataset_columns(example: dict[str, Any], features: Any = None) -> tuple[str | None, str | None]:
    image_column = None
    label_column = None
    for candidate in ["image", "img", "face", "jpg", "jpeg", "pixel_values"]:
        if candidate in example and coerce_to_pil_image(example[candidate]) is not None:
            image_column = candidate
            break
    if image_column is None:
        for key, value in example.items():
            if coerce_to_pil_image(value) is not None:
                image_column = key
                break
    for candidate in [
        "label",
        "emotion",
        "emotion_label",
        "class",
        "category",
        "target",
        "sentiment",
        "cls",
        "json",
    ]:
        if candidate in example and looks_like_label(example[candidate]):
            label_column = candidate
            break
    for candidate in ["annotations", "json"]:
        if label_column is None and candidate in example and looks_like_annotation_label(example[candidate]):
            label_column = candidate
    if label_column is None:
        for candidate in ["annotations", "json"]:
            if candidate in example and looks_like_annotation_label(example[candidate]):
                label_column = candidate
                break
    if label_column is None:
        for candidate in ["label", "emotion", "emotion_label", "class", "category", "target", "sentiment"]:
            if candidate in example and looks_like_label(example[candidate]):
                label_column = candidate
                break
    if label_column is None and features is not None:
        for key, feature in dict(features).items():
            if hasattr(feature, "names"):
                label_column = key
                break
    if label_column is None:
        for key, value in example.items():
            if key == image_column:
                continue
            if looks_like_label(value) or looks_like_annotation_label(value):
                label_column = key
                break
    return image_column, label_column


def parse_annotation_scores(value: Any) -> dict[str, float]:
    scores: dict[str, float] = {}

    def add_score(label_name: str, score: float) -> None:
        slug = sanitize_slug(label_name)
        if not slug:
            return
        scores[slug] = scores.get(slug, 0.0) + float(score)

    if isinstance(value, dict):
        named_label = None
        for key in ["label", "emotion", "name", "text", "class", "category"]:
            if key in value and isinstance(value[key], str):
                named_label = value[key]
                break
        if named_label:
            weight = 1.0
            for candidate in ["score", "value", "weight", "confidence", "intensity"]:
                raw_score = value.get(candidate)
                if isinstance(raw_score, (int, float, np.number, bool)):
                    weight = float(raw_score)
                    break
            add_score(str(named_label), weight)
        numeric_found = False
        for key, raw_score in value.items():
            if isinstance(raw_score, (int, float, np.number, bool)):
                numeric_found = True
                add_score(str(key), float(raw_score))
        if not numeric_found:
            for nested in value.values():
                nested_scores = parse_annotation_scores(nested)
                for key, raw_score in nested_scores.items():
                    add_score(key, raw_score)
    elif isinstance(value, (list, tuple)):
        for item in value:
            nested_scores = parse_annotation_scores(item)
            for key, raw_score in nested_scores.items():
                add_score(key, raw_score)
    elif isinstance(value, str):
        add_score(value, 1.0)
    elif isinstance(value, (int, np.integer)):
        add_score(str(int(value)), 1.0)
    return scores


def parse_emonet_json_label(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    metadata_items = value.get("metadata")
    if not isinstance(metadata_items, list):
        return None
    scores: dict[str, float] = {}
    for metadata in metadata_items:
        if not isinstance(metadata, dict):
            continue
        predictions = metadata.get("predictions")
        if not isinstance(predictions, dict):
            continue
        emotions = predictions.get("emotions")
        if not isinstance(emotions, dict):
            continue
        for emotion_name, payload in emotions.items():
            rating = None
            if isinstance(payload, dict):
                rating = payload.get("rating")
            elif isinstance(payload, (int, float, np.number)):
                rating = float(payload)
            if not isinstance(rating, (int, float, np.number)):
                continue
            rating_value = float(rating)
            if np.isnan(rating_value):
                continue
            scores[str(emotion_name)] = scores.get(str(emotion_name), 0.0) + rating_value
    if not scores:
        return None
    return max(scores.items(), key=lambda item: item[1])[0]


def label_name_from_value(value: Any, feature: Any = None, source_key: str | None = None) -> str | None:
    if value is None:
        return None
    label_map = DATASET_NUMERIC_LABEL_MAPS.get(source_key or "", {})
    if source_key == "EmoNet-Face-Big":
        parsed = parse_emonet_json_label(value)
        if parsed:
            return parsed
    if feature is not None and hasattr(feature, "names") and isinstance(value, (int, np.integer)):
        names = list(feature.names)
        index = int(value)
        if 0 <= index < len(names):
            return str(names[index])
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit() and label_map:
            mapped = label_map.get(int(stripped))
            if mapped:
                return mapped
        return stripped
    if isinstance(value, (int, np.integer)):
        if label_map:
            mapped = label_map.get(int(value))
            if mapped:
                return mapped
        return str(int(value))
    if isinstance(value, dict):
        for key in ["label", "emotion", "name", "text"]:
            if key in value and isinstance(value[key], str):
                return str(value[key])
        scores = parse_annotation_scores(value)
        if scores:
            return max(scores.items(), key=lambda item: item[1])[0]
    if isinstance(value, (list, tuple)):
        scores = parse_annotation_scores(value)
        if scores:
            return max(scores.items(), key=lambda item: item[1])[0]
        for item in value:
            label_name = label_name_from_value(item, feature=feature, source_key=source_key)
            if label_name:
                return label_name
    return None


def normalize_allowed_labels(
    discovered_labels: list[str] | None = None,
    allowed_labels: Iterable[str] | None = None,
) -> list[str]:
    labels: list[str] = []
    if discovered_labels:
        labels.extend(str(label) for label in discovered_labels)
    if allowed_labels:
        for label in allowed_labels:
            label_str = str(label)
            if label_str not in labels:
                labels.append(label_str)
    return labels


def choose_split(train_ratio: float, val_ratio: float) -> str:
    roll = random.random()
    if roll < train_ratio:
        return "train"
    if roll < train_ratio + val_ratio:
        return "val"
    return "test"


def save_stream_examples(
    dataset_iterable: Iterable[dict[str, Any]],
    output_dir: Path,
    limit: int,
    seed: int,
    assigned_split: str | None = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    max_classes: int | None = None,
    source_key: str | None = None,
    discovered_labels: list[str] | None = None,
    allowed_labels: Iterable[str] | None = None,
) -> dict[str, Any]:
    random.seed(seed)
    stats = {
        "saved": 0,
        "class_counts": {},
        "split_counts": {"train": 0, "val": 0, "test": 0},
        "image_column": None,
        "label_column": None,
        "classes_used": [],
    }
    features = getattr(dataset_iterable, "features", None)
    label_buffer = normalize_allowed_labels(discovered_labels=discovered_labels, allowed_labels=allowed_labels)
    allowed_set = set(str(label) for label in allowed_labels) if allowed_labels is not None else None
    for raw_index, example in enumerate(dataset_iterable):
        if stats["saved"] >= limit:
            break
        if not isinstance(example, dict):
            continue
        if stats["image_column"] is None or stats["label_column"] is None:
            image_column, label_column = detect_dataset_columns(example, features=features)
            stats["image_column"] = image_column
            stats["label_column"] = label_column
        image_column = stats["image_column"]
        label_column = stats["label_column"]
        if not image_column or not label_column:
            continue
        feature = dict(features).get(label_column) if features is not None else None
        label_name = label_name_from_value(
            example.get(label_column),
            feature=feature,
            source_key=source_key,
        )
        if not label_name:
            continue
        label_slug = sanitize_slug(label_name)
        if allowed_set is not None and label_slug not in allowed_set:
            continue
        if max_classes and label_slug not in label_buffer and len(label_buffer) >= max_classes:
            continue
        if label_slug not in label_buffer:
            label_buffer.append(label_slug)
        image = coerce_to_pil_image(example.get(image_column))
        if image is None:
            continue
        split_name = assigned_split or choose_split(train_ratio, val_ratio)
        class_dir = output_dir / split_name / label_slug
        class_dir.mkdir(parents=True, exist_ok=True)
        file_path = class_dir / f"{stats['saved']:06d}_{raw_index:06d}.jpg"
        image.convert("RGB").save(file_path, format="JPEG", quality=92)
        stats["saved"] += 1
        stats["split_counts"][split_name] += 1
        stats["class_counts"][label_slug] = stats["class_counts"].get(label_slug, 0) + 1
    stats["classes_used"] = sorted(label_buffer)
    return stats


def clear_dataset_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def fetch_dataset_splits(dataset_id: str) -> list[str]:
    if not HF_DATASETS_AVAILABLE:
        return []
    try:
        return list(get_dataset_split_names(dataset_id))
    except Exception:
        return []


def prepare_public_dataset(
    source_key: str,
    output_dir: Path,
    sample_limit: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
    max_classes: int | None = None,
) -> dict[str, Any]:
    if not HF_DATASETS_AVAILABLE:
        raise RuntimeError("Install `datasets` and `huggingface-hub` to enable automatic downloads.")
    source = DATASET_SOURCES[source_key]
    if not source.auto_download:
        raise RuntimeError("This source is listed for planning only. Use manual ZIP upload instead.")
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise RuntimeError("Train/val/test ratios must sum to 1.0.")
    clear_dataset_dir(output_dir)
    split_names = fetch_dataset_splits(source.dataset_id)
    normalized_split_map = {}
    for name in split_names:
        lowered = name.lower()
        if lowered == "train":
            normalized_split_map["train"] = name
        elif lowered in {"validation", "val"}:
            normalized_split_map["val"] = name
        elif lowered == "test":
            normalized_split_map["test"] = name

    manifest: dict[str, Any] = {
        "source_key": source.key,
        "source_label": source.label,
        "dataset_id": source.dataset_id,
        "created_at_utc": datetime.utcnow().isoformat(),
        "sample_limit": sample_limit,
        "max_classes": max_classes,
        "split_names": split_names,
        "split_counts": {"train": 0, "val": 0, "test": 0},
        "class_counts": {},
        "notes": source.notes,
    }
    selected_labels: list[str] = []

    def merge_stats(partial: dict[str, Any]) -> None:
        for split_name, count in partial["split_counts"].items():
            manifest["split_counts"][split_name] += int(count)
        for label_name, count in partial["class_counts"].items():
            manifest["class_counts"][label_name] = manifest["class_counts"].get(label_name, 0) + int(count)
        manifest["image_column"] = partial.get("image_column")
        manifest["label_column"] = partial.get("label_column")
        classes_used = normalize_allowed_labels(
            discovered_labels=partial.get("classes_used"),
            allowed_labels=selected_labels,
        )
        selected_labels[:] = classes_used
        manifest["classes_used"] = classes_used

    if {"train", "val", "test"}.issubset(normalized_split_map):
        quotas = {
            "train": max(1, int(sample_limit * train_ratio)),
            "val": max(1, int(sample_limit * val_ratio)),
            "test": max(1, sample_limit - int(sample_limit * train_ratio) - int(sample_limit * val_ratio)),
        }
        for canonical_split, quota in quotas.items():
            dataset_iterable = load_dataset(
                source.dataset_id,
                split=normalized_split_map[canonical_split],
                streaming=True,
            )
            split_stats = save_stream_examples(
                dataset_iterable=dataset_iterable,
                output_dir=output_dir,
                limit=quota,
                seed=seed,
                assigned_split=canonical_split,
                max_classes=max_classes,
                source_key=source_key,
                discovered_labels=selected_labels,
                allowed_labels=selected_labels or None,
            )
            merge_stats(split_stats)
    else:
        source_split = normalized_split_map.get("train") or (split_names[0] if split_names else "train")
        dataset_iterable = load_dataset(
            source.dataset_id,
            split=source_split,
            streaming=True,
        )
        split_stats = save_stream_examples(
            dataset_iterable=dataset_iterable,
            output_dir=output_dir,
            limit=sample_limit,
            seed=seed,
            assigned_split=None,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            max_classes=max_classes,
            source_key=source_key,
        )
        merge_stats(split_stats)

    if not manifest["class_counts"]:
        raise RuntimeError(
            "No images were saved. The dataset schema may differ from the generic loader assumptions."
        )
    manifest_path = output_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def normalize_dataset_root(output_dir: Path) -> None:
    special_names = {"train", "val", "validation", "test"}
    if any((output_dir / name).exists() for name in special_names):
        if (output_dir / "validation").exists() and not (output_dir / "val").exists():
            shutil.move(str(output_dir / "validation"), str(output_dir / "val"))
        return
    child_candidates = [path for path in output_dir.iterdir() if path.is_dir()]
    for candidate in child_candidates:
        if any((candidate / name).exists() for name in special_names):
            for child in candidate.iterdir():
                destination = output_dir / child.name
                if destination.exists():
                    continue
                shutil.move(str(child), str(destination))
            try:
                candidate.rmdir()
            except OSError:
                pass
            break
    if (output_dir / "validation").exists() and not (output_dir / "val").exists():
        shutil.move(str(output_dir / "validation"), str(output_dir / "val"))


def extract_uploaded_zip(zip_bytes: bytes, output_dir: Path = DATASET_DIR) -> dict[str, Any]:
    clear_dataset_dir(output_dir)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        archive.extractall(output_dir)
    normalize_dataset_root(output_dir)
    summary_df = dataset_directory_summary(output_dir)
    return {
        "output_dir": str(output_dir),
        "rows": int(summary_df["count"].sum()) if not summary_df.empty else 0,
        "classes": sorted(summary_df["class_name"].unique().tolist()) if not summary_df.empty else [],
    }


def dataset_directory_summary(root: Path = DATASET_DIR) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split_name in ["train", "val", "test"]:
        split_dir = root / split_name
        if not split_dir.exists():
            continue
        for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            count = sum(
                1
                for item in class_dir.iterdir()
                if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
            )
            rows.append({"split": split_name, "class_name": class_dir.name, "count": count})
    if not rows:
        return pd.DataFrame(columns=["split", "class_name", "count"])
    return pd.DataFrame(rows).sort_values(["split", "class_name"]).reset_index(drop=True)


def render_resource(item: dict[str, Any]) -> None:
    if st is None:
        return
    st.subheader(item["title"])
    st.caption(f"{item['mood'].title()} | {item['source']} | {item['resource_type']}")
    if bool(item.get("playable")) and "youtube.com/watch" in item.get("url", ""):
        st.video(item["url"])
    else:
        st.markdown(f"[Open recommended resource]({item['url']})")
    st.caption(item.get("offline_fallback", ""))


def low_ram_recommendation_text() -> str:
    return (
        "For the listed Windows 11 hardware (4 GB RAM), keep sample downloads below ~1200 images, "
        "batch size at 8, and training epochs at 1-3 for demo runs."
    )


def render_analysis_result(result: dict[str, Any], user: str) -> None:
    face_result = result["face_result"]
    scores = result["fusion_scores"]
    recommendation = result["recommendation"]
    st.success(
        "Detected moods - "
        f"Face: {face_result['mood']} ({face_result['method']}), "
        f"Emoji: {result['emoji_mood']}, "
        f"Voice: {result['voice_mood']}, "
        f"Text: {result['text_mood']}"
    )
    st.info(f"Fused mood: {result['fused_mood']}")
    st.json(scores)
    if face_result.get("overlay") is not None:
        st.image(face_result["overlay"], caption="Grad-CAM overlay", use_container_width=True)
    render_resource(recommendation)
    spotify_tracks = spotify_search_tracks(result["fused_mood"])
    if spotify_tracks:
        st.markdown("**Spotify picks**")
        for track_name, url in spotify_tracks:
            st.markdown(f"- [{track_name}]({url})")
    feedback_value = st.session_state.get("feedback_saved")
    left, right = st.columns(2)
    with left:
        if st.button(
            "👍 I liked this",
            key=f"like_{recommendation['id']}",
            disabled=feedback_value is not None,
        ):
            update_stats(recommendation["id"], feedback="liked")
            update_last_feedback(user, recommendation["id"], "liked")
            st.session_state["feedback_saved"] = "liked"
            st.success("Preference saved.")
    with right:
        if st.button(
            "👎 Skip / Not relevant",
            key=f"skip_{recommendation['id']}",
            disabled=feedback_value is not None,
        ):
            update_stats(recommendation["id"], feedback="skipped")
            update_last_feedback(user, recommendation["id"], "skipped")
            st.session_state["feedback_saved"] = "skipped"
            st.info("Feedback saved.")


def render_dataset_sources_table() -> None:
    rows = [
        {
            "Label": source.label,
            "Year": source.year,
            "Auto download": source.auto_download,
            "Recommended sample": source.recommended_sample,
            "Suggested max classes": source.suggested_max_classes,
            "License": source.license_name,
            "Notes": source.notes,
        }
        for source in DATASET_SOURCES.values()
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def run_cli() -> int:
    parser = argparse.ArgumentParser(description="CEI-ALOS utility entrypoints")
    parser.add_argument("--export-catalog", action="store_true", help="Export the resource catalog CSV.")
    parser.add_argument(
        "--catalog-path",
        default=str(CATALOG_EXPORT_PATH),
        help="Path for exported resource catalog CSV.",
    )
    args, _ = parser.parse_known_args()
    if args.export_catalog:
        target = export_resource_catalog_csv(Path(args.catalog_path))
        print(f"Catalog exported to {target}")
        return 0
    return -1


def main() -> None:
    cli_result = run_cli()
    if cli_result == 0:
        return

    require_streamlit()
    ensure_runtime_files()

    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(f"🧠 {APP_TITLE}")
    st.markdown(
        "A single-file Streamlit app for multi-modal emotion detection, transfer learning, "
        "Grad-CAM explainability, adaptive recommendation, Digital Emotional Twin logging, "
        "and a 100+ item structured resource catalog."
    )

    export_resource_catalog_csv()

    if "analysis_result" not in st.session_state:
        st.session_state["analysis_result"] = None
    if "captured_voice_text" not in st.session_state:
        st.session_state["captured_voice_text"] = ""
    if "feedback_saved" not in st.session_state:
        st.session_state["feedback_saved"] = None

    with st.sidebar:
        st.header("Inputs")
        user = st.text_input("Username", value="guest_user")
        emoji = st.select_slider("Emoji signal", options=list(EMOJI_MAP.keys()), value="😊")
        camera_file = st.camera_input("Capture face (optional)")
        upload_file = st.file_uploader("Or upload a face image", type=["png", "jpg", "jpeg"])
        face_bytes = None
        if upload_file is not None:
            face_bytes = upload_file.getvalue()
        elif camera_file is not None:
            face_bytes = camera_file.getvalue()

        st.markdown("**Voice input**")
        live_mic = st.checkbox("Enable short microphone capture")
        if live_mic and st.button("Record 5s voice sample"):
            with st.spinner("Recording..."):
                st.session_state["captured_voice_text"] = record_voice_text(timeout=5)
        typed_voice_text = st.text_input(
            "Voice/text fallback",
            value=st.session_state.get("captured_voice_text", ""),
        )
        user_text = st.text_area("Context text", value="")
        st.caption(low_ram_recommendation_text())

    analysis_tab, training_tab, twin_tab, setup_tab = st.tabs(
        ["Analyze & Recommend", "Dataset + Training", "Twin + Catalog", "Setup + APIs"]
    )

    with analysis_tab:
        st.subheader("Multi-modal fusion")
        if st.button("🚀 Analyze & Recommend", type="primary"):
            voice_text = typed_voice_text or st.session_state.get("captured_voice_text", "")
            face_result = predict_face_emotion(face_bytes)
            emoji_mood = EMOJI_MAP.get(emoji, "calm")
            voice_mood = simple_text_to_mood(voice_text)
            text_mood = simple_text_to_mood(user_text)
            fused_mood, fusion_scores = emotion_fusion(
                face_result["mood"],
                emoji_mood,
                voice_mood,
                text_mood,
            )
            recommendation = recommend_no_repeat(user, fused_mood, last_n=5)
            log_interaction(
                user=user,
                face_label=face_result["label"],
                face_mood=face_result["mood"],
                emoji_mood=emoji_mood,
                voice_mood=voice_mood,
                text_mood=text_mood,
                fused_mood=fused_mood,
                recommendation=recommendation,
            )
            st.session_state["analysis_result"] = {
                "face_result": face_result,
                "emoji_mood": emoji_mood,
                "voice_mood": voice_mood,
                "text_mood": text_mood,
                "fused_mood": fused_mood,
                "fusion_scores": fusion_scores,
                "recommendation": recommendation,
            }
            st.session_state["feedback_saved"] = None
        result = st.session_state.get("analysis_result")
        if result:
            render_analysis_result(result, user=user)
        st.markdown("---")
        st.subheader("Chat with Emotional AI")
        chat_input = st.text_input("Say something to the assistant", value="")
        if st.button("Send Chat"):
            st.write(openai_chat(chat_input))

    with training_tab:
        st.subheader("Automatic dataset preparation (Option A)")
        render_dataset_sources_table()
        source_key = st.selectbox(
            "Public dataset source",
            options=list(DATASET_SOURCES.keys()),
            format_func=lambda key: DATASET_SOURCES[key].label,
        )
        selected_source = DATASET_SOURCES[source_key]
        st.info(selected_source.description)
        left, middle, right = st.columns(3)
        with left:
            sample_limit = st.slider(
                "Sample size",
                min_value=200,
                max_value=2500,
                value=max(200, selected_source.recommended_sample),
                step=100,
            )
        with middle:
            max_classes = st.slider(
                "Max classes to keep",
                min_value=2,
                max_value=20,
                value=max(2, selected_source.suggested_max_classes),
                step=1,
            )
        with right:
            batch_size = st.select_slider("Training batch size", options=[4, 8, 12, 16], value=8)
        st.caption(
            "This downloader uses sample-mode and local train/val/test folders so it remains practical "
            "for low-RAM machines."
        )
        if st.button("Prepare public dataset"):
            if not selected_source.auto_download:
                st.warning(selected_source.notes)
            else:
                with st.spinner("Downloading a sampled subset and building train/val/test folders..."):
                    try:
                        manifest = prepare_public_dataset(
                            source_key=source_key,
                            output_dir=DATASET_DIR,
                            sample_limit=sample_limit,
                            train_ratio=0.70,
                            val_ratio=0.15,
                            test_ratio=0.15,
                            seed=42,
                            max_classes=max_classes,
                        )
                        st.success("Dataset prepared successfully.")
                        st.json(manifest)
                    except Exception as exc:
                        st.error(f"Dataset preparation failed: {exc}")
        st.markdown("**Manual ZIP import**")
        manual_zip = st.file_uploader(
            "Upload a ZIP with train/val/test or train/validation/test folders",
            type=["zip"],
            key="manual_dataset_zip",
        )
        if manual_zip is not None and st.button("Extract uploaded ZIP"):
            try:
                result = extract_uploaded_zip(manual_zip.getvalue(), output_dir=DATASET_DIR)
                st.success(f"Dataset extracted to {result['output_dir']}")
                st.json(result)
            except Exception as exc:
                st.error(f"ZIP extraction failed: {exc}")

        summary_df = dataset_directory_summary(DATASET_DIR)
        if not summary_df.empty:
            st.markdown("**Current dataset summary**")
            st.dataframe(summary_df, use_container_width=True)
        else:
            st.info("No dataset prepared yet.")

        st.markdown("---")
        st.subheader("MobileNetV2 transfer learning")
        if TF_AVAILABLE:
            build_classes = int(summary_df["class_name"].nunique()) if not summary_df.empty else 3
            build_classes = max(2, build_classes)
            if st.button("Build untrained MobileNetV2 model"):
                model = build_transfer_model(build_classes)
                save_model_metadata([f"class_{i}" for i in range(build_classes)], extra={"stage": "untrained"})
                model.save(MODEL_PATH)
                load_saved_model.cache_clear()
                st.success(f"Untrained model saved to {MODEL_PATH.name}")
                st.code(format_model_summary(model), language="text")
            epochs = st.slider("Demo training epochs", min_value=1, max_value=6, value=3)
            if st.button("Train current dataset"):
                try:
                    with st.spinner("Training MobileNetV2 on the prepared dataset..."):
                        train_ds, val_ds, test_ds, class_names = build_training_datasets(DATASET_DIR, batch_size=batch_size)
                        model = build_transfer_model(len(class_names))
                        history = model.fit(
                            train_ds,
                            validation_data=val_ds,
                            epochs=epochs,
                            verbose=1,
                        )
                        model.save(MODEL_PATH)
                        save_model_metadata(
                            class_names,
                            extra={
                                "dataset_root": str(DATASET_DIR),
                                "epochs": epochs,
                                "batch_size": batch_size,
                            },
                        )
                        load_saved_model.cache_clear()
                        st.success(f"Training complete. Saved model to {MODEL_PATH}")
                        history_df = pd.DataFrame(history.history)
                        st.dataframe(history_df, use_container_width=True)
                        if not history_df.empty:
                            st.line_chart(history_df)
                        eval_dataset = test_ds or val_ds
                        if eval_dataset is not None:
                            evaluation = evaluate_model_on_dataset(model, eval_dataset, class_names)
                            st.markdown("**Evaluation metrics**")
                            st.json(evaluation)
                        else:
                            st.info("No validation/test split found, so evaluation was skipped.")
                except Exception as exc:
                    st.error(f"Training failed: {exc}")
        else:
            st.warning(
                "TensorFlow is not installed, so training is disabled. The app will still use heuristic face mood detection."
            )

    with twin_tab:
        st.subheader("Digital Emotional Twin")
        if DATA_FILE.exists():
            logs_df = pd.read_csv(DATA_FILE)
            st.dataframe(logs_df, use_container_width=True)
            if "FusedMood" in logs_df.columns and not logs_df.empty:
                st.bar_chart(logs_df["FusedMood"].value_counts())
            st.download_button(
                "Download twin log CSV",
                DATA_FILE.read_bytes(),
                file_name=DATA_FILE.name,
                mime="text/csv",
            )
        else:
            st.info("No interaction logs yet.")

        st.markdown("---")
        st.subheader("Recommender stats")
        stats_df = init_recommender_stats()
        st.dataframe(stats_df, use_container_width=True)
        if not stats_df.empty:
            score_df = stats_df.copy()
            score_df["PreferenceScore"] = score_df["Likes"] / (score_df["Exposures"].clip(lower=1))
            st.bar_chart(score_df.set_index("ItemId")["PreferenceScore"])

        st.markdown("---")
        st.subheader("Structured resource catalog (Option B)")
        catalog_df = catalog_dataframe()
        mood_filter = st.selectbox("Catalog mood filter", options=["all"] + MOOD_CHOICES)
        source_filter = st.selectbox(
            "Catalog source filter",
            options=["all", "YouTube", "Spotify", "YouTube Music"],
        )
        filtered_catalog = catalog_df.copy()
        if mood_filter != "all":
            filtered_catalog = filtered_catalog[filtered_catalog["mood"] == mood_filter]
        if source_filter != "all":
            filtered_catalog = filtered_catalog[filtered_catalog["source"] == source_filter]
        st.metric("Catalog items", len(catalog_df))
        st.dataframe(filtered_catalog, use_container_width=True)
        st.download_button(
            "Download resource catalog CSV",
            catalog_df.to_csv(index=False).encode("utf-8"),
            file_name=CATALOG_EXPORT_PATH.name,
            mime="text/csv",
        )
        if st.button("Save catalog CSV into the project folder"):
            target = export_resource_catalog_csv()
            st.success(f"Catalog exported to {target}")

    with setup_tab:
        st.subheader("Developer setup")
        st.markdown(
            textwrap.dedent(
                """
                1. **Create a virtual environment**
                   - Windows PowerShell:
                     ```powershell
                     python -m venv .venv
                     .\\.venv\\Scripts\\Activate.ps1
                     ```
                   - macOS/Linux:
                     ```bash
                     python -m venv .venv
                     source .venv/bin/activate
                     ```

                2. **Install dependencies**
                   ```bash
                   pip install -r requirements.txt
                   ```

                3. **API keys (optional)**
                   - OpenAI: set `OPENAI_API_KEY`
                   - Spotify: set `SPOTIPY_CLIENT_ID` and `SPOTIPY_CLIENT_SECRET`

                4. **Run the app**
                   ```bash
                   streamlit run app.py
                   ```
                """
            )
        )
        st.markdown("**Low-RAM Windows 11 recommendation**")
        st.info(low_ram_recommendation_text())
        st.markdown("**Android / Windows install recommendation**")
        st.markdown(
            textwrap.dedent(
                """
                The zero-cost path is still the best one:

                - Host the Streamlit app on a public URL.
                - On Android, open it in Chrome and use **Add to Home Screen**.
                - On Windows 11, open it in Edge/Chrome and choose **Install app**.

                This gives you a PWA-like installable experience without paying for wrapper APK services.
                If you later need a real APK, use Kivy/Buildozer or Chaquopy in a Linux build environment.
                """
            )
        )
        st.markdown("**Notes about recent datasets**")
        st.markdown(
            textwrap.dedent(
                """
                - The automatic downloader focuses on recent 2024-2025 public Hugging Face sources.
                - MER2023/MER2024 are listed for project documentation, but not auto-downloaded here due to likely access restrictions.
                - RAF-DB and CK+ remain valid manual-import options if you already have local approved copies, even though they are older than 2023.
                """
            )
        )


if __name__ == "__main__":
    main()
