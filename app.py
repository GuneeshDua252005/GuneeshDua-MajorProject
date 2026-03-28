#!/usr/bin/env python3
"""
CEI-ALOS: Cognitive Emotion Intelligence & Adaptive Lifestyle OS

Single-file Flask + PWA application that bundles:
- Multi-modal emotion analysis (face heuristics / optional CNN, voice text, emoji, free text)
- Reinforcement-style recommendation logic that avoids repetition
- Digital Emotional Twin logging and analytics
- OpenAI emotional chat and optional image-aware assistance
- Spotify search integration
- 100+ structured wellness/content resources
- Optional CNN transfer-learning lab with evaluation metrics and Grad-CAM explainability
- Installable Android/Windows substitute for paid APK wrappers via PWA

Minimal dependency for the web app: Flask
Optional ML dependencies for the CNN lab:
    numpy, Pillow, tensorflow, scikit-learn
"""

from __future__ import annotations

import base64
import io
import json
import os
import random
import secrets
import shutil
import sqlite3
import textwrap
import time
import traceback
import uuid
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from flask import Flask, Response, jsonify, render_template_string, request

try:
    import numpy as np
except Exception:  # pragma: no cover - optional
    np = None

try:
    from PIL import Image as PILImage
except Exception:  # pragma: no cover - optional
    PILImage = None

try:
    import tensorflow as tf
except Exception:  # pragma: no cover - optional
    tf = None

try:
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        precision_recall_fscore_support,
        roc_auc_score,
    )
    from sklearn.preprocessing import label_binarize
except Exception:  # pragma: no cover - optional
    classification_report = None
    confusion_matrix = None
    precision_recall_fscore_support = None
    roc_auc_score = None
    label_binarize = None


APP_TITLE = "CEI-ALOS"
APP_VERSION = "1.0.0"
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "cei_alos_data"
DATASET_DIR = DATA_DIR / "datasets"
MODEL_DIR = DATA_DIR / "models"
DB_PATH = DATA_DIR / "cei_alos.db"
CONFIG_PATH = DATA_DIR / "runtime_config.json"

MOODS = ["happy", "calm", "sad", "energetic", "focus", "anxious"]
EMOJI_MAP = {
    "😀": "happy",
    "😊": "happy",
    "😍": "happy",
    "🤩": "happy",
    "😌": "calm",
    "😴": "calm",
    "😐": "calm",
    "😢": "sad",
    "😭": "sad",
    "💔": "sad",
    "😡": "energetic",
    "🔥": "energetic",
    "💪": "energetic",
    "😰": "anxious",
    "😟": "anxious",
    "🧠": "focus",
    "📚": "focus",
}

TEXT_MOOD_LEXICON = {
    "happy": [
        "happy",
        "great",
        "good",
        "awesome",
        "joy",
        "excited",
        "grateful",
        "smile",
        "cheerful",
        "positive",
    ],
    "calm": [
        "calm",
        "peace",
        "relaxed",
        "steady",
        "okay",
        "fine",
        "balanced",
        "grounded",
    ],
    "sad": [
        "sad",
        "cry",
        "down",
        "lonely",
        "hurt",
        "tired",
        "empty",
        "low",
        "upset",
        "heartbroken",
    ],
    "energetic": [
        "angry",
        "mad",
        "pumped",
        "hype",
        "gym",
        "workout",
        "energy",
        "fired up",
        "motivated",
    ],
    "focus": [
        "focus",
        "study",
        "work",
        "coding",
        "exam",
        "deadline",
        "concentrate",
        "productivity",
        "project",
    ],
    "anxious": [
        "anxious",
        "stress",
        "stressed",
        "panic",
        "worried",
        "overthinking",
        "nervous",
        "fear",
        "pressure",
        "burnout",
    ],
}

ARCHITECTURE_PIPELINE = [
    "Face (heuristic / optional CNN transfer learning)",
    "Voice transcript via browser speech recognition",
    "Emoji + free-text affect signals",
    "Multi-modal weighted fusion engine",
    "History-aware reinforcement recommendation layer",
    "YouTube + Spotify + wellness recommendation engine",
    "Digital Emotional Twin logging + analytics",
    "Optional CNN evaluation + Grad-CAM explainability",
]

VIVA_STATEMENT = (
    "Our system implements a multi-modal reinforcement learning-based emotional "
    "intelligence OS integrating facial analysis, voice sentiment NLP, emoji-based "
    "interaction, Spotify and YouTube recommendation engines, and GPT-based emotional "
    "AI, forming a personalized digital emotional twin with adaptive lifestyle recommendations."
)

SETUP_COMMANDS = {
    "Windows CMD": [
        'setx OPENAI_API_KEY "your_openai_key"',
        'setx SPOTIFY_CLIENT_ID "your_spotify_client_id"',
        'setx SPOTIFY_CLIENT_SECRET "your_spotify_client_secret"',
    ],
    "Windows PowerShell": [
        '[System.Environment]::SetEnvironmentVariable("OPENAI_API_KEY","your_openai_key","User")',
        '[System.Environment]::SetEnvironmentVariable("SPOTIFY_CLIENT_ID","your_spotify_client_id","User")',
        '[System.Environment]::SetEnvironmentVariable("SPOTIFY_CLIENT_SECRET","your_spotify_client_secret","User")',
    ],
    "Linux / macOS": [
        'export OPENAI_API_KEY="your_openai_key"',
        'export SPOTIFY_CLIENT_ID="your_spotify_client_id"',
        'export SPOTIFY_CLIENT_SECRET="your_spotify_client_secret"',
    ],
}

PWA_GUIDE = {
    "android": [
        "Run `python app.py`, open the local or deployed URL in Chrome.",
        "Tap the browser menu and choose `Add to Home screen` / `Install app`.",
        "The project becomes an installable mobile app without a paid APK wrapper.",
    ],
    "windows11": [
        "Open the app URL in Edge or Chrome on Windows 11.",
        "Click the install icon in the address bar or menu -> Apps -> Install this site as an app.",
        "The project launches like a desktop application and works for major-project demos.",
    ],
}

SPOTIFY_TOKEN_CACHE: Dict[str, Any] = {"access_token": "", "expires_at": 0.0}
MODEL_CACHE: Dict[str, Any] = {"path": "", "model": None, "metadata": None}


def ensure_runtime_dirs() -> None:
    for path in (DATA_DIR, DATASET_DIR, MODEL_DIR):
        path.mkdir(parents=True, exist_ok=True)


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_json_loads(raw: Optional[str], default: Any) -> Any:
    if not raw:
        return default
    try:
        return json.loads(raw)
    except Exception:
        return default


def load_runtime_config() -> Dict[str, str]:
    if not CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_runtime_config(new_values: Dict[str, str]) -> Dict[str, str]:
    current = load_runtime_config()
    for key in ("OPENAI_API_KEY", "SPOTIFY_CLIENT_ID", "SPOTIFY_CLIENT_SECRET"):
        value = (new_values.get(key) or "").strip()
        if value:
            current[key] = value
    CONFIG_PATH.write_text(json.dumps(current, indent=2), encoding="utf-8")
    return current


def get_secret(name: str) -> str:
    return os.getenv(name, "").strip() or load_runtime_config().get(name, "").strip()


def mask_secret(value: str) -> str:
    if not value:
        return "Not configured"
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


def get_dependency_status() -> Dict[str, bool]:
    return {
        "flask": True,
        "numpy": np is not None,
        "pillow": PILImage is not None,
        "tensorflow": tf is not None,
        "scikit_learn": classification_report is not None,
    }


def config_status() -> Dict[str, Any]:
    openai_key = get_secret("OPENAI_API_KEY")
    spotify_client_id = get_secret("SPOTIFY_CLIENT_ID")
    spotify_client_secret = get_secret("SPOTIFY_CLIENT_SECRET")

    def source_for(name: str) -> str:
        if os.getenv(name):
            return "environment"
        if load_runtime_config().get(name):
            return "saved-local"
        return "missing"

    return {
        "openai": {
            "configured": bool(openai_key),
            "masked": mask_secret(openai_key),
            "source": source_for("OPENAI_API_KEY"),
        },
        "spotify": {
            "configured": bool(spotify_client_id and spotify_client_secret),
            "masked_client_id": mask_secret(spotify_client_id),
            "masked_client_secret": mask_secret(spotify_client_secret),
            "source": (
                "environment"
                if os.getenv("SPOTIFY_CLIENT_ID") or os.getenv("SPOTIFY_CLIENT_SECRET")
                else ("saved-local" if load_runtime_config().get("SPOTIFY_CLIENT_ID") else "missing")
            ),
        },
        "dependencies": get_dependency_status(),
    }


def db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with db_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS emotion_events (
                id TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                created_at TEXT NOT NULL,
                face_mood TEXT,
                face_source TEXT,
                voice_mood TEXT,
                emoji_mood TEXT,
                text_mood TEXT,
                final_mood TEXT,
                confidence REAL,
                primary_resource_title TEXT,
                primary_resource_url TEXT,
                spotify_json TEXT,
                notes_json TEXT
            );

            CREATE TABLE IF NOT EXISTS trained_models (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                model_path TEXT NOT NULL,
                labels_json TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                base_model TEXT NOT NULL,
                image_size INTEGER NOT NULL
            );
            """
        )


def youtube_resource(
    title: str,
    query: Optional[str] = None,
    url: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "title": title,
        "provider": "YouTube",
        "type": "video",
        "tags": tags or [],
        "url": url or f"https://www.youtube.com/results?search_query={quote(query or title)}",
    }


def spotify_resource(title: str, query: str, tags: Optional[List[str]] = None) -> Dict[str, Any]:
    return {
        "title": title,
        "provider": "Spotify",
        "type": "audio",
        "tags": tags or [],
        "url": f"https://open.spotify.com/search/{quote(query)}",
    }


def build_resource_catalog() -> Dict[str, List[Dict[str, Any]]]:
    seeds: Dict[str, List[Dict[str, Any]]] = {
        "happy": [
            youtube_resource("Featured happy video 1", url="https://www.youtube.com/watch?v=d-diB65scQU", tags=["happy"]),
            youtube_resource("Featured happy video 2", url="https://www.youtube.com/watch?v=OPf0YbXqDm0", tags=["happy"]),
            youtube_resource("Justin Timberlake - Can't Stop The Feeling", "Justin Timberlake Can't Stop The Feeling official video", tags=["happy"]),
            youtube_resource("OneRepublic - Good Life", "OneRepublic Good Life official video", tags=["happy"]),
            youtube_resource("Imagine Dragons - On Top of the World", "Imagine Dragons On Top Of The World official video", tags=["happy"]),
            youtube_resource("Katy Perry - Firework", "Katy Perry Firework official video", tags=["happy"]),
            youtube_resource("Black Eyed Peas - I Gotta Feeling", "Black Eyed Peas I Gotta Feeling official video", tags=["happy"]),
            youtube_resource("Walk The Moon - Shut Up And Dance", "Walk The Moon Shut Up and Dance official video", tags=["happy"]),
            youtube_resource("Lizzo - Good As Hell", "Lizzo Good As Hell official video", tags=["happy"]),
            youtube_resource("American Authors - Best Day Of My Life", "American Authors Best Day Of My Life official video", tags=["happy"]),
            youtube_resource("Kool & The Gang - Celebration", "Kool and the Gang Celebration official video", tags=["happy"]),
            youtube_resource("Fun upbeat dance practice", "upbeat dance practice playlist", tags=["happy"]),
            youtube_resource("10-minute gratitude meditation", "10 minute gratitude meditation", tags=["happy"]),
            youtube_resource("Positive affirmations", "positive affirmations confidence happiness", tags=["happy"]),
            youtube_resource("Feel-good pop mix", "feel good pop mix playlist", tags=["happy"]),
            youtube_resource("Morning motivation", "morning motivation positivity", tags=["happy"]),
            spotify_resource("Spotify Happy Hits", "happy hits", tags=["happy"]),
            spotify_resource("Spotify Mood Booster", "mood booster", tags=["happy"]),
        ],
        "sad": [
            youtube_resource("Healing meditation session", url="https://www.youtube.com/watch?v=inpok4MKVLM", tags=["sad"]),
            youtube_resource("Gentle emotional healing session", url="https://www.youtube.com/watch?v=2XU0oxnq2qU", tags=["sad"]),
            youtube_resource("RAIN meditation for sadness", "RAIN meditation sadness", tags=["sad"]),
            youtube_resource("Comfort piano music", "comfort piano music for sadness", tags=["sad"]),
            youtube_resource("Self-compassion meditation", "self compassion meditation guided", tags=["sad"]),
            youtube_resource("Therapeutic journaling prompt session", "journaling prompts for sadness guided", tags=["sad"]),
            youtube_resource("Mindfulness for grief", "mindfulness for grief and loss", tags=["sad"]),
            youtube_resource("Rainy lofi for recovery", "rainy lofi for healing", tags=["sad"]),
            youtube_resource("Breathing for emotional release", "breathing exercise emotional release", tags=["sad"]),
            youtube_resource("Sleep story for low mood", "sleep story comfort low mood", tags=["sad"]),
            youtube_resource("5-minute reset after crying", "5 minute reset after crying meditation", tags=["sad"]),
            youtube_resource("Gentle yoga for stress relief", "gentle yoga for stress relief sadness", tags=["sad"]),
            youtube_resource("Hopeful worship / reflective music", "hopeful reflective music healing", tags=["sad"]),
            youtube_resource("Therapy-style grounding", "therapy grounding exercise sadness", tags=["sad"]),
            youtube_resource("Ocean sounds for calm recovery", "ocean sounds emotional healing", tags=["sad"]),
            youtube_resource("Mindset shift for difficult days", "mindset shift difficult day motivation", tags=["sad"]),
            spotify_resource("Spotify Sad Bops", "sad bops", tags=["sad"]),
            spotify_resource("Spotify Life Sucks", "life sucks", tags=["sad"]),
        ],
        "calm": [
            youtube_resource("Lofi calm live stream", url="https://www.youtube.com/watch?v=5qap5aO4i9A", tags=["calm"]),
            youtube_resource("Yoga flow for calm", url="https://www.youtube.com/watch?v=v7AYKMP6rOE", tags=["calm"]),
            youtube_resource("Nature sound immersion", "nature sounds for deep calm", tags=["calm"]),
            youtube_resource("10-minute box breathing", "10 minute box breathing", tags=["calm"]),
            youtube_resource("Guided body scan meditation", "guided body scan meditation", tags=["calm"]),
            youtube_resource("Slow stretching for relaxation", "slow stretching relaxation routine", tags=["calm"]),
            youtube_resource("Forest ambience", "forest ambience relaxation", tags=["calm"]),
            youtube_resource("Soft rain for sleep and calm", "soft rain relaxation sleep", tags=["calm"]),
            youtube_resource("Tea-time piano music", "soft piano relaxation", tags=["calm"]),
            youtube_resource("Yoga nidra", "yoga nidra guided meditation", tags=["calm"]),
            youtube_resource("Grounding meditation", "grounding meditation 5 minutes", tags=["calm"]),
            youtube_resource("Calm classical mix", "calm classical playlist", tags=["calm"]),
            youtube_resource("Mindful walking meditation", "mindful walking meditation", tags=["calm"]),
            youtube_resource("Breathwork for nervous system reset", "nervous system reset breathwork", tags=["calm"]),
            youtube_resource("Peaceful evening playlist", "peaceful evening playlist calm", tags=["calm"]),
            youtube_resource("Lo-fi reading room", "lofi reading room calm", tags=["calm"]),
            spotify_resource("Spotify Peaceful Piano", "peaceful piano", tags=["calm"]),
            spotify_resource("Spotify Calm Vibes", "calm vibes", tags=["calm"]),
        ],
        "energetic": [
            youtube_resource("Featured energetic video 1", url="https://www.youtube.com/watch?v=HgzGwKwLmgM", tags=["energetic"]),
            youtube_resource("Featured energetic video 2", url="https://www.youtube.com/watch?v=ml6cT4AZdqI", tags=["energetic"]),
            youtube_resource("Workout EDM mix", "EDM workout mix motivation", tags=["energetic"]),
            youtube_resource("HIIT motivation", "HIIT workout motivation music", tags=["energetic"]),
            youtube_resource("Gym warm-up mix", "gym warm up playlist", tags=["energetic"]),
            youtube_resource("Push-day hype music", "push day hype music", tags=["energetic"]),
            youtube_resource("Morning power routine", "morning power routine motivation", tags=["energetic"]),
            youtube_resource("Beast mode speech", "beast mode motivation speech", tags=["energetic"]),
            youtube_resource("Tabata cardio playlist", "tabata cardio workout music", tags=["energetic"]),
            youtube_resource("Running motivation", "running motivation playlist", tags=["energetic"]),
            youtube_resource("Athletic focus mix", "athletic focus high energy mix", tags=["energetic"]),
            youtube_resource("Dance cardio workout", "dance cardio workout follow along", tags=["energetic"]),
            youtube_resource("Sports highlight motivation", "sports highlight motivation speech", tags=["energetic"]),
            youtube_resource("Rock power mix", "rock power mix playlist", tags=["energetic"]),
            youtube_resource("Confidence activation", "confidence activation affirmations", tags=["energetic"]),
            youtube_resource("Pre-exam energy reset", "quick energy reset for exam", tags=["energetic"]),
            spotify_resource("Spotify Beast Mode", "beast mode", tags=["energetic"]),
            spotify_resource("Spotify Power Workout", "power workout", tags=["energetic"]),
        ],
        "focus": [
            youtube_resource("Pomodoro deep work timer", "pomodoro deep work timer", tags=["focus"]),
            youtube_resource("Lofi for coding", "lofi coding music stream", tags=["focus"]),
            youtube_resource("Deep focus binaural beats", "deep focus binaural beats", tags=["focus"]),
            youtube_resource("Classical study music", "classical study music concentration", tags=["focus"]),
            youtube_resource("ADHD focus ambient mix", "ADHD focus ambient music", tags=["focus"]),
            youtube_resource("Library ambience", "library ambience for studying", tags=["focus"]),
            youtube_resource("Morning study with me", "study with me 2 hour focus", tags=["focus"]),
            youtube_resource("Calm concentration piano", "concentration piano instrumental", tags=["focus"]),
            youtube_resource("Low-distraction white noise", "white noise for concentration", tags=["focus"]),
            youtube_resource("Exam revision music", "exam revision background music", tags=["focus"]),
            youtube_resource("Coding ambient synthwave", "coding synthwave focus mix", tags=["focus"]),
            youtube_resource("Mind mapping productivity talk", "mind mapping productivity students", tags=["focus"]),
            youtube_resource("Deep work strategy", "deep work strategy concentration", tags=["focus"]),
            youtube_resource("Note-taking masterclass", "note taking masterclass students", tags=["focus"]),
            youtube_resource("Project planning sprint", "project planning sprint technique", tags=["focus"]),
            youtube_resource("Task batching technique", "task batching productivity guide", tags=["focus"]),
            spotify_resource("Spotify Deep Focus", "deep focus", tags=["focus"]),
            spotify_resource("Spotify Brain Food", "brain food", tags=["focus"]),
        ],
        "anxious": [
            youtube_resource("Guided box breathing", "guided box breathing anxiety", tags=["anxious"]),
            youtube_resource("5-4-3-2-1 grounding", "5 4 3 2 1 grounding exercise anxiety", tags=["anxious"]),
            youtube_resource("Panic attack relief", "panic attack relief now breathing", tags=["anxious"]),
            youtube_resource("Nervous system calming meditation", "nervous system calming meditation", tags=["anxious"]),
            youtube_resource("CBT anxiety reframing", "CBT anxiety reframing guide", tags=["anxious"]),
            youtube_resource("Short vagus nerve reset", "vagus nerve reset exercise", tags=["anxious"]),
            youtube_resource("Chair yoga for stress", "chair yoga stress relief", tags=["anxious"]),
            youtube_resource("Guided sleep for anxious mind", "sleep meditation anxious mind", tags=["anxious"]),
            youtube_resource("Exam anxiety reset", "exam anxiety breathing exercise", tags=["anxious"]),
            youtube_resource("Overthinking stop technique", "overthinking stop technique guided", tags=["anxious"]),
            youtube_resource("Safe-space visualization", "safe place visualization anxiety", tags=["anxious"]),
            youtube_resource("Breathing with hand tracking", "breathing hand tracing anxiety", tags=["anxious"]),
            youtube_resource("Somatic shaking release", "somatic shaking exercise stress", tags=["anxious"]),
            youtube_resource("Rain ambience for pressure relief", "rain ambience stress relief", tags=["anxious"]),
            youtube_resource("Gentle affirmations for worry", "affirmations for worry anxiety", tags=["anxious"]),
            youtube_resource("Mindful tea break", "mindful tea break anxiety relief", tags=["anxious"]),
            spotify_resource("Spotify Anxiety Relief", "anxiety relief", tags=["anxious"]),
            spotify_resource("Spotify Stress Relief", "stress relief", tags=["anxious"]),
        ],
    }
    return seeds


RESOURCE_CATALOG = build_resource_catalog()
RESOURCE_TOTAL = sum(len(items) for items in RESOURCE_CATALOG.values())


def normalize_mood(label: Optional[str]) -> str:
    value = (label or "").strip().lower()
    if value in MOODS:
        return value

    alias_map = {
        "joy": "happy",
        "positive": "happy",
        "smile": "happy",
        "neutral": "calm",
        "relaxed": "calm",
        "peaceful": "calm",
        "anger": "energetic",
        "angry": "energetic",
        "excited": "energetic",
        "hype": "energetic",
        "fear": "anxious",
        "stress": "anxious",
        "worried": "anxious",
        "depress": "sad",
        "focus": "focus",
        "study": "focus",
    }
    for key, normalized in alias_map.items():
        if key in value:
            return normalized
    return "calm"


def mood_from_text(text: str) -> str:
    lowered = (text or "").lower()
    if not lowered.strip():
        return "calm"
    scores: Counter[str] = Counter()
    for mood, keywords in TEXT_MOOD_LEXICON.items():
        for keyword in keywords:
            if keyword in lowered:
                scores[mood] += 1
    if not scores:
        return "calm"
    return normalize_mood(scores.most_common(1)[0][0])


def heuristic_face_mood(image_stats: Optional[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    stats = image_stats or {}
    brightness = float(stats.get("brightness", 125.0))
    contrast = float(stats.get("contrast", 35.0))
    warmth = float(stats.get("warmth", 0.0))
    saturation = float(stats.get("saturation", 20.0))

    if brightness >= 165 and saturation >= 26:
        return "happy", {"source": "heuristic", "confidence": 0.64, "stats": stats}
    if brightness <= 80:
        return "sad", {"source": "heuristic", "confidence": 0.66, "stats": stats}
    if contrast >= 60 and warmth >= 8:
        return "energetic", {"source": "heuristic", "confidence": 0.62, "stats": stats}
    if contrast >= 52 and brightness < 130:
        return "anxious", {"source": "heuristic", "confidence": 0.58, "stats": stats}
    if 105 <= brightness <= 150 and 35 <= contrast <= 55:
        return "focus", {"source": "heuristic", "confidence": 0.57, "stats": stats}
    return "calm", {"source": "heuristic", "confidence": 0.55, "stats": stats}


def parse_data_url(data_url: str) -> bytes:
    if not data_url or "," not in data_url:
        return b""
    return base64.b64decode(data_url.split(",", 1)[1])


def decode_image_for_ml(image_data_url: str, image_size: int) -> Optional[Any]:
    if PILImage is None or np is None:
        return None
    image_bytes = parse_data_url(image_data_url)
    if not image_bytes:
        return None
    image = PILImage.open(io.BytesIO(image_bytes)).convert("RGB").resize((image_size, image_size))
    return np.array(image)


def latest_model_record() -> Optional[Dict[str, Any]]:
    with db_connection() as conn:
        row = conn.execute(
            "SELECT * FROM trained_models ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
    return dict(row) if row else None


def load_latest_model_bundle() -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    record = latest_model_record()
    if not record or tf is None:
        return None, record
    model_path = record["model_path"]
    if MODEL_CACHE["path"] == model_path and MODEL_CACHE["model"] is not None:
        return MODEL_CACHE["model"], MODEL_CACHE["metadata"]
    model = tf.keras.models.load_model(model_path)
    metadata = {
        "labels": safe_json_loads(record["labels_json"], []),
        "metrics": safe_json_loads(record["metrics_json"], {}),
        "base_model": record["base_model"],
        "image_size": record["image_size"],
        "model_path": record["model_path"],
    }
    MODEL_CACHE.update({"path": model_path, "model": model, "metadata": metadata})
    return model, metadata


def choose_gradcam_layer(model: Any) -> Optional[str]:
    if tf is None:
        return None
    for layer in reversed(model.layers):
        try:
            output_shape = getattr(layer, "output_shape", None)
            if output_shape is not None and len(output_shape) == 4:
                return layer.name
        except Exception:
            continue
    return None


def make_gradcam_overlay(model: Any, image_array: Any, class_index: int) -> Optional[str]:
    if tf is None or np is None or PILImage is None:
        return None
    layer_name = choose_gradcam_layer(model)
    if not layer_name:
        return None

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )
    inputs = np.expand_dims(image_array.astype("float32"), axis=0)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(inputs)
        loss = predictions[:, class_index]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap.numpy(), 0)
    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)
    heatmap_img = PILImage.fromarray(np.uint8(255 * heatmap)).resize(
        (image_array.shape[1], image_array.shape[0])
    )
    base_img = PILImage.fromarray(image_array.astype("uint8"))
    overlay = PILImage.blend(base_img.convert("RGBA"), heatmap_img.convert("RGBA"), alpha=0.35)
    buffer = io.BytesIO()
    overlay.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")


def predict_with_latest_model(image_data_url: str) -> Optional[Dict[str, Any]]:
    model, metadata = load_latest_model_bundle()
    if model is None or metadata is None:
        return None
    image_size = int(metadata["image_size"])
    image_array = decode_image_for_ml(image_data_url, image_size)
    if image_array is None or np is None:
        return None
    predictions = model.predict(np.expand_dims(image_array.astype("float32"), axis=0), verbose=0)[0]
    labels = metadata["labels"]
    best_index = int(np.argmax(predictions))
    best_label = labels[best_index] if best_index < len(labels) else f"class_{best_index}"
    gradcam = make_gradcam_overlay(model, image_array, best_index)
    return {
        "label": best_label,
        "normalized_label": normalize_mood(best_label),
        "confidence": round(float(predictions[best_index]), 4),
        "probabilities": {
            labels[idx] if idx < len(labels) else f"class_{idx}": round(float(prob), 4)
            for idx, prob in enumerate(predictions)
        },
        "gradcam": gradcam,
        "source": "cnn",
    }


def analyze_face_signal(
    image_data_url: str, image_stats: Optional[Dict[str, Any]]
) -> Tuple[str, Dict[str, Any]]:
    if image_data_url:
        try:
            prediction = predict_with_latest_model(image_data_url)
            if prediction:
                return prediction["normalized_label"], prediction
        except Exception:
            pass
    return heuristic_face_mood(image_stats)


def row_to_event(row: sqlite3.Row) -> Dict[str, Any]:
    event = dict(row)
    event["spotify"] = safe_json_loads(event.get("spotify_json"), [])
    event["notes"] = safe_json_loads(event.get("notes_json"), {})
    return event


def get_user_events(username: str, limit: int = 20) -> List[Dict[str, Any]]:
    if not username:
        return []
    with db_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM emotion_events WHERE username = ? ORDER BY created_at DESC LIMIT ?",
            (username, int(limit)),
        ).fetchall()
    return [row_to_event(row) for row in rows]


def digital_twin_summary(username: str) -> Dict[str, Any]:
    history = get_user_events(username, limit=25)
    if not history:
        return {
            "total_sessions": 0,
            "dominant_mood": "new-user",
            "recent_moods": [],
            "streak": 0,
            "last_resource": "No content served yet",
            "mood_counts": {},
            "insight": "The Digital Emotional Twin will start learning after the first analysis.",
        }

    mood_counts = Counter(event["final_mood"] for event in history if event.get("final_mood"))
    dominant_mood = mood_counts.most_common(1)[0][0]
    recent_moods = [event["final_mood"] for event in history[:5]]
    streak = 1
    latest = history[0]["final_mood"]
    for event in history[1:]:
        if event["final_mood"] == latest:
            streak += 1
        else:
            break

    if dominant_mood in {"sad", "anxious"}:
        insight = "Recent history suggests recovery-oriented content and grounding exercises should be prioritised."
    elif dominant_mood == "focus":
        insight = "The twin sees a productivity pattern, so it prefers structured focus playlists and low-distraction content."
    elif dominant_mood == "energetic":
        insight = "The twin detects activation and benefits from workout, motivation, or high-intensity recommendations."
    else:
        insight = "The twin is stable and can safely rotate balanced wellness and uplifting content."

    return {
        "total_sessions": len(history),
        "dominant_mood": dominant_mood,
        "recent_moods": recent_moods,
        "streak": streak,
        "last_resource": history[0].get("primary_resource_title") or "Not available",
        "mood_counts": dict(mood_counts),
        "insight": insight,
    }


def fuse_modalities(
    face_mood: str,
    voice_mood: str,
    emoji_mood: str,
    text_mood: str,
    history: List[Dict[str, Any]],
) -> Tuple[str, float, Dict[str, float]]:
    weights = {"face": 0.40, "voice": 0.20, "emoji": 0.15, "text": 0.25}
    contributions = {
        "face": normalize_mood(face_mood),
        "voice": normalize_mood(voice_mood),
        "emoji": normalize_mood(emoji_mood),
        "text": normalize_mood(text_mood),
    }
    scorebook: Dict[str, float] = defaultdict(float)
    active_weight = 0.0
    for source, mood in contributions.items():
        if mood:
            scorebook[mood] += weights[source]
            active_weight += weights[source]

    if history:
        last_mood = normalize_mood(history[0].get("final_mood"))
        if scorebook.get(last_mood):
            scorebook[last_mood] += 0.05

    if not scorebook:
        return "calm", 50.0, {}

    final_mood = max(scorebook, key=scorebook.get)
    confidence = round((scorebook[final_mood] / max(active_weight, 1e-6)) * 100, 2)
    return final_mood, confidence, {key: round(value, 3) for key, value in scorebook.items()}


def ranked_resources_for_user(username: str, mood: str, count: int = 4) -> List[Dict[str, Any]]:
    pool = RESOURCE_CATALOG.get(normalize_mood(mood), RESOURCE_CATALOG["calm"])
    history = get_user_events(username, limit=25)
    seen_urls = [event.get("primary_resource_url") for event in history if event.get("primary_resource_url")]
    seen_counter = Counter(seen_urls)
    recent_urls = seen_urls[:8]
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for item in pool:
        novelty = 2.0 / (1 + seen_counter.get(item["url"], 0))
        recent_penalty = 1.4 if item["url"] in recent_urls else 0.0
        provider_bonus = 0.2 if item["provider"] == "YouTube" else 0.15
        exploration = random.uniform(0.01, 0.2)
        score = novelty + provider_bonus + exploration - recent_penalty
        scored.append((score, item))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in scored[:count]]


def youtube_embed_url(url: str) -> str:
    if "watch?v=" in url:
        video_id = url.split("watch?v=", 1)[1].split("&", 1)[0]
        return f"https://www.youtube.com/embed/{video_id}"
    return ""


def spotify_access_token() -> str:
    client_id = get_secret("SPOTIFY_CLIENT_ID")
    client_secret = get_secret("SPOTIFY_CLIENT_SECRET")
    if not client_id or not client_secret:
        return ""
    if SPOTIFY_TOKEN_CACHE["access_token"] and time.time() < SPOTIFY_TOKEN_CACHE["expires_at"]:
        return SPOTIFY_TOKEN_CACHE["access_token"]

    payload = urlencode({"grant_type": "client_credentials"}).encode("utf-8")
    auth = base64.b64encode(f"{client_id}:{client_secret}".encode("utf-8")).decode("utf-8")
    req = Request(
        "https://accounts.spotify.com/api/token",
        data=payload,
        headers={
            "Authorization": f"Basic {auth}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    with urlopen(req, timeout=20) as response:
        parsed = json.loads(response.read().decode("utf-8"))
    token = parsed.get("access_token", "")
    expires_in = int(parsed.get("expires_in", 3600))
    SPOTIFY_TOKEN_CACHE["access_token"] = token
    SPOTIFY_TOKEN_CACHE["expires_at"] = time.time() + max(expires_in - 60, 60)
    return token


def spotify_recommendations(mood: str, limit: int = 5) -> List[Dict[str, str]]:
    fallback_queries = {
        "happy": ["happy hits", "mood booster"],
        "sad": ["sad bops", "life sucks"],
        "calm": ["peaceful piano", "calm vibes"],
        "energetic": ["beast mode", "power workout"],
        "focus": ["deep focus", "brain food"],
        "anxious": ["stress relief", "anxiety relief"],
    }
    try:
        token = spotify_access_token()
        if not token:
            raise RuntimeError("Spotify not configured")
        query = fallback_queries.get(mood, [mood])[0]
        params = urlencode({"q": query, "type": "track", "limit": str(limit)})
        req = Request(
            f"https://api.spotify.com/v1/search?{params}",
            headers={"Authorization": f"Bearer {token}"},
        )
        with urlopen(req, timeout=20) as response:
            parsed = json.loads(response.read().decode("utf-8"))
        tracks = []
        for item in parsed.get("tracks", {}).get("items", []):
            artists = ", ".join(artist["name"] for artist in item.get("artists", []))
            tracks.append(
                {
                    "title": item.get("name", "Track"),
                    "artist": artists,
                    "url": item.get("external_urls", {}).get("spotify", ""),
                }
            )
        if tracks:
            return tracks
    except Exception:
        pass

    return [
        {
            "title": query.title(),
            "artist": "Spotify search",
            "url": f"https://open.spotify.com/search/{quote(query)}",
        }
        for query in fallback_queries.get(mood, [mood])[:limit]
    ]


def openai_chat(messages: List[Dict[str, Any]], max_tokens: int = 300) -> Optional[str]:
    api_key = get_secret("OPENAI_API_KEY")
    if not api_key:
        return None
    payload = {
        "model": "gpt-4o-mini",
        "messages": messages,
        "temperature": 0.6,
        "max_tokens": max_tokens,
    }
    req = Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=35) as response:
            parsed = json.loads(response.read().decode("utf-8"))
        return parsed["choices"][0]["message"]["content"].strip()
    except (HTTPError, URLError, KeyError, IndexError, ValueError):
        return None


def fallback_support_reply(username: str, mood: str, twin: Dict[str, Any]) -> str:
    prefix = f"{username or 'User'}, " if username else ""
    suggestions = {
        "happy": "keep the momentum with one gratitude note, one stretch break, and one uplifting track.",
        "sad": "try a slower pace: breathing, water, and one healing resource before making big decisions.",
        "calm": "this is a good moment for reflection, planning, or mindful rest.",
        "energetic": "channel the activation into movement, a focused sprint, or a workout playlist.",
        "focus": "protect the state with a timer, one priority, and a distraction-free work block.",
        "anxious": "shift to grounding: longer exhale breathing, reduce stimulation, and choose one simple next step.",
    }
    return (
        prefix
        + f"your detected mood is {mood}. "
        + suggestions.get(mood, "take one supportive step and keep the routine simple.")
        + f" Digital Twin insight: {twin.get('insight', '')}"
    )


def generate_support_reply(
    username: str,
    mood: str,
    user_text: str,
    voice_text: str,
    twin: Dict[str, Any],
) -> str:
    combined_text = " ".join(part for part in [user_text, voice_text] if part).strip()
    system = (
        "You are an empathetic emotional intelligence assistant for a major project demo. "
        "Be practical, supportive, concise, and safe. Give 3 short action steps at the end."
    )
    user = (
        f"User: {username or 'guest'}\n"
        f"Detected mood: {mood}\n"
        f"Combined text: {combined_text or 'No direct text provided.'}\n"
        f"Digital twin summary: {json.dumps(twin)}\n"
        "Respond in one supportive paragraph followed by 3 bullet action steps."
    )
    reply = openai_chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=320,
    )
    return reply or fallback_support_reply(username, mood, twin)


def generate_image_chat_reply(username: str, message: str, image_data_url: str) -> str:
    if not message and not image_data_url:
        return "Add a message or image so the assistant can respond."
    if not get_secret("OPENAI_API_KEY"):
        return (
            "OpenAI API is not configured. Save your key in the setup panel to enable image-aware chat."
        )
    content: List[Dict[str, Any]] = []
    if message.strip():
        content.append({"type": "text", "text": message.strip()})
    if image_data_url:
        content.append({"type": "image_url", "image_url": {"url": image_data_url}})
    reply = openai_chat(
        [
            {
                "role": "system",
                "content": (
                    "You are a multimodal assistant for CEI-ALOS. "
                    "Explain the image briefly, relate it to emotional context if relevant, "
                    "and suggest one helpful next step."
                ),
            },
            {"role": "user", "content": content},
        ],
        max_tokens=320,
    )
    return reply or f"{username or 'User'}, the image-aware assistant could not generate a response right now."


def log_emotion_event(
    username: str,
    face_mood: str,
    face_source: str,
    voice_mood: str,
    emoji_mood: str,
    text_mood: str,
    final_mood: str,
    confidence: float,
    primary_resource: Dict[str, Any],
    spotify_tracks: List[Dict[str, str]],
    notes: Dict[str, Any],
) -> None:
    with db_connection() as conn:
        conn.execute(
            """
            INSERT INTO emotion_events (
                id, username, created_at, face_mood, face_source, voice_mood, emoji_mood,
                text_mood, final_mood, confidence, primary_resource_title, primary_resource_url,
                spotify_json, notes_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                username,
                utcnow_iso(),
                face_mood,
                face_source,
                voice_mood,
                emoji_mood,
                text_mood,
                final_mood,
                float(confidence),
                primary_resource.get("title", ""),
                primary_resource.get("url", ""),
                json.dumps(spotify_tracks),
                json.dumps(notes),
            ),
        )


def aggregate_dashboard(username: str = "") -> Dict[str, Any]:
    with db_connection() as conn:
        if username:
            rows = conn.execute(
                "SELECT * FROM emotion_events WHERE username = ? ORDER BY created_at DESC LIMIT 50",
                (username,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM emotion_events ORDER BY created_at DESC LIMIT 50"
            ).fetchall()
    events = [row_to_event(row) for row in rows]
    mood_counts = Counter(event["final_mood"] for event in events if event.get("final_mood"))
    resource_counts = Counter(
        event["primary_resource_title"] for event in events if event.get("primary_resource_title")
    )
    return {
        "username": username or "all-users",
        "total_sessions": len(events),
        "mood_counts": dict(mood_counts),
        "recent_events": [
            {
                "timestamp": event["created_at"],
                "final_mood": event["final_mood"],
                "resource": event["primary_resource_title"],
                "confidence": event["confidence"],
            }
            for event in events[:10]
        ],
        "top_resources": resource_counts.most_common(5),
        "digital_twin": digital_twin_summary(username) if username else None,
    }


def missing_ml_dependencies() -> List[str]:
    missing = []
    if np is None:
        missing.append("numpy")
    if PILImage is None:
        missing.append("Pillow")
    if tf is None:
        missing.append("tensorflow")
    if classification_report is None or confusion_matrix is None or roc_auc_score is None:
        missing.append("scikit-learn")
    return missing


def dataset_image_paths(folder: Path) -> List[Path]:
    patterns = ["*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"]
    files: List[Path] = []
    for pattern in patterns:
        files.extend(folder.rglob(pattern))
    return [path for path in files if path.is_file()]


def extract_and_prepare_dataset(upload_bytes: bytes) -> Path:
    extract_root = DATASET_DIR / f"dataset_{int(time.time())}_{secrets.token_hex(4)}"
    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(upload_bytes)) as archive:
        archive.extractall(extract_root)

    direct_splits = [extract_root / split for split in ("train", "val", "test")]
    if all(path.exists() for path in direct_splits):
        return extract_root

    candidates = [path for path in extract_root.iterdir() if path.is_dir()]
    base_dir = candidates[0] if len(candidates) == 1 and not (extract_root / "train").exists() else extract_root
    class_dirs = [path for path in base_dir.iterdir() if path.is_dir()]
    if not class_dirs:
        raise ValueError("Dataset zip must contain class folders or train/val/test folders.")

    prepared = DATASET_DIR / f"prepared_{int(time.time())}_{secrets.token_hex(4)}"
    for split in ("train", "val", "test"):
        (prepared / split).mkdir(parents=True, exist_ok=True)

    rng = random.Random(42)
    for class_dir in class_dirs:
        image_paths = dataset_image_paths(class_dir)
        if len(image_paths) < 3:
            raise ValueError(
                f"Class '{class_dir.name}' needs at least 3 images for train/val/test splits."
            )
        rng.shuffle(image_paths)
        total = len(image_paths)
        train_cutoff = max(1, int(total * 0.7))
        val_cutoff = max(train_cutoff + 1, int(total * 0.85))
        splits = {
            "train": image_paths[:train_cutoff],
            "val": image_paths[train_cutoff:val_cutoff],
            "test": image_paths[val_cutoff:],
        }
        for split, paths in splits.items():
            if not paths:
                paths = image_paths[-1:]
            target_class_dir = prepared / split / class_dir.name
            target_class_dir.mkdir(parents=True, exist_ok=True)
            for path in paths:
                shutil.copy2(path, target_class_dir / path.name)
    return prepared


def build_transfer_model(base_model_name: str, image_size: int, num_classes: int) -> Any:
    if tf is None:
        raise RuntimeError("TensorFlow is not available.")

    model_library = {
        "MobileNetV2": (
            tf.keras.applications.MobileNetV2,
            tf.keras.applications.mobilenet_v2.preprocess_input,
        ),
        "EfficientNetB0": (
            tf.keras.applications.EfficientNetB0,
            tf.keras.applications.efficientnet.preprocess_input,
        ),
        "ResNet50": (
            tf.keras.applications.ResNet50,
            tf.keras.applications.resnet.preprocess_input,
        ),
    }
    if base_model_name not in model_library:
        raise ValueError(f"Unsupported base model: {base_model_name}")
    base_model_builder, preprocess_fn = model_library[base_model_name]
    base_model = base_model_builder(
        include_top=False, weights="imagenet", input_shape=(image_size, image_size, 3)
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="augmentation",
    )
    x = augmentation(inputs)
    x = preprocess_fn(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model, base_model


def train_transfer_learning_model(
    dataset_zip_bytes: bytes,
    base_model_name: str,
    epochs: int,
    image_size: int,
) -> Dict[str, Any]:
    missing = missing_ml_dependencies()
    if missing:
        raise RuntimeError(
            "CNN lab requires the following optional packages: " + ", ".join(sorted(set(missing)))
        )

    prepared_dataset = extract_and_prepare_dataset(dataset_zip_bytes)
    train_dir = prepared_dataset / "train"
    val_dir = prepared_dataset / "val"
    test_dir = prepared_dataset / "test"

    batch_size = 16
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        label_mode="categorical",
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=True,
        seed=42,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        label_mode="categorical",
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=False,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        label_mode="categorical",
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=False,
    )

    class_names = train_ds.class_names
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)

    model, base_model = build_transfer_model(base_model_name, image_size, len(class_names))
    history = model.fit(train_ds, validation_data=val_ds, epochs=max(1, int(epochs)), verbose=0)

    # Fine-tune the last layers after the transfer-learning warm-up.
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    fine_tune_epochs = 1 if epochs <= 2 else 2
    fine_tune_history = model.fit(
        train_ds, validation_data=val_ds, epochs=fine_tune_epochs, verbose=0
    )

    probabilities = model.predict(test_ds, verbose=0)
    y_true_one_hot = np.concatenate([labels.numpy() for _, labels in test_ds], axis=0)
    y_true = np.argmax(y_true_one_hot, axis=1)
    y_pred = np.argmax(probabilities, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    accuracy = float(np.mean(y_true == y_pred))
    matrix = confusion_matrix(y_true, y_pred).tolist()

    try:
        y_bin = label_binarize(y_true, classes=list(range(len(class_names))))
        roc_auc = float(roc_auc_score(y_bin, probabilities, multi_class="ovr", average="macro"))
    except Exception:
        roc_auc = None

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    run_id = f"model_{int(time.time())}_{secrets.token_hex(4)}"
    output_dir = MODEL_DIR / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.keras"
    model.save(model_path)

    history_data = {
        "warmup": {key: [float(v) for v in values] for key, values in history.history.items()},
        "finetune": {
            key: [float(v) for v in values] for key, values in fine_tune_history.history.items()
        },
    }
    metrics = {
        "accuracy": round(accuracy, 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4) if roc_auc is not None else None,
        "confusion_matrix": matrix,
        "classification_report": report,
        "history": history_data,
        "dataset_path": str(prepared_dataset),
    }

    with db_connection() as conn:
        conn.execute(
            """
            INSERT INTO trained_models (id, created_at, model_path, labels_json, metrics_json, base_model, image_size)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                utcnow_iso(),
                str(model_path),
                json.dumps(class_names),
                json.dumps(metrics),
                base_model_name,
                int(image_size),
            ),
        )

    MODEL_CACHE.update({"path": "", "model": None, "metadata": None})
    return {
        "message": "Transfer-learning model trained successfully.",
        "base_model": base_model_name,
        "image_size": image_size,
        "class_names": class_names,
        "metrics": metrics,
    }


def predict_cnn_image(image_bytes: bytes) -> Dict[str, Any]:
    missing = missing_ml_dependencies()
    if missing:
        raise RuntimeError(
            "CNN prediction requires the following optional packages: " + ", ".join(sorted(set(missing)))
        )
    model, metadata = load_latest_model_bundle()
    if model is None or metadata is None:
        raise RuntimeError("No trained model found. Train a transfer-learning model first.")

    image_size = int(metadata["image_size"])
    image = PILImage.open(io.BytesIO(image_bytes)).convert("RGB").resize((image_size, image_size))
    image_array = np.array(image)
    probabilities = model.predict(np.expand_dims(image_array.astype("float32"), axis=0), verbose=0)[0]
    best_index = int(np.argmax(probabilities))
    labels = metadata["labels"]
    label = labels[best_index] if best_index < len(labels) else f"class_{best_index}"
    gradcam = make_gradcam_overlay(model, image_array, best_index)
    return {
        "label": label,
        "normalized_label": normalize_mood(label),
        "confidence": round(float(probabilities[best_index]), 4),
        "probabilities": {
            labels[idx] if idx < len(labels) else f"class_{idx}": round(float(prob), 4)
            for idx, prob in enumerate(probabilities)
        },
        "gradcam": gradcam,
        "metadata": metadata,
    }


def explain_prediction_with_llm(prediction: Dict[str, Any]) -> str:
    label = prediction.get("label", "unknown")
    confidence = prediction.get("confidence", 0.0)
    prompt = (
        f"A CNN transfer-learning model predicted class '{label}' with confidence {confidence}. "
        "Explain this result simply for a student viva, mention why Grad-CAM is helpful, "
        "and give one limitation of the model."
    )
    reply = openai_chat(
        [
            {
                "role": "system",
                "content": "You explain machine learning results clearly for academic demos.",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=220,
    )
    if reply:
        return reply
    return (
        "The model used transfer learning to recognise the uploaded image class. "
        "Grad-CAM highlights which visual regions most influenced the prediction, "
        "which improves explainability during the viva. A limitation is that accuracy "
        "depends heavily on dataset quality and class balance."
    )


def create_app() -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def index() -> str:
        catalog_preview = {
            mood: items[:4] for mood, items in RESOURCE_CATALOG.items()
        }
        return render_template_string(
            PAGE_TEMPLATE,
            title=APP_TITLE,
            version=APP_VERSION,
            resource_total=RESOURCE_TOTAL,
            architecture_pipeline=ARCHITECTURE_PIPELINE,
            viva_statement=VIVA_STATEMENT,
            setup_commands=SETUP_COMMANDS,
            pwa_guide=PWA_GUIDE,
            api_status=config_status(),
            catalog_preview=catalog_preview,
            catalog_counts={mood: len(items) for mood, items in RESOURCE_CATALOG.items()},
            moods=MOODS,
            emoji_options=list(EMOJI_MAP.keys()),
        )

    @app.route("/manifest.json")
    def manifest() -> Response:
        return jsonify(
            {
                "name": APP_TITLE,
                "short_name": "CEI-ALOS",
                "description": "Installable CEI-ALOS emotional intelligence app with multimodal AI and CNN lab.",
                "start_url": "/",
                "display": "standalone",
                "background_color": "#0b1020",
                "theme_color": "#7c89ff",
                "icons": [{"src": "/icon.svg", "sizes": "any", "type": "image/svg+xml"}],
            }
        )

    @app.route("/service-worker.js")
    def service_worker() -> Response:
        return Response(
            SERVICE_WORKER_JS,
            mimetype="application/javascript",
            headers={"Cache-Control": "no-cache"},
        )

    @app.route("/icon.svg")
    def icon_svg() -> Response:
        return Response(ICON_SVG, mimetype="image/svg+xml")

    @app.post("/api/save-config")
    def api_save_config() -> Response:
        payload = request.get_json(silent=True) or {}
        saved = save_runtime_config(payload)
        return jsonify(
            {
                "ok": True,
                "saved_keys": sorted(saved.keys()),
                "status": config_status(),
                "message": "Configuration saved locally inside cei_alos_data/runtime_config.json",
            }
        )

    @app.post("/api/analyze")
    def api_analyze() -> Response:
        payload = request.get_json(silent=True) or {}
        username = (payload.get("username") or "guest").strip() or "guest"
        user_text = (payload.get("user_text") or "").strip()
        voice_text = (payload.get("voice_text") or "").strip()
        emoji = (payload.get("emoji") or "😌").strip()
        image_data_url = (payload.get("image_data_url") or "").strip()
        image_stats = payload.get("image_stats") or {}

        history = get_user_events(username, limit=10)
        face_mood, face_details = analyze_face_signal(image_data_url, image_stats)
        voice_mood = mood_from_text(voice_text)
        emoji_mood = normalize_mood(EMOJI_MAP.get(emoji, "calm"))
        text_mood = mood_from_text(user_text)
        final_mood, confidence, scorebook = fuse_modalities(
            face_mood, voice_mood, emoji_mood, text_mood, history
        )

        recommended_resources = ranked_resources_for_user(username, final_mood, count=4)
        primary_resource = recommended_resources[0] if recommended_resources else RESOURCE_CATALOG["calm"][0]
        spotify_tracks = spotify_recommendations(final_mood, limit=5)
        twin = digital_twin_summary(username)
        assistant_reply = generate_support_reply(username, final_mood, user_text, voice_text, twin)

        notes = {
            "scorebook": scorebook,
            "face_details": face_details,
            "digital_twin": twin,
            "recommended_resources": recommended_resources,
            "assistant_reply": assistant_reply,
        }
        log_emotion_event(
            username=username,
            face_mood=face_mood,
            face_source=face_details.get("source", "heuristic"),
            voice_mood=voice_mood,
            emoji_mood=emoji_mood,
            text_mood=text_mood,
            final_mood=final_mood,
            confidence=confidence,
            primary_resource=primary_resource,
            spotify_tracks=spotify_tracks,
            notes=notes,
        )

        return jsonify(
            {
                "ok": True,
                "username": username,
                "final_mood": final_mood,
                "confidence": confidence,
                "modalities": {
                    "face": face_mood,
                    "voice": voice_mood,
                    "emoji": emoji_mood,
                    "text": text_mood,
                },
                "scorebook": scorebook,
                "face_details": face_details,
                "recommended_resources": recommended_resources,
                "primary_resource": {
                    **primary_resource,
                    "embed_url": youtube_embed_url(primary_resource["url"]),
                },
                "spotify_tracks": spotify_tracks,
                "digital_twin": digital_twin_summary(username),
                "assistant_reply": assistant_reply,
            }
        )

    @app.post("/api/chat")
    def api_chat() -> Response:
        payload = request.get_json(silent=True) or {}
        username = (payload.get("username") or "guest").strip() or "guest"
        message = (payload.get("message") or "").strip()
        image_data_url = (payload.get("image_data_url") or "").strip()
        reply = generate_image_chat_reply(username, message, image_data_url)
        return jsonify({"ok": True, "reply": reply})

    @app.get("/api/dashboard")
    def api_dashboard() -> Response:
        username = (request.args.get("username") or "").strip()
        return jsonify({"ok": True, "dashboard": aggregate_dashboard(username)})

    @app.post("/api/train")
    def api_train() -> Response:
        dataset_file = request.files.get("dataset_zip")
        base_model = (request.form.get("base_model") or "MobileNetV2").strip()
        epochs = int(request.form.get("epochs") or 3)
        image_size = int(request.form.get("image_size") or 224)
        if dataset_file is None or not dataset_file.filename:
            return jsonify({"ok": False, "error": "Upload a dataset zip first."}), 400
        try:
            result = train_transfer_learning_model(
                dataset_file.read(), base_model_name=base_model, epochs=epochs, image_size=image_size
            )
            return jsonify({"ok": True, "result": result})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc), "trace": traceback.format_exc(limit=2)}), 500

    @app.post("/api/predict-cnn")
    def api_predict_cnn() -> Response:
        image_file = request.files.get("image")
        if image_file is None or not image_file.filename:
            return jsonify({"ok": False, "error": "Upload an image to run CNN prediction."}), 400
        try:
            prediction = predict_cnn_image(image_file.read())
            explanation = explain_prediction_with_llm(prediction)
            return jsonify({"ok": True, "prediction": prediction, "explanation": explanation})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc), "trace": traceback.format_exc(limit=2)}), 500

    return app


PAGE_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta name="theme-color" content="#7c89ff">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <title>{{ title }} - Single File Major Project</title>
    <link rel="manifest" href="/manifest.json">
    <style>
        :root {
            --bg: #0b1020;
            --panel: #121933;
            --panel-2: #1a2242;
            --text: #eef2ff;
            --muted: #b6c2ff;
            --accent: #7c89ff;
            --ok: #3ddc97;
            --warn: #ffb020;
            --danger: #ff6b81;
            --border: rgba(255,255,255,0.08);
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            font-family: Arial, Helvetica, sans-serif;
            background: linear-gradient(180deg, #0b1020 0%, #121933 100%);
            color: var(--text);
        }
        a { color: #9fb0ff; }
        .container { max-width: 1320px; margin: 0 auto; padding: 24px; }
        .hero {
            background: radial-gradient(circle at top right, rgba(124,137,255,0.25), transparent 34%), var(--panel);
            padding: 28px;
            border-radius: 20px;
            border: 1px solid var(--border);
            margin-bottom: 20px;
        }
        .hero h1 { margin: 0 0 12px; font-size: 2.2rem; }
        .hero p { color: var(--muted); margin: 8px 0; line-height: 1.5; }
        .pill-row { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 18px; }
        .pill {
            background: rgba(124,137,255,0.15);
            border: 1px solid rgba(124,137,255,0.35);
            color: #e7ebff;
            padding: 8px 12px;
            border-radius: 999px;
            font-size: 0.92rem;
        }
        .grid { display: grid; gap: 18px; }
        .grid.two { grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); }
        .grid.three { grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }
        .card {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.22);
        }
        .card h2, .card h3 { margin-top: 0; }
        .subtle { color: var(--muted); font-size: 0.96rem; line-height: 1.55; }
        .status {
            display: inline-block;
            padding: 6px 10px;
            border-radius: 999px;
            font-size: 0.85rem;
            margin-right: 8px;
            margin-bottom: 8px;
        }
        .ok { background: rgba(61,220,151,0.14); color: #9ff7cf; }
        .warn { background: rgba(255,176,32,0.12); color: #ffd48c; }
        .danger { background: rgba(255,107,129,0.12); color: #ffbdc8; }
        .form-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; }
        label { display: block; font-weight: 600; margin-bottom: 6px; }
        input, select, textarea, button {
            width: 100%;
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.12);
            background: var(--panel-2);
            color: var(--text);
            padding: 12px 14px;
            font-size: 0.96rem;
        }
        textarea { min-height: 100px; resize: vertical; }
        button {
            cursor: pointer;
            background: linear-gradient(90deg, #5b6fff, #8a6bff);
            border: none;
            font-weight: 700;
        }
        button.secondary { background: #273258; }
        button.small { padding: 10px 12px; font-size: 0.9rem; }
        .actions { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 12px; }
        .actions button { flex: 1 1 180px; }
        video, canvas, iframe {
            width: 100%;
            border-radius: 14px;
            border: 1px solid var(--border);
            background: #090d18;
        }
        iframe { min-height: 320px; }
        .result {
            margin-top: 18px;
            padding: 18px;
            border-radius: 16px;
            background: rgba(255,255,255,0.03);
            border: 1px solid var(--border);
        }
        .list { margin: 0; padding-left: 18px; }
        .mono {
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            font-size: 0.88rem;
            white-space: pre-wrap;
            background: #0b1226;
            border-radius: 12px;
            padding: 12px;
            border: 1px solid var(--border);
        }
        .resource-card {
            border: 1px solid var(--border);
            background: rgba(255,255,255,0.03);
            border-radius: 14px;
            padding: 14px;
            margin-bottom: 10px;
        }
        .bar-row { margin-bottom: 12px; }
        .bar-label { margin-bottom: 4px; color: var(--muted); }
        .bar {
            height: 12px;
            border-radius: 999px;
            background: rgba(255,255,255,0.08);
            overflow: hidden;
        }
        .bar > span {
            display: block;
            height: 100%;
            background: linear-gradient(90deg, #6f86ff, #9f7bff);
        }
        .split {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 18px;
        }
        .footer { padding: 30px 0 10px; color: var(--muted); text-align: center; }
        @media (max-width: 760px) {
            .container { padding: 16px; }
            .hero h1 { font-size: 1.8rem; }
        }
    </style>
</head>
<body>
    <div class="container">
        <section class="hero">
            <h1>{{ title }} <span style="font-size:0.7em;color:#aeb8ff;">v{{ version }}</span></h1>
            <p>
                Single-file major project implementation for <strong>Cognitive Emotion Intelligence & Adaptive Lifestyle OS</strong>.
                This build includes a multi-modal emotion engine, reinforcement-style non-repetition, Digital Emotional Twin analytics,
                OpenAI + Spotify integration, a structured {{ resource_total }}-resource catalog, and an optional CNN transfer-learning lab.
            </p>
            <p>
                <strong>Best substitute for paid APK wrapper tools:</strong> this project is delivered as an installable
                <strong>PWA</strong> so it can be installed on Android and Windows 11 directly from the browser without needing
                AppsGeyser or WebViewGold.
            </p>
            <div class="pill-row">
                {% for item in architecture_pipeline %}
                    <span class="pill">{{ item }}</span>
                {% endfor %}
            </div>
        </section>

        <section class="grid two">
            <div class="card">
                <h2>1. Final Architecture</h2>
                <ol class="list">
                    {% for item in architecture_pipeline %}
                        <li>{{ item }}</li>
                    {% endfor %}
                </ol>
                <p class="subtle">The pipeline is tuned for major-project demonstrations and can work in a pure heuristic mode or upgrade into a CNN-based mode after model training.</p>
            </div>
            <div class="card">
                <h2>2. Viva Statement</h2>
                <p class="subtle">{{ viva_statement }}</p>
                <h3>Why this replaces paid APK wrappers</h3>
                <ul class="list">
                    <li>PWA install works on Android with Add to Home Screen / Install App.</li>
                    <li>Windows 11 users can install it from Edge/Chrome as a desktop app.</li>
                    <li>No separate static files, wrapper service, or paid converter is required for the single-file version.</li>
                </ul>
            </div>
        </section>

        <section class="card">
            <h2>3. API Setup Wizard</h2>
            <p class="subtle">
                To keep the project self-contained, you can save your keys directly from this page. They are stored locally inside
                <span class="mono">cei_alos_data/runtime_config.json</span> unless environment variables already exist.
            </p>
            <div id="config-status" class="result">
                <span class="status {{ 'ok' if api_status['openai']['configured'] else 'warn' }}">OpenAI: {{ api_status['openai']['source'] }} / {{ api_status['openai']['masked'] }}</span>
                <span class="status {{ 'ok' if api_status['spotify']['configured'] else 'warn' }}">Spotify: {{ api_status['spotify']['source'] }} / {{ api_status['spotify']['masked_client_id'] }}</span>
                {% for dep_name, dep_ok in api_status['dependencies'].items() %}
                    <span class="status {{ 'ok' if dep_ok else 'danger' }}">{{ dep_name }}: {{ 'ready' if dep_ok else 'missing' }}</span>
                {% endfor %}
            </div>
            <div class="form-grid" style="margin-top:16px;">
                <div>
                    <label for="openai_key">OpenAI API Key</label>
                    <input id="openai_key" type="password" placeholder="sk-..." />
                </div>
                <div>
                    <label for="spotify_client_id">Spotify Client ID</label>
                    <input id="spotify_client_id" type="password" placeholder="Spotify Client ID" />
                </div>
                <div>
                    <label for="spotify_client_secret">Spotify Client Secret</label>
                    <input id="spotify_client_secret" type="password" placeholder="Spotify Client Secret" />
                </div>
            </div>
            <div class="actions">
                <button class="small" onclick="saveConfig()">Save Local Configuration</button>
                <button class="small secondary" onclick="refreshDashboard()">Refresh Dashboard</button>
            </div>
            <details style="margin-top:16px;">
                <summary>Manual environment-variable setup commands</summary>
                <div class="grid three" style="margin-top:12px;">
                    {% for name, commands in setup_commands.items() %}
                    <div>
                        <h3>{{ name }}</h3>
                        <div class="mono">{{ commands|join('\\n') }}</div>
                    </div>
                    {% endfor %}
                </div>
            </details>
        </section>

        <section class="grid two">
            <div class="card">
                <h2>4. Multi-Modal Emotion Analyzer</h2>
                <div class="form-grid">
                    <div>
                        <label for="username">Username</label>
                        <input id="username" placeholder="Enter username" value="demo-user" />
                    </div>
                    <div>
                        <label for="emoji">Emoji signal</label>
                        <select id="emoji">
                            {% for item in emoji_options %}
                                <option value="{{ item }}">{{ item }}</option>
                            {% endfor %}
                        </select>
                    </div>
                </div>
                <div class="split" style="margin-top:14px;">
                    <div>
                        <label for="user_text">Text input</label>
                        <textarea id="user_text" placeholder="Describe how you feel..."></textarea>
                    </div>
                    <div>
                        <label for="voice_text">Voice transcript</label>
                        <textarea id="voice_text" placeholder="Use browser microphone or paste transcript..."></textarea>
                        <div class="actions">
                            <button class="small secondary" onclick="startVoiceInput()">Start Voice Input</button>
                        </div>
                    </div>
                </div>
                <div class="split" style="margin-top:14px;">
                    <div>
                        <label>Live camera (desktop / browser permission)</label>
                        <video id="camera" autoplay playsinline muted></video>
                        <div class="actions">
                            <button class="small secondary" onclick="startCamera()">Start Camera</button>
                            <button class="small secondary" onclick="captureSnapshot()">Capture Snapshot</button>
                            <button class="small secondary" onclick="stopCamera()">Stop Camera</button>
                        </div>
                    </div>
                    <div>
                        <label for="image_upload">Image upload / mobile camera capture</label>
                        <input id="image_upload" type="file" accept="image/*" capture="user" />
                        <canvas id="preview_canvas" width="320" height="240" style="margin-top:14px;"></canvas>
                        <p class="subtle">If no trained CNN model exists, the system uses image brightness/contrast heuristics as the face signal.</p>
                    </div>
                </div>
                <input id="image_data" type="hidden" />
                <input id="image_stats" type="hidden" />
                <div class="actions">
                    <button onclick="analyzeEmotion()">Analyze Emotion + Recommend Content</button>
                </div>
                <div id="analysis-output" class="result">Results will appear here.</div>
            </div>

            <div class="card">
                <h2>5. Chatbot with Image Input</h2>
                <p class="subtle">This section covers the advanced enhancement requirement of CNN + LLM / chatbot with image input.</p>
                <label for="chat_message">Message</label>
                <textarea id="chat_message" placeholder="Ask for emotional support, study help, or image explanation..."></textarea>
                <label for="chat_image" style="margin-top:12px;">Optional image</label>
                <input id="chat_image" type="file" accept="image/*" />
                <div class="actions">
                    <button onclick="sendChat()">Send to AI</button>
                </div>
                <div id="chat-output" class="result">Chat response will appear here.</div>
            </div>
        </section>

        <section class="grid two">
            <div class="card">
                <h2>6. Digital Emotional Twin Dashboard</h2>
                <div class="form-grid">
                    <div>
                        <label for="dashboard_user">Username for analytics</label>
                        <input id="dashboard_user" placeholder="Same username as analyzer" value="demo-user" />
                    </div>
                </div>
                <div class="actions">
                    <button class="small" onclick="refreshDashboard()">Load Dashboard</button>
                </div>
                <div id="dashboard-output" class="result">Dashboard data will appear here.</div>
            </div>

            <div class="card">
                <h2>7. PWA Install Guide (Android + Windows 11)</h2>
                <div class="grid two">
                    <div>
                        <h3>Android</h3>
                        <ol class="list">
                            {% for step in pwa_guide['android'] %}
                                <li>{{ step }}</li>
                            {% endfor %}
                        </ol>
                    </div>
                    <div>
                        <h3>Windows 11</h3>
                        <ol class="list">
                            {% for step in pwa_guide['windows11'] %}
                                <li>{{ step }}</li>
                            {% endfor %}
                        </ol>
                    </div>
                </div>
                <p class="subtle">
                    For a strict native APK or Windows installer, you would need a separate native packaging stack
                    (Android Studio / Flutter / Kivy / BeeWare). For a true single-file Python implementation, PWA is the most appropriate substitute.
                </p>
            </div>
        </section>

        <section class="card">
            <h2>8. CNN Transfer Learning + Evaluation + Grad-CAM Lab</h2>
            <p class="subtle">
                This module satisfies the CNN project requirements:
                transfer learning, real-world dataset upload, deployment UI, prediction, precision/recall/F1/confusion matrix/ROC-AUC, and Grad-CAM explainability.
                Upload a zip structured as either <span class="mono">train/ val/ test/ class_name/*.jpg</span> or just <span class="mono">class_name/*.jpg</span>.
            </p>
            <div class="grid two">
                <div>
                    <h3>Train transfer-learning model</h3>
                    <div class="form-grid">
                        <div>
                            <label for="dataset_zip">Dataset zip</label>
                            <input id="dataset_zip" type="file" accept=".zip" />
                        </div>
                        <div>
                            <label for="base_model">Base model</label>
                            <select id="base_model">
                                <option>MobileNetV2</option>
                                <option>EfficientNetB0</option>
                                <option>ResNet50</option>
                            </select>
                        </div>
                        <div>
                            <label for="epochs">Epochs</label>
                            <input id="epochs" type="number" value="3" min="1" max="20" />
                        </div>
                        <div>
                            <label for="image_size">Image size</label>
                            <input id="image_size" type="number" value="224" min="96" max="384" />
                        </div>
                    </div>
                    <div class="actions">
                        <button onclick="trainCnn()">Train CNN Model</button>
                    </div>
                    <div id="cnn-train-output" class="result">Training metrics will appear here.</div>
                </div>
                <div>
                    <h3>Predict + explain uploaded image</h3>
                    <label for="cnn_image">Prediction image</label>
                    <input id="cnn_image" type="file" accept="image/*" />
                    <div class="actions">
                        <button onclick="predictCnn()">Run CNN Prediction</button>
                    </div>
                    <div id="cnn-predict-output" class="result">Prediction result and Grad-CAM will appear here.</div>
                </div>
            </div>
        </section>

        <section class="card">
            <h2>9. Large Structured Resource Catalog ({{ resource_total }} items)</h2>
            <p class="subtle">
                The catalog is expandable and already structured into major affective states for content recommendation.
                The reinforcement layer avoids recent repetition per user.
            </p>
            <div class="grid three">
                {% for mood, items in catalog_preview.items() %}
                <div class="resource-card">
                    <h3>{{ mood.title() }} ({{ catalog_counts[mood] }})</h3>
                    <ul class="list">
                        {% for item in items %}
                            <li><a href="{{ item['url'] }}" target="_blank" rel="noopener">{{ item['title'] }}</a> <span class="subtle">({{ item['provider'] }})</span></li>
                        {% endfor %}
                    </ul>
                </div>
                {% endfor %}
            </div>
        </section>

        <div class="footer">
            {{ title }} is packaged as a single Python source with browser-native install support for Android and Windows 11.
        </div>
    </div>

    <script>
        let liveStream = null;
        let deferredPrompt = null;

        function byId(id) { return document.getElementById(id); }
        function escapeHtml(value) {
            return String(value || "")
                .replaceAll("&", "&amp;")
                .replaceAll("<", "&lt;")
                .replaceAll(">", "&gt;")
                .replaceAll('"', "&quot;")
                .replaceAll("'", "&#39;");
        }

        function setHtml(id, html) {
            byId(id).innerHTML = html;
        }

        async function saveConfig() {
            const payload = {
                OPENAI_API_KEY: byId("openai_key").value.trim(),
                SPOTIFY_CLIENT_ID: byId("spotify_client_id").value.trim(),
                SPOTIFY_CLIENT_SECRET: byId("spotify_client_secret").value.trim()
            };
            const res = await fetch("/api/save-config", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            const html = `
                <span class="status ${data.status.openai.configured ? 'ok' : 'warn'}">OpenAI: ${escapeHtml(data.status.openai.source)} / ${escapeHtml(data.status.openai.masked)}</span>
                <span class="status ${data.status.spotify.configured ? 'ok' : 'warn'}">Spotify: ${escapeHtml(data.status.spotify.source)} / ${escapeHtml(data.status.spotify.masked_client_id)}</span>
                <div class="subtle" style="margin-top:10px;">${escapeHtml(data.message)}</div>
            `;
            setHtml("config-status", html);
        }

        async function startCamera() {
            try {
                liveStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
                byId("camera").srcObject = liveStream;
            } catch (error) {
                alert("Camera permission is required for live capture.");
            }
        }

        function stopCamera() {
            if (liveStream) {
                liveStream.getTracks().forEach(track => track.stop());
                liveStream = null;
            }
        }

        function computeImageStats(imageData) {
            const data = imageData.data;
            let brightness = 0;
            let warmth = 0;
            let saturation = 0;
            let values = [];
            for (let i = 0; i < data.length; i += 4) {
                const r = data[i];
                const g = data[i + 1];
                const b = data[i + 2];
                const lum = (r + g + b) / 3;
                brightness += lum;
                warmth += (r - b);
                saturation += Math.max(r, g, b) - Math.min(r, g, b);
                values.push(lum);
            }
            brightness = brightness / values.length;
            warmth = warmth / values.length;
            saturation = saturation / values.length;
            const mean = brightness;
            const variance = values.reduce((acc, value) => acc + Math.pow(value - mean, 2), 0) / values.length;
            return {
                brightness: Number(brightness.toFixed(2)),
                warmth: Number(warmth.toFixed(2)),
                saturation: Number(saturation.toFixed(2)),
                contrast: Number(Math.sqrt(variance).toFixed(2))
            };
        }

        function processImageDataUrl(dataUrl) {
            const canvas = byId("preview_canvas");
            const ctx = canvas.getContext("2d");
            const img = new Image();
            img.onload = () => {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                const stats = computeImageStats(imageData);
                byId("image_data").value = dataUrl;
                byId("image_stats").value = JSON.stringify(stats);
            };
            img.src = dataUrl;
        }

        function captureSnapshot() {
            const video = byId("camera");
            if (!video.srcObject) {
                alert("Start the camera first.");
                return;
            }
            const canvas = byId("preview_canvas");
            const ctx = canvas.getContext("2d");
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            const dataUrl = canvas.toDataURL("image/jpeg", 0.9);
            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            byId("image_data").value = dataUrl;
            byId("image_stats").value = JSON.stringify(computeImageStats(imageData));
        }

        byId("image_upload").addEventListener("change", (event) => {
            const file = event.target.files[0];
            if (!file) return;
            const reader = new FileReader();
            reader.onload = () => processImageDataUrl(reader.result);
            reader.readAsDataURL(file);
        });

        function startVoiceInput() {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            if (!SpeechRecognition) {
                alert("Browser speech recognition is not supported here. You can still paste the transcript manually.");
                return;
            }
            const recognition = new SpeechRecognition();
            recognition.lang = "en-US";
            recognition.interimResults = false;
            recognition.maxAlternatives = 1;
            recognition.onresult = (event) => {
                byId("voice_text").value = event.results[0][0].transcript;
            };
            recognition.onerror = () => {
                alert("Voice capture failed. Try again or paste text manually.");
            };
            recognition.start();
        }

        function renderBars(counts) {
            const values = Object.values(counts || {});
            const max = values.length ? Math.max(...values) : 1;
            return Object.entries(counts || {}).map(([label, value]) => `
                <div class="bar-row">
                    <div class="bar-label">${escapeHtml(label)} - ${escapeHtml(value)}</div>
                    <div class="bar"><span style="width:${(value / max) * 100}%"></span></div>
                </div>
            `).join("");
        }

        async function analyzeEmotion() {
            const payload = {
                username: byId("username").value.trim() || "guest",
                emoji: byId("emoji").value,
                user_text: byId("user_text").value,
                voice_text: byId("voice_text").value,
                image_data_url: byId("image_data").value,
                image_stats: JSON.parse(byId("image_stats").value || "{}")
            };
            setHtml("analysis-output", "Analyzing emotion signals and generating recommendations...");
            const res = await fetch("/api/analyze", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            if (!data.ok) {
                setHtml("analysis-output", `<span class="status danger">Analysis failed</span><div class="mono">${escapeHtml(JSON.stringify(data, null, 2))}</div>`);
                return;
            }
            byId("dashboard_user").value = payload.username;
            const resourcesHtml = data.recommended_resources.map(item => `
                <div class="resource-card">
                    <strong>${escapeHtml(item.title)}</strong><br>
                    <span class="subtle">${escapeHtml(item.provider)} / ${escapeHtml(item.type)}</span><br>
                    <a href="${escapeHtml(item.url)}" target="_blank" rel="noopener">Open resource</a>
                </div>
            `).join("");
            const spotifyHtml = data.spotify_tracks.map(track => `
                <li><a href="${escapeHtml(track.url)}" target="_blank" rel="noopener">${escapeHtml(track.title)}</a> - ${escapeHtml(track.artist)}</li>
            `).join("");
            const embedHtml = data.primary_resource.embed_url
                ? `<iframe src="${escapeHtml(data.primary_resource.embed_url)}" title="Recommended content" allowfullscreen></iframe>`
                : `<div class="resource-card"><a href="${escapeHtml(data.primary_resource.url)}" target="_blank" rel="noopener">Open primary recommended resource</a></div>`;
            const gradcamHtml = data.face_details.gradcam
                ? `<div><h4>Grad-CAM (trained CNN available)</h4><img src="${escapeHtml(data.face_details.gradcam)}" alt="Grad-CAM overlay" style="width:100%;border-radius:14px;border:1px solid rgba(255,255,255,0.1);"></div>`
                : "";
            const html = `
                <div class="status ok">Final mood: ${escapeHtml(data.final_mood)}</div>
                <div class="status warn">Confidence: ${escapeHtml(data.confidence)}%</div>
                <div class="status ${data.face_details.source === 'cnn' ? 'ok' : 'warn'}">Face source: ${escapeHtml(data.face_details.source)}</div>
                <div class="split" style="margin-top:14px;">
                    <div>
                        <h3>Fusion details</h3>
                        <ul class="list">
                            <li>Face: ${escapeHtml(data.modalities.face)}</li>
                            <li>Voice: ${escapeHtml(data.modalities.voice)}</li>
                            <li>Emoji: ${escapeHtml(data.modalities.emoji)}</li>
                            <li>Text: ${escapeHtml(data.modalities.text)}</li>
                        </ul>
                        <div class="mono" style="margin-top:12px;">${escapeHtml(JSON.stringify(data.scorebook, null, 2))}</div>
                        <h3 style="margin-top:14px;">AI Coach Reply</h3>
                        <div class="subtle">${escapeHtml(data.assistant_reply)}</div>
                        <h3 style="margin-top:14px;">Digital Twin</h3>
                        <div class="subtle">Dominant mood: ${escapeHtml(data.digital_twin.dominant_mood)} | Sessions: ${escapeHtml(data.digital_twin.total_sessions)} | Streak: ${escapeHtml(data.digital_twin.streak)}</div>
                        <div class="subtle" style="margin-top:8px;">${escapeHtml(data.digital_twin.insight)}</div>
                    </div>
                    <div>
                        <h3>Primary Recommendation</h3>
                        ${embedHtml}
                        <h3 style="margin-top:14px;">More curated resources</h3>
                        ${resourcesHtml}
                    </div>
                </div>
                <h3 style="margin-top:14px;">Spotify</h3>
                <ul class="list">${spotifyHtml}</ul>
                ${gradcamHtml}
            `;
            setHtml("analysis-output", html);
            refreshDashboard();
        }

        async function sendChat() {
            const message = byId("chat_message").value.trim();
            const file = byId("chat_image").files[0];
            let imageDataUrl = "";
            if (file) {
                imageDataUrl = await new Promise((resolve) => {
                    const reader = new FileReader();
                    reader.onload = () => resolve(reader.result);
                    reader.readAsDataURL(file);
                });
            }
            setHtml("chat-output", "Sending to AI...");
            const res = await fetch("/api/chat", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({
                    username: byId("username").value.trim() || "guest",
                    message,
                    image_data_url: imageDataUrl
                })
            });
            const data = await res.json();
            setHtml("chat-output", `<div class="subtle">${escapeHtml(data.reply || "No response received.")}</div>`);
        }

        async function refreshDashboard() {
            const username = byId("dashboard_user").value.trim();
            setHtml("dashboard-output", "Loading dashboard...");
            const res = await fetch(`/api/dashboard?username=${encodeURIComponent(username)}`);
            const data = await res.json();
            const dashboard = data.dashboard;
            const recentHtml = (dashboard.recent_events || []).map(event => `
                <li>${escapeHtml(event.timestamp)} - ${escapeHtml(event.final_mood)} - ${escapeHtml(event.resource || "")}</li>
            `).join("");
            const topResourcesHtml = (dashboard.top_resources || []).map(item => `
                <li>${escapeHtml(item[0])} (${escapeHtml(item[1])})</li>
            `).join("");
            const twin = dashboard.digital_twin || {};
            const html = `
                <div class="status ok">Sessions: ${escapeHtml(dashboard.total_sessions)}</div>
                <div class="status warn">User scope: ${escapeHtml(dashboard.username)}</div>
                <h3>Mood distribution</h3>
                ${renderBars(dashboard.mood_counts)}
                <h3>Recent events</h3>
                <ul class="list">${recentHtml || "<li>No events yet.</li>"}</ul>
                <h3>Top served resources</h3>
                <ul class="list">${topResourcesHtml || "<li>No resources served yet.</li>"}</ul>
                <h3>Digital Twin insight</h3>
                <div class="subtle">${escapeHtml(twin.insight || "No twin data yet.")}</div>
            `;
            setHtml("dashboard-output", html);
        }

        async function trainCnn() {
            const datasetFile = byId("dataset_zip").files[0];
            if (!datasetFile) {
                alert("Please upload a dataset zip.");
                return;
            }
            const form = new FormData();
            form.append("dataset_zip", datasetFile);
            form.append("base_model", byId("base_model").value);
            form.append("epochs", byId("epochs").value);
            form.append("image_size", byId("image_size").value);
            setHtml("cnn-train-output", "Training transfer-learning model... this can take time.");
            const res = await fetch("/api/train", { method: "POST", body: form });
            const data = await res.json();
            if (!data.ok) {
                setHtml("cnn-train-output", `<div class="status danger">Training failed</div><div class="mono">${escapeHtml(data.error || "Unknown error")}</div>`);
                return;
            }
            const metrics = data.result.metrics || {};
            const html = `
                <div class="status ok">Base model: ${escapeHtml(data.result.base_model)}</div>
                <div class="status ok">Image size: ${escapeHtml(data.result.image_size)}</div>
                <div class="status warn">Classes: ${escapeHtml((data.result.class_names || []).join(", "))}</div>
                <h3>Core metrics</h3>
                <ul class="list">
                    <li>Accuracy: ${escapeHtml(metrics.accuracy)}</li>
                    <li>Precision: ${escapeHtml(metrics.precision)}</li>
                    <li>Recall: ${escapeHtml(metrics.recall)}</li>
                    <li>F1-score: ${escapeHtml(metrics.f1_score)}</li>
                    <li>ROC-AUC: ${escapeHtml(metrics.roc_auc)}</li>
                </ul>
                <h3>Confusion matrix</h3>
                <div class="mono">${escapeHtml(JSON.stringify(metrics.confusion_matrix, null, 2))}</div>
                <h3>Classification report</h3>
                <div class="mono">${escapeHtml(JSON.stringify(metrics.classification_report, null, 2))}</div>
            `;
            setHtml("cnn-train-output", html);
        }

        async function predictCnn() {
            const imageFile = byId("cnn_image").files[0];
            if (!imageFile) {
                alert("Please upload an image for prediction.");
                return;
            }
            const form = new FormData();
            form.append("image", imageFile);
            setHtml("cnn-predict-output", "Running CNN prediction...");
            const res = await fetch("/api/predict-cnn", { method: "POST", body: form });
            const data = await res.json();
            if (!data.ok) {
                setHtml("cnn-predict-output", `<div class="status danger">Prediction failed</div><div class="mono">${escapeHtml(data.error || "Unknown error")}</div>`);
                return;
            }
            const prediction = data.prediction;
            const probabilities = Object.entries(prediction.probabilities || {}).map(([label, score]) => `
                <li>${escapeHtml(label)}: ${escapeHtml(score)}</li>
            `).join("");
            const gradcamHtml = prediction.gradcam
                ? `<img src="${escapeHtml(prediction.gradcam)}" alt="Grad-CAM overlay" style="width:100%;border-radius:14px;border:1px solid rgba(255,255,255,0.1);margin-top:12px;">`
                : `<div class="subtle">Grad-CAM is unavailable for this model or environment.</div>`;
            const html = `
                <div class="status ok">Predicted label: ${escapeHtml(prediction.label)}</div>
                <div class="status warn">Normalized mood: ${escapeHtml(prediction.normalized_label)}</div>
                <div class="status ok">Confidence: ${escapeHtml(prediction.confidence)}</div>
                <h3>Class probabilities</h3>
                <ul class="list">${probabilities}</ul>
                <h3>LLM Explanation</h3>
                <div class="subtle">${escapeHtml(data.explanation)}</div>
                <h3 style="margin-top:14px;">Grad-CAM</h3>
                ${gradcamHtml}
            `;
            setHtml("cnn-predict-output", html);
        }

        if ("serviceWorker" in navigator) {
            window.addEventListener("load", () => {
                navigator.serviceWorker.register("/service-worker.js").catch(() => null);
            });
        }

        window.addEventListener("beforeinstallprompt", (event) => {
            event.preventDefault();
            deferredPrompt = event;
        });

        refreshDashboard();
    </script>
</body>
</html>
"""


SERVICE_WORKER_JS = """
const CACHE_NAME = "cei-alos-v1";
const ASSETS = ["/", "/manifest.json", "/icon.svg"];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(ASSETS))
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key)))
    )
  );
});

self.addEventListener("fetch", (event) => {
  event.respondWith(
    caches.match(event.request).then((cached) => cached || fetch(event.request))
  );
});
"""


ICON_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">
  <defs>
    <linearGradient id="g" x1="0" x2="1" y1="0" y2="1">
      <stop offset="0%" stop-color="#556dff" />
      <stop offset="100%" stop-color="#8b63ff" />
    </linearGradient>
  </defs>
  <rect width="256" height="256" rx="56" fill="#0b1020" />
  <circle cx="128" cy="128" r="88" fill="url(#g)" opacity="0.20" />
  <path d="M76 140c10-36 33-54 52-54 21 0 42 17 52 54" fill="none" stroke="#ffffff" stroke-width="16" stroke-linecap="round"/>
  <circle cx="102" cy="112" r="10" fill="#ffffff" />
  <circle cx="154" cy="112" r="10" fill="#ffffff" />
  <path d="M96 162c9 8 20 12 32 12 12 0 23-4 32-12" fill="none" stroke="#ffffff" stroke-width="12" stroke-linecap="round"/>
  <text x="128" y="232" text-anchor="middle" fill="#dfe4ff" font-family="Arial" font-size="22">CEI-ALOS</text>
</svg>
"""


ensure_runtime_dirs()
init_db()
app = create_app()


if __name__ == "__main__":
    print(
        textwrap.dedent(
            f"""
            {APP_TITLE} {APP_VERSION}
            Run this on desktop or mobile browser, then install it as a PWA.
            Minimal dependency: Flask
            Optional CNN dependencies: numpy Pillow tensorflow scikit-learn
            """
        ).strip()
    )
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=False)
