#!/usr/bin/env python3
"""
GuneeshDua Major Project - Single-file Streamlit application.

Core features:
- Multimodal emotion input fusion (face image, emoji, free-text, voice transcript)
- Free Hugging Face API integration for emotion/NLP/ASR
- CNN with Global Average Pooling (GAP), training, and Grad-CAM explainability
- Precision/Recall/F1/Confusion Matrix/ROC-AUC evaluation
- History-aware, no-repetition recommendation with feedback-based reward updates
- Digital Emotional Twin CSV logging
- 100+ item resource catalog generation/export
- Automatic sampled dataset preparation from recent public emotion datasets
"""

from __future__ import annotations

import io
import json
import math
import os
import random
import re
import shutil
import sys
import textwrap
import uuid
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models

    TF_AVAILABLE = True
    TF_IMPORT_ERROR = ""
except Exception as tf_exc:  # pragma: no cover - optional runtime dependency
    TF_AVAILABLE = False
    TF_IMPORT_ERROR = str(tf_exc)

try:
    from datasets import load_dataset

    HF_DATASETS_AVAILABLE = True
    HF_DATASET_IMPORT_ERROR = ""
except Exception as ds_exc:  # pragma: no cover - optional runtime dependency
    load_dataset = None
    HF_DATASETS_AVAILABLE = False
    HF_DATASET_IMPORT_ERROR = str(ds_exc)

try:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials

    SPOTIPY_AVAILABLE = True
except Exception:  # pragma: no cover - optional runtime dependency
    spotipy = None
    SpotifyClientCredentials = None
    SPOTIPY_AVAILABLE = False


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATASET_DIR = BASE_DIR / "dataset"
MANIFEST_PATH = DATASET_DIR / "dataset_manifest.json"
CATALOG_PATH = BASE_DIR / "resource_catalog.csv"
TWIN_LOG_PATH = BASE_DIR / "cei_twin_log.csv"
RECOMMENDER_STATS_PATH = BASE_DIR / "recommender_stats.csv"
MODEL_PATH = MODELS_DIR / "mobile_transfer.keras"
MODEL_META_PATH = MODELS_DIR / "mobile_transfer_metadata.json"

DEFAULT_IMAGE_SIZE = 96
RANDOM_SEED = 42


if TF_AVAILABLE:
    tf.random.set_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

EMOTION_NORMALIZATION = {
    "joy": "happy",
    "happiness": "happy",
    "calm": "neutral",
    "relaxed": "neutral",
    "anger": "angry",
    "frustrated": "angry",
    "frustration": "angry",
    "anxious": "fear",
    "anxiety": "fear",
    "scared": "fear",
    "sorrow": "sad",
    "depressed": "sad",
    "astonished": "surprise",
    "surprised": "surprise",
    "disgusted": "disgust",
}

EMOJI_TO_EMOTION = {
    "😀": "happy",
    "😌": "neutral",
    "😢": "sad",
    "😡": "angry",
    "😱": "fear",
    "😮": "surprise",
    "🤢": "disgust",
}

TEXT_EMOTION_KEYWORDS: Dict[str, List[str]] = {
    "happy": [
        "happy",
        "excited",
        "joy",
        "grateful",
        "energetic",
        "motivated",
        "celebrate",
        "great",
        "awesome",
        "love",
    ],
    "sad": [
        "sad",
        "down",
        "lonely",
        "tired",
        "hopeless",
        "upset",
        "cry",
        "hurt",
        "broken",
    ],
    "angry": [
        "angry",
        "mad",
        "annoyed",
        "frustrated",
        "rage",
        "irritated",
        "furious",
    ],
    "fear": [
        "fear",
        "afraid",
        "anxious",
        "nervous",
        "worried",
        "panic",
        "stressed",
    ],
    "surprise": [
        "surprised",
        "unexpected",
        "wow",
        "shocked",
        "amazed",
    ],
    "disgust": [
        "disgust",
        "gross",
        "nausea",
        "hate",
        "repulsive",
    ],
    "neutral": [
        "normal",
        "okay",
        "fine",
        "balanced",
        "steady",
        "neutral",
    ],
}

QUESTION_BANK = {
    "happy": [
        "What made you feel this positive today, and how can you repeat it tomorrow?",
        "Which one achievement are you proud of this week?",
        "How can you channel this energy into your top academic or project task?",
        "Which person should you thank right now to strengthen this good momentum?",
        "What micro-goal can you finish in 30 minutes while your focus is high?",
    ],
    "sad": [
        "Which specific event triggered this feeling?",
        "What is one gentle action that can improve your mood in the next hour?",
        "Who is one trusted person you can talk to today?",
        "What evidence do you have that this difficult phase can still improve?",
        "What self-care action can you complete before sleeping tonight?",
    ],
    "angry": [
        "What boundary was crossed that triggered this reaction?",
        "What is within your control right now?",
        "How can you communicate your concern assertively without escalation?",
        "What 5-minute reset can reduce your emotional intensity?",
        "What would be a constructive next action after calming down?",
    ],
    "fear": [
        "What is the worst realistic outcome, and how likely is it?",
        "What preparation can reduce uncertainty by 20% immediately?",
        "Which small step can you take right now to regain control?",
        "What support system can you activate today?",
        "What past challenge did you overcome that proves your capability?",
    ],
    "surprise": [
        "What new opportunity or risk emerged from this unexpected event?",
        "What information do you still need before reacting?",
        "How can you quickly adapt your next 24-hour plan?",
        "What assumption was challenged, and what did you learn?",
        "What positive angle can you extract from this surprise?",
    ],
    "disgust": [
        "What exactly felt unacceptable or unethical in this situation?",
        "What standards or values are important to protect here?",
        "What clean alternative can replace what bothered you?",
        "How can you respond without carrying this emotional burden all day?",
        "What environment change would reduce this trigger in the future?",
    ],
    "neutral": [
        "What is your highest-priority task for the next 2 hours?",
        "Which habit, if repeated daily, will improve your results this semester?",
        "What can you optimize in your workflow today?",
        "What one metric should you track to ensure progress this week?",
        "How can you convert this stable state into focused productivity?",
    ],
}

DATASET_SOURCES = {
    "FER2025": {
        "repo_id": "imadhavan/FER2025",
        "split": "train",
        "auto_download": True,
        "notes": "Large public facial emotion dataset (sampled mode recommended).",
    },
    "EmoNet-Face-Big": {
        "repo_id": "laion/EmoNet-Face-Big",
        "split": "train",
        "auto_download": True,
        "notes": "Synthetic, diverse facial emotion dataset (sampled mode).",
    },
    "MER2024": {
        "repo_id": "",
        "split": "",
        "auto_download": False,
        "notes": "Listed for planning/documentation. Manual import only.",
    },
    "MER2023": {
        "repo_id": "",
        "split": "",
        "auto_download": False,
        "notes": "Listed for planning/documentation. Manual import only.",
    },
}


def normalize_emotion(raw_label: object) -> str:
    if raw_label is None:
        return "neutral"
    label = str(raw_label).strip().lower()
    label = EMOTION_NORMALIZATION.get(label, label)
    if label in EMOTIONS:
        return label
    if re.fullmatch(r"\d+", label):
        idx = int(label)
        if 0 <= idx < len(EMOTIONS):
            return EMOTIONS[idx]
    return "neutral"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def build_resource_catalog() -> pd.DataFrame:
    mood_to_queries = {
        "happy": ["feel good music", "uplifting playlist", "positive vibes tracks"],
        "sad": ["healing songs", "comfort music", "emotional support music"],
        "angry": ["calming music", "de-stress playlist", "focus after frustration"],
        "fear": ["anxiety relief music", "guided calm sounds", "stress control music"],
        "surprise": ["fresh discovery songs", "new release picks", "unexpected gems"],
        "neutral": ["deep work focus music", "balanced background songs", "study beats"],
        "disgust": ["detox mind music", "clean ambient tones", "mind reset audio"],
        "focus": ["coding music", "pomodoro beats", "instrumental concentration"],
        "sleep": ["sleep music", "night rain sounds", "deep rest ambient"],
        "workout": ["high energy workout songs", "cardio beats", "power mix tracks"],
        "mindfulness": ["mindfulness music", "meditation soundtrack", "breathwork sounds"],
        "study": ["exam study playlist", "lofi study beats", "memory focus tracks"],
    }
    source_templates = [
        ("YouTube Search", "search", "https://www.youtube.com/results?search_query={}"),
        ("Spotify Search", "search", "https://open.spotify.com/search/{}"),
        ("YouTube Music Search", "search", "https://music.youtube.com/search?q={}"),
        ("Wellness Search", "guide", "https://duckduckgo.com/?q={}+wellness"),
    ]

    rows: List[Dict[str, str]] = []
    for mood, query_pool in mood_to_queries.items():
        for i in range(1, 9):  # 8 items x 12 moods x 4 sources = 384 entries
            query = f"{query_pool[(i - 1) % len(query_pool)]} session {i}"
            encoded_query = quote_plus(query)
            for source_name, item_type, url_template in source_templates:
                rows.append(
                    {
                        "mood": mood,
                        "title": f"{mood.title()} Adaptive Session {i} ({source_name})",
                        "url": url_template.format(encoded_query),
                        "source": source_name,
                        "type": item_type,
                        "offline_fallback": (
                            "If internet is unavailable, play local instrumental tracks, "
                            "practice 4-7-8 breathing, and journal emotional state."
                        ),
                    }
                )
    return pd.DataFrame(rows)


def ensure_catalog_exists() -> pd.DataFrame:
    if CATALOG_PATH.exists():
        return pd.read_csv(CATALOG_PATH)
    catalog_df = build_resource_catalog()
    catalog_df.to_csv(CATALOG_PATH, index=False)
    return catalog_df


def export_catalog_cli_if_requested() -> bool:
    if "--export-catalog" not in sys.argv:
        return False
    catalog_df = build_resource_catalog()
    catalog_df.to_csv(CATALOG_PATH, index=False)
    print(f"Exported {len(catalog_df)} resources to: {CATALOG_PATH}")
    return True


def local_text_emotion(text: str) -> Tuple[str, float, Dict[str, float]]:
    cleaned = re.sub(r"[^a-zA-Z\s]", " ", text.lower())
    tokens = [tok for tok in cleaned.split() if tok]
    if not tokens:
        return "neutral", 0.35, {"neutral": 0.35}

    scores: Dict[str, float] = {emotion: 0.0 for emotion in EMOTIONS}
    for token in tokens:
        for emotion, words in TEXT_EMOTION_KEYWORDS.items():
            if token in words:
                scores[emotion] += 1.0
    if sum(scores.values()) == 0:
        return "neutral", 0.45, {"neutral": 0.45}

    best = max(scores, key=scores.get)
    conf = min(0.95, 0.45 + 0.1 * scores[best])
    normalized = {k: (v / sum(scores.values())) for k, v in scores.items() if v > 0}
    return best, conf, normalized


def huggingface_text_emotion(
    text: str, hf_token: str, model_id: str = "j-hartmann/emotion-english-distilroberta-base"
) -> Optional[Tuple[str, float, Dict[str, float]]]:
    if not hf_token.strip():
        return None
    endpoint = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {
        "inputs": text,
        "options": {"wait_for_model": True},
    }
    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=90)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict) and data.get("error"):
            return None
        if isinstance(data, list) and data and isinstance(data[0], list):
            labels = data[0]
        elif isinstance(data, list):
            labels = data
        else:
            return None
        parsed = {}
        for item in labels:
            label = normalize_emotion(item.get("label", "neutral"))
            parsed[label] = max(parsed.get(label, 0.0), float(item.get("score", 0.0)))
        if not parsed:
            return None
        emotion = max(parsed, key=parsed.get)
        confidence = float(parsed[emotion])
        return emotion, confidence, parsed
    except Exception:
        return None


def huggingface_generate_questions(
    mood: str,
    context: str,
    hf_token: str,
    model_id: str = "google/flan-t5-base",
) -> List[str]:
    if not hf_token.strip():
        return []
    prompt = (
        f"Generate 5 short reflective questions for a user feeling {mood}. "
        "Questions should be practical, supportive, and suitable for an engineering student. "
        f"Context: {context[:300]}"
    )
    endpoint = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 180,
            "temperature": 0.7,
            "return_full_text": False,
        },
        "options": {"wait_for_model": True},
    }
    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict) and data.get("error"):
            return []
        generated_text = ""
        if isinstance(data, list) and data and "generated_text" in data[0]:
            generated_text = data[0]["generated_text"]
        elif isinstance(data, dict) and "generated_text" in data:
            generated_text = data["generated_text"]
        if not generated_text:
            return []
        raw_lines = [line.strip(" -\t") for line in generated_text.splitlines() if line.strip()]
        cleaned = []
        for line in raw_lines:
            line = re.sub(r"^\d+[\).\s-]*", "", line).strip()
            if len(line) >= 8:
                cleaned.append(line if line.endswith("?") else f"{line}?")
        deduped = list(dict.fromkeys(cleaned))
        return deduped[:5]
    except Exception:
        return []


def huggingface_transcribe_audio(
    audio_bytes: bytes, hf_token: str, model_id: str = "openai/whisper-small"
) -> str:
    if not hf_token.strip() or not audio_bytes:
        return ""
    endpoint = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/octet-stream",
    }
    try:
        response = requests.post(endpoint, headers=headers, data=audio_bytes, timeout=180)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict):
            if "text" in data:
                return str(data["text"]).strip()
            if "error" in data:
                return ""
        return ""
    except Exception:
        return ""


def fallback_questions(mood: str, context: str) -> List[str]:
    default_qs = QUESTION_BANK.get(mood, QUESTION_BANK["neutral"])
    if context.strip():
        prefix = f"Considering your context: \"{context[:120]}\""
        return [f"{prefix}, {q[0].lower()}{q[1:]}" if q else q for q in default_qs]
    return default_qs


def model_paths_signature() -> Tuple[float, float]:
    model_mtime = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else -1.0
    meta_mtime = MODEL_META_PATH.stat().st_mtime if MODEL_META_PATH.exists() else -1.0
    return model_mtime, meta_mtime


@st.cache_resource(show_spinner=False)
def load_saved_model(_model_mtime: float, _meta_mtime: float):
    if not TF_AVAILABLE or not MODEL_PATH.exists():
        return None, [], DEFAULT_IMAGE_SIZE
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = EMOTIONS
    image_size = DEFAULT_IMAGE_SIZE
    if MODEL_META_PATH.exists():
        with open(MODEL_META_PATH, "r", encoding="utf-8") as meta_file:
            metadata = json.load(meta_file)
        class_names = metadata.get("class_names", EMOTIONS)
        image_size = int(metadata.get("image_size", DEFAULT_IMAGE_SIZE))
    return model, class_names, image_size


def preprocess_image(pil_image: Image.Image, image_size: int) -> np.ndarray:
    rgb = pil_image.convert("RGB").resize((image_size, image_size))
    arr = np.array(rgb, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    return arr


def face_emotion_inference(
    pil_image: Optional[Image.Image],
    model,
    class_names: List[str],
    image_size: int,
) -> Optional[Tuple[str, float, Dict[str, float]]]:
    if pil_image is None:
        return None
    if model is None or not class_names:
        return ("neutral", 0.40, {"neutral": 0.40})
    arr = preprocess_image(pil_image, image_size)
    probs = model.predict(arr, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    emotion = normalize_emotion(class_names[pred_idx])
    confidence = float(probs[pred_idx])
    dist = {
        normalize_emotion(class_names[idx]): float(prob)
        for idx, prob in enumerate(probs)
        if idx < len(class_names)
    }
    return emotion, confidence, dist


def fuse_emotions(modalities: Dict[str, Tuple[str, float]]) -> Tuple[str, float, Dict[str, float]]:
    # Weighted late fusion; missing modalities are ignored automatically.
    modality_weights = {
        "face": 0.40,
        "text": 0.25,
        "emoji": 0.20,
        "voice": 0.15,
    }
    scores = defaultdict(float)
    total_weight = 0.0
    for modality, (emotion, confidence) in modalities.items():
        weight = modality_weights.get(modality, 0.1)
        scores[normalize_emotion(emotion)] += weight * float(confidence)
        total_weight += weight
    if not scores:
        return "neutral", 0.30, {"neutral": 0.30}
    final_emotion = max(scores, key=scores.get)
    normalized_conf = scores[final_emotion] / max(total_weight, 1e-6)
    normalized_dist = {
        k: float(v / max(sum(scores.values()), 1e-9))
        for k, v in dict(scores).items()
    }
    return final_emotion, float(normalized_conf), normalized_dist


def ethical_ai_monitor(
    fused_emotion: str, fused_confidence: float, modalities: Dict[str, Tuple[str, float]]
) -> Tuple[str, List[str]]:
    warnings = []
    unique_emotions = {normalize_emotion(v[0]) for v in modalities.values()}
    if fused_confidence < 0.50:
        warnings.append("Low-confidence prediction detected; ask user for confirmation.")
    if len(unique_emotions) >= 3:
        warnings.append(
            "High modality disagreement detected; decision should be treated as advisory."
        )
    if len(modalities) < 2:
        warnings.append("Limited modalities available; include text/emoji/face for robustness.")
    risk_level = "low"
    if len(warnings) >= 2:
        risk_level = "medium"
    if fused_confidence < 0.35:
        risk_level = "high"
    if fused_emotion in {"sad", "fear", "angry"} and fused_confidence > 0.85:
        warnings.append("Sensitive emotional state detected; provide supportive neutral guidance.")
    return risk_level, warnings


def init_csv_if_missing(path: Path, columns: List[str]) -> None:
    ensure_parent(path)
    if not path.exists():
        pd.DataFrame(columns=columns).to_csv(path, index=False)


def append_twin_log(record: Dict[str, object]) -> None:
    columns = [
        "timestamp",
        "session_id",
        "user_id",
        "fused_emotion",
        "fused_confidence",
        "face_emotion",
        "face_confidence",
        "text_emotion",
        "text_confidence",
        "voice_emotion",
        "voice_confidence",
        "emoji_emotion",
        "emoji_confidence",
        "risk_level",
        "warnings",
        "recommended_titles",
        "feedback",
    ]
    init_csv_if_missing(TWIN_LOG_PATH, columns)
    df = pd.read_csv(TWIN_LOG_PATH)
    row = {col: record.get(col, "") for col in columns}
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(TWIN_LOG_PATH, index=False)


def load_recommender_stats() -> pd.DataFrame:
    columns = ["mood", "title", "source", "reward_sum", "count", "avg_reward", "last_updated"]
    init_csv_if_missing(RECOMMENDER_STATS_PATH, columns)
    return pd.read_csv(RECOMMENDER_STATS_PATH)


def update_recommender_feedback(mood: str, title: str, source: str, reward: int) -> None:
    stats_df = load_recommender_stats()
    mask = (
        (stats_df["mood"] == mood)
        & (stats_df["title"] == title)
        & (stats_df["source"] == source)
    )
    now = datetime.utcnow().isoformat()
    if mask.any():
        idx = stats_df[mask].index[0]
        stats_df.loc[idx, "reward_sum"] = float(stats_df.loc[idx, "reward_sum"]) + reward
        stats_df.loc[idx, "count"] = int(stats_df.loc[idx, "count"]) + 1
        stats_df.loc[idx, "avg_reward"] = float(stats_df.loc[idx, "reward_sum"]) / max(
            int(stats_df.loc[idx, "count"]), 1
        )
        stats_df.loc[idx, "last_updated"] = now
    else:
        stats_df = pd.concat(
            [
                stats_df,
                pd.DataFrame(
                    [
                        {
                            "mood": mood,
                            "title": title,
                            "source": source,
                            "reward_sum": reward,
                            "count": 1,
                            "avg_reward": reward,
                            "last_updated": now,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    stats_df.to_csv(RECOMMENDER_STATS_PATH, index=False)


def recommend_resources(
    catalog_df: pd.DataFrame,
    mood: str,
    seen_titles: set,
    n_items: int = 5,
) -> pd.DataFrame:
    stats_df = load_recommender_stats()
    pool = catalog_df[catalog_df["mood"] == mood].copy()
    if pool.empty:
        pool = catalog_df.copy()
    pool = pool.merge(stats_df[["mood", "title", "source", "avg_reward", "count"]], how="left")
    pool["avg_reward"] = pool["avg_reward"].fillna(0.0)
    pool["count"] = pool["count"].fillna(0)
    pool["exploration"] = np.random.uniform(0.0, 0.1, size=len(pool))
    pool["score"] = (0.75 * pool["avg_reward"]) + (0.15 * np.log1p(pool["count"])) + pool[
        "exploration"
    ]
    pool = pool.sort_values("score", ascending=False)

    selected_rows = []
    for _, row in pool.iterrows():
        if row["title"] in seen_titles:
            continue
        selected_rows.append(row)
        if len(selected_rows) >= n_items:
            break

    if len(selected_rows) < n_items:
        # Reset no-repeat memory when pool is exhausted
        seen_titles.clear()
        for _, row in pool.head(n_items).iterrows():
            selected_rows.append(row)
            if len(selected_rows) >= n_items:
                break

    result = pd.DataFrame(selected_rows).drop(columns=["avg_reward", "count", "exploration", "score"], errors="ignore")
    return result.reset_index(drop=True)


def spotify_recommendations(
    query: str,
    client_id: str,
    client_secret: str,
    limit: int = 5,
) -> Tuple[pd.DataFrame, str]:
    if not SPOTIPY_AVAILABLE:
        return pd.DataFrame(), "Spotipy is not installed."
    if not client_id or not client_secret:
        return pd.DataFrame(), "Spotify credentials missing."
    try:
        auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
        sp_client = spotipy.Spotify(auth_manager=auth_manager)
        search_result = sp_client.search(q=query, type="track", limit=limit)
        items = search_result.get("tracks", {}).get("items", [])
        rows = []
        for item in items:
            artist = ", ".join(artist["name"] for artist in item.get("artists", []))
            rows.append(
                {
                    "title": item.get("name", "Unknown"),
                    "artist": artist,
                    "url": item.get("external_urls", {}).get("spotify", ""),
                    "source": "Spotify API",
                }
            )
        return pd.DataFrame(rows), ""
    except Exception as exc:
        return pd.DataFrame(), f"Spotify API error: {exc}"


def detect_image_label_columns(example_row: Dict[str, object]) -> Tuple[Optional[str], Optional[str]]:
    image_candidates = ["image", "img", "face", "pixel_values", "pixels", "photo"]
    label_candidates = ["label", "emotion", "target", "sentiment", "class"]
    image_col, label_col = None, None
    for key in example_row.keys():
        key_lower = key.lower()
        if image_col is None and key_lower in image_candidates:
            image_col = key
        if label_col is None and key_lower in label_candidates:
            label_col = key
    if image_col is None:
        for key, value in example_row.items():
            if isinstance(value, Image.Image):
                image_col = key
                break
            if isinstance(value, dict) and ("bytes" in value or "path" in value):
                image_col = key
                break
    if label_col is None:
        for key, value in example_row.items():
            if isinstance(value, (int, np.integer, str)):
                key_lower = key.lower()
                if "id" not in key_lower and "index" not in key_lower:
                    label_col = key
                    break
    return image_col, label_col


def _image_from_pixel_string(pixel_string: str) -> Optional[Image.Image]:
    values = pixel_string.strip().split()
    if not values:
        return None
    try:
        arr = np.array(values, dtype=np.uint8)
    except Exception:
        return None
    side = int(math.sqrt(arr.size))
    if side * side != arr.size:
        return None
    arr = arr.reshape(side, side)
    return Image.fromarray(arr).convert("RGB")


def extract_pil_image(raw_value: object) -> Optional[Image.Image]:
    if raw_value is None:
        return None
    if isinstance(raw_value, Image.Image):
        return raw_value.convert("RGB")
    if isinstance(raw_value, np.ndarray):
        if raw_value.ndim == 2:
            return Image.fromarray(raw_value.astype(np.uint8)).convert("RGB")
        return Image.fromarray(raw_value.astype(np.uint8)).convert("RGB")
    if isinstance(raw_value, dict):
        if "bytes" in raw_value and raw_value["bytes"] is not None:
            try:
                return Image.open(io.BytesIO(raw_value["bytes"])).convert("RGB")
            except Exception:
                return None
        if "path" in raw_value and raw_value["path"]:
            path = str(raw_value["path"])
            if os.path.exists(path):
                try:
                    return Image.open(path).convert("RGB")
                except Exception:
                    return None
    if isinstance(raw_value, str):
        if os.path.exists(raw_value):
            try:
                return Image.open(raw_value).convert("RGB")
            except Exception:
                return None
        return _image_from_pixel_string(raw_value)
    return None


def can_stratify(labels: List[str]) -> bool:
    if len(set(labels)) < 2:
        return False
    counts = Counter(labels)
    return all(count >= 2 for count in counts.values())


def prepare_sampled_dataset(
    dataset_key: str,
    sample_size: int,
    image_size: int,
) -> Tuple[bool, str, Dict[str, object]]:
    source = DATASET_SOURCES.get(dataset_key)
    if source is None:
        return False, "Unknown dataset selection.", {}
    if not source["auto_download"]:
        return (
            False,
            f"{dataset_key} is manual-only due license/access constraints.",
            {},
        )
    if not HF_DATASETS_AVAILABLE:
        return (
            False,
            f"`datasets` package not available: {HF_DATASET_IMPORT_ERROR}",
            {},
        )
    repo_id = source["repo_id"]
    split = source["split"]

    try:
        ds_stream = load_dataset(repo_id, split=split, streaming=True)
    except Exception as exc:
        return False, f"Unable to stream dataset `{repo_id}`: {exc}", {}

    sampled: List[Tuple[Image.Image, str]] = []
    image_col = None
    label_col = None
    inspected = 0
    for row in ds_stream:
        inspected += 1
        if image_col is None or label_col is None:
            image_col, label_col = detect_image_label_columns(row)
        if image_col is None:
            continue
        image = extract_pil_image(row.get(image_col))
        if image is None:
            continue
        raw_label = row.get(label_col) if label_col is not None else "neutral"
        label = normalize_emotion(raw_label)
        image = image.resize((image_size, image_size))
        sampled.append((image, label))
        if len(sampled) >= sample_size:
            break
        if inspected >= sample_size * 40:
            break

    if len(sampled) < 80:
        return (
            False,
            "Too few valid samples extracted. Try different dataset or lower sample size.",
            {"extracted": len(sampled), "inspected": inspected},
        )

    labels = [label for _, label in sampled]
    idx_all = list(range(len(sampled)))
    stratify_labels = labels if can_stratify(labels) else None
    train_idx, temp_idx = train_test_split(
        idx_all,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=stratify_labels,
    )
    temp_labels = [labels[idx] for idx in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temp_labels if can_stratify(temp_labels) else None,
    )

    for split_name in ["train", "val", "test"]:
        split_dir = DATASET_DIR / split_name
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)

    def _write_split(indices: Iterable[int], split_name: str) -> Dict[str, int]:
        split_counter = Counter()
        for local_idx, idx in enumerate(indices):
            image, label = sampled[idx]
            out_dir = DATASET_DIR / split_name / label
            out_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{dataset_key.lower().replace('-', '_')}_{local_idx:06d}.jpg"
            image.save(out_dir / filename, format="JPEG", quality=94)
            split_counter[label] += 1
        return dict(split_counter)

    train_counts = _write_split(train_idx, "train")
    val_counts = _write_split(val_idx, "val")
    test_counts = _write_split(test_idx, "test")

    manifest = {
        "dataset_key": dataset_key,
        "repo_id": repo_id,
        "sample_size_requested": sample_size,
        "sample_size_extracted": len(sampled),
        "image_size": image_size,
        "created_at": datetime.utcnow().isoformat(),
        "splits": {
            "train": train_counts,
            "val": val_counts,
            "test": test_counts,
        },
    }
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)
    return True, "Dataset prepared successfully.", manifest


def build_custom_gap_cnn(num_classes: int, image_size: int) -> models.Model:
    inputs = layers.Input(shape=(image_size, image_size, 3))
    x = layers.Rescaling(1.0 / 255.0)(inputs)
    for filters in [32, 64, 128]:
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(256, 3, padding="same", activation="relu", name="last_conv")(x)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs, name="cei_cnn_gap")


def build_efficientnet_gap(num_classes: int, image_size: int) -> models.Model:
    base = tf.keras.applications.EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
        input_shape=(image_size, image_size, 3),
    )
    base.trainable = False
    inputs = layers.Input(shape=(image_size, image_size, 3))
    x = tf.keras.applications.efficientnet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.Conv2D(256, 1, padding="same", activation="relu", name="last_conv")(x)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs, name="cei_efficientnetv2_gap")


def load_training_datasets(image_size: int, batch_size: int):
    train_dir = DATASET_DIR / "train"
    val_dir = DATASET_DIR / "val"
    test_dir = DATASET_DIR / "test"
    if not train_dir.exists():
        raise FileNotFoundError("dataset/train not found. Prepare dataset first.")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode="categorical",
        seed=RANDOM_SEED,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode="categorical",
        seed=RANDOM_SEED,
    )
    test_ds = None
    if test_dir.exists():
        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            image_size=(image_size, image_size),
            batch_size=batch_size,
            label_mode="categorical",
            seed=RANDOM_SEED,
            shuffle=False,
        )
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    if test_ds is not None:
        test_ds = test_ds.prefetch(autotune)
    return train_ds, val_ds, test_ds, train_ds.class_names


def collect_predictions(model, ds) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    true_labels = []
    pred_labels = []
    prob_list = []
    for batch_x, batch_y in ds:
        probs = model.predict(batch_x, verbose=0)
        y_true = np.argmax(batch_y.numpy(), axis=1)
        y_pred = np.argmax(probs, axis=1)
        true_labels.extend(y_true.tolist())
        pred_labels.extend(y_pred.tolist())
        prob_list.append(probs)
    return np.array(true_labels), np.array(pred_labels), np.vstack(prob_list)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
) -> Dict[str, object]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    roc_auc = None
    try:
        y_true_onehot = np.eye(len(class_names))[y_true]
        roc_auc = roc_auc_score(y_true_onehot, y_prob, multi_class="ovr")
    except Exception:
        roc_auc = None
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def train_cnn_model(
    architecture: str,
    image_size: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
) -> Dict[str, object]:
    if not TF_AVAILABLE:
        raise RuntimeError(f"TensorFlow unavailable: {TF_IMPORT_ERROR}")
    train_ds, val_ds, test_ds, class_names = load_training_datasets(image_size, batch_size)
    num_classes = len(class_names)
    if architecture == "EfficientNetV2B0-GAP":
        model = build_efficientnet_gap(num_classes, image_size)
    else:
        model = build_custom_gap_cnn(num_classes, image_size)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True
        )
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)

    eval_ds = test_ds if test_ds is not None else val_ds
    y_true, y_pred, y_prob = collect_predictions(model, eval_ds)
    metrics = evaluate_predictions(y_true, y_pred, y_prob, class_names)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    metadata = {
        "class_names": class_names,
        "image_size": image_size,
        "architecture": architecture,
        "trained_at": datetime.utcnow().isoformat(),
        "metrics": metrics,
        "history": history.history,
    }
    with open(MODEL_META_PATH, "w", encoding="utf-8") as meta_file:
        json.dump(metadata, meta_file, indent=2)
    return metadata


def find_last_conv_layer_name(model) -> Optional[str]:
    if not TF_AVAILABLE:
        return None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None


def make_gradcam_heatmap(
    img_array: np.ndarray,
    model,
    last_conv_layer_name: str,
    pred_index: Optional[int] = None,
) -> np.ndarray:
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = int(tf.argmax(predictions[0]))
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = tf.reduce_sum(conv_output * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap + 1e-9)
    return heatmap.numpy()


def overlay_heatmap_on_image(
    original_image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45
) -> Image.Image:
    import matplotlib.cm as cm

    heatmap_uint8 = np.uint8(255 * heatmap)
    colormap = cm.get_cmap("jet")
    colorized = colormap(heatmap_uint8)[:, :, :3]
    colorized_img = Image.fromarray((colorized * 255).astype(np.uint8))
    colorized_img = colorized_img.resize(original_image.size)
    base = original_image.convert("RGB")
    return Image.blend(base, colorized_img, alpha=alpha)


def generate_gradcam_visual(
    model,
    pil_image: Image.Image,
    image_size: int,
) -> Tuple[Image.Image, int, float]:
    arr = preprocess_image(pil_image, image_size)
    probs = model.predict(arr, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_conf = float(probs[pred_idx])
    last_conv = find_last_conv_layer_name(model)
    if not last_conv:
        raise ValueError("No convolution layer found for Grad-CAM.")
    heatmap = make_gradcam_heatmap(arr, model, last_conv, pred_idx)
    overlay = overlay_heatmap_on_image(pil_image, heatmap)
    return overlay, pred_idx, pred_conf


def render_markdown_block(text: str) -> None:
    st.markdown(textwrap.dedent(text).strip())


def streamlit_main() -> None:
    st.set_page_config(
        page_title="CEI Adaptive Lifestyle System",
        page_icon="🧠",
        layout="wide",
    )
    st.title("Cognitive Emotional Intelligence & Adaptive Lifestyle System")
    st.caption(
        "Single-file Streamlit major project with multimodal emotion fusion, free API options, "
        "CNN+GAP training, Grad-CAM explainability, ethical AI monitoring, and digital twin logging."
    )

    catalog_df = ensure_catalog_exists()

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]
    if "seen_titles" not in st.session_state:
        st.session_state.seen_titles = set()
    if "latest_recommendations" not in st.session_state:
        st.session_state.latest_recommendations = []
    if "latest_mood" not in st.session_state:
        st.session_state.latest_mood = "neutral"

    with st.sidebar:
        st.subheader("Configuration")
        user_id = st.text_input("User ID", value="student_user")
        hf_token = st.text_input("Hugging Face Token (optional, free tier)", type="password")
        spotify_client_id = st.text_input("Spotify Client ID (optional)", type="password")
        spotify_client_secret = st.text_input("Spotify Client Secret (optional)", type="password")
        use_hf_text_classifier = st.checkbox("Use Hugging Face emotion classifier", value=False)
        use_hf_question_gen = st.checkbox("Use Hugging Face question generator", value=False)

        st.markdown("---")
        st.markdown("**Ethical AI Note**")
        st.info(
            "Outputs are advisory and not medical diagnosis. "
            "Always use human judgment for sensitive emotional decisions."
        )

    model, class_names, model_image_size = load_saved_model(*model_paths_signature())

    tab_runtime, tab_data, tab_train, tab_logs, tab_setup = st.tabs(
        [
            "Runtime CEI System",
            "Automatic Dataset Preparation",
            "Train CNN + Grad-CAM",
            "Results & Logs",
            "API + Deployment Guide",
        ]
    )

    with tab_runtime:
        st.subheader("Multimodal Emotion Fusion and Recommendation")
        col_a, col_b = st.columns(2)
        with col_a:
            face_upload = st.file_uploader(
                "Upload face image (jpg/png)", type=["jpg", "jpeg", "png"], key="face_upload"
            )
            emoji_choice = st.selectbox("Emoji mood input", options=list(EMOJI_TO_EMOTION.keys()))
            text_context = st.text_area(
                "Text context (mindset, day summary, stress/workload, etc.)",
                height=150,
                placeholder="Example: I have viva tomorrow and I feel nervous but motivated.",
            )
        with col_b:
            audio_upload = st.file_uploader(
                "Voice note for transcription (wav/mp3/m4a, optional)",
                type=["wav", "mp3", "m4a", "ogg"],
                key="audio_upload",
            )
            manual_voice_text = st.text_area(
                "Voice transcript fallback (if no API token)",
                height=110,
                placeholder="Type transcript manually if automatic ASR is unavailable.",
            )
            top_n = st.slider("Recommendations to display", 3, 10, 5)

        run_button = st.button("Analyze Emotion + Generate Recommendations", type="primary")

        if run_button:
            face_image = None
            if face_upload is not None:
                face_image = Image.open(face_upload).convert("RGB")
                st.image(face_image, caption="Input Face Image", width=280)

            modalities: Dict[str, Tuple[str, float]] = {}
            diagnostics: Dict[str, Dict[str, float]] = {}

            face_result = face_emotion_inference(face_image, model, class_names, model_image_size)
            if face_result is not None:
                modalities["face"] = (face_result[0], face_result[1])
                diagnostics["face_distribution"] = face_result[2]

            emoji_emotion = EMOJI_TO_EMOTION.get(emoji_choice, "neutral")
            modalities["emoji"] = (emoji_emotion, 0.8)

            text_result = None
            if text_context.strip():
                if use_hf_text_classifier and hf_token.strip():
                    text_result = huggingface_text_emotion(text_context, hf_token)
                if text_result is None:
                    text_result = local_text_emotion(text_context)
                modalities["text"] = (text_result[0], text_result[1])
                diagnostics["text_distribution"] = text_result[2]

            transcript = ""
            if audio_upload is not None:
                audio_bytes = audio_upload.read()
                transcript = huggingface_transcribe_audio(audio_bytes, hf_token) if hf_token else ""
                if not transcript:
                    transcript = manual_voice_text.strip()
            elif manual_voice_text.strip():
                transcript = manual_voice_text.strip()

            if transcript:
                voice_result = (
                    huggingface_text_emotion(transcript, hf_token)
                    if (use_hf_text_classifier and hf_token.strip())
                    else None
                )
                if voice_result is None:
                    voice_result = local_text_emotion(transcript)
                modalities["voice"] = (voice_result[0], voice_result[1])
                diagnostics["voice_distribution"] = voice_result[2]

            fused_emotion, fused_confidence, fused_distribution = fuse_emotions(modalities)
            risk_level, warnings = ethical_ai_monitor(fused_emotion, fused_confidence, modalities)

            st.markdown("### Fused Emotion Output")
            c1, c2, c3 = st.columns(3)
            c1.metric("Final Emotion", fused_emotion.title())
            c2.metric("Confidence", f"{fused_confidence:.2f}")
            c3.metric("Risk Level", risk_level.title())
            st.bar_chart(pd.DataFrame([fused_distribution]).T.rename(columns={0: "score"}))

            if warnings:
                for warning in warnings:
                    st.warning(warning)

            st.markdown("### Modality-wise Outputs")
            modality_rows = []
            for modality_name in ["face", "text", "voice", "emoji"]:
                if modality_name in modalities:
                    modality_rows.append(
                        {
                            "modality": modality_name,
                            "emotion": modalities[modality_name][0],
                            "confidence": round(modalities[modality_name][1], 4),
                        }
                    )
            st.dataframe(pd.DataFrame(modality_rows), use_container_width=True)

            recommendations = recommend_resources(
                catalog_df,
                mood=fused_emotion,
                seen_titles=st.session_state.seen_titles,
                n_items=top_n,
            )
            if not recommendations.empty:
                st.markdown("### History-aware Recommendations (No Repetition)")
                for _, rec in recommendations.iterrows():
                    st.session_state.seen_titles.add(rec["title"])
                st.dataframe(recommendations, use_container_width=True)
                st.session_state.latest_recommendations = recommendations.to_dict(orient="records")
                st.session_state.latest_mood = fused_emotion

            spotify_df = pd.DataFrame()
            spotify_msg = ""
            if spotify_client_id and spotify_client_secret:
                spotify_query = f"{fused_emotion} mood music"
                spotify_df, spotify_msg = spotify_recommendations(
                    spotify_query, spotify_client_id, spotify_client_secret, limit=5
                )
            if spotify_msg:
                st.info(spotify_msg)
            if not spotify_df.empty:
                st.markdown("### Spotify API Suggestions (Optional)")
                st.dataframe(spotify_df, use_container_width=True)

            questions = []
            if use_hf_question_gen and hf_token.strip():
                questions = huggingface_generate_questions(fused_emotion, text_context, hf_token)
            if not questions:
                questions = fallback_questions(fused_emotion, text_context)
            st.markdown("### Important Mindset Questions")
            for idx, question in enumerate(questions, start=1):
                st.write(f"{idx}. {question}")

            append_twin_log(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "session_id": st.session_state.session_id,
                    "user_id": user_id,
                    "fused_emotion": fused_emotion,
                    "fused_confidence": round(fused_confidence, 4),
                    "face_emotion": modalities.get("face", ("", ""))[0],
                    "face_confidence": modalities.get("face", ("", ""))[1],
                    "text_emotion": modalities.get("text", ("", ""))[0],
                    "text_confidence": modalities.get("text", ("", ""))[1],
                    "voice_emotion": modalities.get("voice", ("", ""))[0],
                    "voice_confidence": modalities.get("voice", ("", ""))[1],
                    "emoji_emotion": modalities.get("emoji", ("", ""))[0],
                    "emoji_confidence": modalities.get("emoji", ("", ""))[1],
                    "risk_level": risk_level,
                    "warnings": " | ".join(warnings),
                    "recommended_titles": ", ".join(
                        rec["title"] for rec in st.session_state.latest_recommendations
                    ),
                    "feedback": "",
                }
            )
            st.success("Digital Emotional Twin log updated.")

        if st.session_state.latest_recommendations:
            st.markdown("---")
            st.subheader("Feedback for RL-style Recommender Update")
            with st.form("feedback_form"):
                feedback_inputs = {}
                for idx, rec in enumerate(st.session_state.latest_recommendations):
                    feedback_inputs[idx] = st.selectbox(
                        f"{rec['title']} ({rec['source']})",
                        options=["No feedback", "Helpful (+1)", "Not helpful (-1)"],
                        key=f"fb_{idx}",
                    )
                submitted = st.form_submit_button("Submit Feedback")
            if submitted:
                mood = st.session_state.latest_mood
                updates = 0
                for idx, rec in enumerate(st.session_state.latest_recommendations):
                    choice = feedback_inputs.get(idx, "No feedback")
                    if choice.startswith("Helpful"):
                        update_recommender_feedback(mood, rec["title"], rec["source"], 1)
                        updates += 1
                    elif choice.startswith("Not helpful"):
                        update_recommender_feedback(mood, rec["title"], rec["source"], -1)
                        updates += 1
                st.success(f"Recommender stats updated with {updates} feedback entries.")

    with tab_data:
        st.subheader("Automatic Dataset Preparation (Sample Mode)")
        st.write(
            "Recommended for 4 GB RAM laptops: sample size 600-1200, image size 96x96, "
            "batch size 4 or 8, epochs 1-3."
        )
        selected_dataset = st.selectbox("Dataset source", options=list(DATASET_SOURCES.keys()))
        st.caption(DATASET_SOURCES[selected_dataset]["notes"])
        sample_size = st.slider("Sample size", min_value=200, max_value=1500, value=900, step=100)
        image_size = st.select_slider("Image size", options=[64, 96, 128, 160], value=96)

        if st.button("Prepare Dataset Automatically"):
            with st.spinner("Preparing dataset..."):
                ok, msg, manifest = prepare_sampled_dataset(selected_dataset, sample_size, image_size)
            if ok:
                st.success(msg)
                st.json(manifest)
            else:
                st.error(msg)
                if manifest:
                    st.write(manifest)

        if MANIFEST_PATH.exists():
            st.markdown("### Latest dataset_manifest.json")
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                st.json(json.load(f))

    with tab_train:
        st.subheader("Train Upgraded CNN + GAP + Grad-CAM")
        if not TF_AVAILABLE:
            st.error(f"TensorFlow is unavailable in this environment: {TF_IMPORT_ERROR}")
        else:
            architecture = st.selectbox(
                "Architecture",
                options=["Custom-CNN-GAP", "EfficientNetV2B0-GAP"],
                help=(
                    "Custom-CNN-GAP is recommended for low-RAM systems. "
                    "EfficientNetV2B0-GAP is stronger but heavier."
                ),
            )
            epochs = st.slider("Epochs", 1, 8, 2)
            batch_size = st.select_slider("Batch size", options=[4, 8, 16], value=8)
            train_image_size = st.select_slider("Training image size", options=[64, 96, 128, 160], value=96)
            learning_rate = st.select_slider(
                "Learning rate",
                options=[1e-4, 3e-4, 5e-4, 1e-3],
                value=1e-3,
            )

            if st.button("Start Training"):
                try:
                    with st.spinner("Training model..."):
                        metadata = train_cnn_model(
                            architecture=architecture,
                            image_size=train_image_size,
                            batch_size=batch_size,
                            epochs=epochs,
                            learning_rate=learning_rate,
                        )
                    st.success("Training completed and model saved.")
                    metrics = metadata.get("metrics", {})
                    st.write("### Evaluation Metrics")
                    metric_cols = st.columns(4)
                    metric_cols[0].metric("Precision", f"{metrics.get('precision', 0.0):.4f}")
                    metric_cols[1].metric("Recall", f"{metrics.get('recall', 0.0):.4f}")
                    metric_cols[2].metric("F1", f"{metrics.get('f1', 0.0):.4f}")
                    metric_cols[3].metric(
                        "ROC-AUC", f"{metrics.get('roc_auc'):.4f}" if metrics.get("roc_auc") is not None else "N/A"
                    )
                    cm = np.array(metrics.get("confusion_matrix", []))
                    if cm.size > 0:
                        st.write("Confusion Matrix")
                        cm_df = pd.DataFrame(cm, index=metadata["class_names"], columns=metadata["class_names"])
                        st.dataframe(cm_df, use_container_width=True)
                    st.code(metrics.get("classification_report", ""), language="text")
                    load_saved_model.clear()
                except Exception as exc:
                    st.error(f"Training failed: {exc}")

            st.markdown("### Grad-CAM Explainability")
            gradcam_img_file = st.file_uploader(
                "Upload an image for Grad-CAM", type=["jpg", "jpeg", "png"], key="gradcam_upload"
            )
            if gradcam_img_file is not None and st.button("Generate Grad-CAM"):
                model_current, class_names_current, img_size_current = load_saved_model(*model_paths_signature())
                if model_current is None:
                    st.error("No trained model found. Train the model first.")
                else:
                    try:
                        original = Image.open(gradcam_img_file).convert("RGB")
                        overlay, pred_idx, pred_conf = generate_gradcam_visual(
                            model_current, original, img_size_current
                        )
                        pred_label = (
                            class_names_current[pred_idx]
                            if pred_idx < len(class_names_current)
                            else "unknown"
                        )
                        c1, c2 = st.columns(2)
                        with c1:
                            st.image(original, caption="Original", use_column_width=True)
                        with c2:
                            st.image(overlay, caption="Grad-CAM Overlay", use_column_width=True)
                        st.info(f"Predicted emotion: {pred_label} (confidence: {pred_conf:.4f})")
                    except Exception as exc:
                        st.error(f"Grad-CAM generation failed: {exc}")

    with tab_logs:
        st.subheader("Results, Logs, and Exports")
        if TWIN_LOG_PATH.exists():
            twin_df = pd.read_csv(TWIN_LOG_PATH)
            st.write("Digital Emotional Twin Log")
            st.dataframe(twin_df.tail(200), use_container_width=True)
        else:
            st.info("No emotional twin log found yet.")

        if RECOMMENDER_STATS_PATH.exists():
            stats_df = pd.read_csv(RECOMMENDER_STATS_PATH)
            st.write("Recommender Reward Statistics")
            st.dataframe(
                stats_df.sort_values("avg_reward", ascending=False).head(100),
                use_container_width=True,
            )
        else:
            st.info("No recommender stats found yet.")

        if MODEL_META_PATH.exists():
            with open(MODEL_META_PATH, "r", encoding="utf-8") as meta_file:
                metadata = json.load(meta_file)
            st.write("Model Metadata")
            st.json(metadata)

        st.download_button(
            "Download Resource Catalog CSV",
            data=CATALOG_PATH.read_bytes() if CATALOG_PATH.exists() else b"",
            file_name="resource_catalog.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download CEI Twin Log CSV",
            data=TWIN_LOG_PATH.read_bytes() if TWIN_LOG_PATH.exists() else b"",
            file_name="cei_twin_log.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download Recommender Stats CSV",
            data=RECOMMENDER_STATS_PATH.read_bytes() if RECOMMENDER_STATS_PATH.exists() else b"",
            file_name="recommender_stats.csv",
            mime="text/csv",
        )

    with tab_setup:
        st.subheader("Free API + Deployment + APK Trick")
        render_markdown_block(
            """
            ### Free API integration strategy
            1. **Primary free path:** Hugging Face Inference API token (free tier).
            2. **Fallback path:** Local keyword emotion classifier + local question bank (no paid API).
            3. **Optional paid path:** OpenAI/Spotify can be plugged in, but not required.

            ### Why Hugging Face is suitable here
            - Free token creation is available for students.
            - Hosts useful models for emotion classification, ASR, and text generation.
            - Works well as a single API provider for this project.

            ### How to create Hugging Face token
            1. Create account at `https://huggingface.co/join`
            2. Open Settings -> Access Tokens
            3. Create new token with **Read** scope
            4. Paste token into the app sidebar

            ### Free installable APK-like experience
            - Run Streamlit app on local network/cloud.
            - On Android Chrome: **Add to Home Screen**.
            - On Windows Edge/Chrome: **Install app**.
            - This gives a zero-cost installable PWA-style experience.

            ### If a true APK is mandatory (still free)
            - Use WebView wrapper toolchain (Kivy/Buildozer or Android Studio WebView wrapper).
            - Keep same backend logic from this single file.
            """
        )

        st.markdown("### Suggested terminal commands")
        st.code(
            "\n".join(
                [
                    "python -m venv .venv",
                    ".venv\\Scripts\\Activate.ps1   # Windows PowerShell",
                    "pip install -r requirements.txt",
                    "streamlit run app.py",
                    "python app.py --export-catalog",
                ]
            ),
            language="bash",
        )


def main() -> None:
    if export_catalog_cli_if_requested():
        return
    streamlit_main()


if __name__ == "__main__":
    main()
