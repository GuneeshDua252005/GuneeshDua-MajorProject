import argparse
import csv
import io
import json
import os
import random
import textwrap
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import speech_recognition as sr
import streamlit as st
from datasets import ClassLabel, load_dataset
from PIL import Image, ImageDraw
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

try:
    import tensorflow as tf

    TF_AVAILABLE = True
    TF_IMPORT_ERROR = ""
except Exception as tf_error:  # pragma: no cover - runtime fallback
    tf = None
    TF_AVAILABLE = False
    TF_IMPORT_ERROR = str(tf_error)

try:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials

    SPOTIFY_AVAILABLE = True
except Exception:
    SPOTIFY_AVAILABLE = False


ROOT_DIR = Path(__file__).resolve().parent
DATASET_DIR = ROOT_DIR / "dataset"
MODELS_DIR = ROOT_DIR / "models"
CATALOG_CSV = ROOT_DIR / "resource_catalog.csv"
TWIN_LOG_CSV = ROOT_DIR / "cei_twin_log.csv"
RECOMMENDER_STATS_CSV = ROOT_DIR / "recommender_stats.csv"
MANIFEST_JSON = DATASET_DIR / "dataset_manifest.json"
MODEL_FILE = MODELS_DIR / "mobile_transfer.keras"
MODEL_META_FILE = MODELS_DIR / "mobile_transfer_metadata.json"


DEFAULT_EMOTIONS = [
    "happy",
    "sad",
    "angry",
    "fear",
    "surprise",
    "neutral",
    "calm",
    "motivated",
]

EMOJI_TO_EMOTION = {
    "😀": "happy",
    "😄": "happy",
    "🙂": "happy",
    "😊": "calm",
    "😌": "calm",
    "😢": "sad",
    "😭": "sad",
    "😠": "angry",
    "😡": "angry",
    "😨": "fear",
    "😱": "surprise",
    "😐": "neutral",
    "🤔": "neutral",
    "💪": "motivated",
    "🔥": "motivated",
}


TEXT_EMOTION_LEXICON = {
    "happy": [
        "happy",
        "joy",
        "great",
        "amazing",
        "good",
        "excited",
        "smile",
        "awesome",
        "fun",
    ],
    "sad": [
        "sad",
        "down",
        "lonely",
        "upset",
        "cry",
        "depressed",
        "hurt",
        "tired",
        "hopeless",
    ],
    "angry": [
        "angry",
        "mad",
        "annoyed",
        "irritated",
        "frustrated",
        "furious",
        "rage",
    ],
    "fear": [
        "afraid",
        "scared",
        "fear",
        "anxious",
        "nervous",
        "panic",
        "worried",
    ],
    "surprise": ["surprised", "shock", "unexpected", "wow", "astonished"],
    "calm": ["calm", "peaceful", "relaxed", "slow", "quiet", "stable"],
    "motivated": ["motivated", "focus", "discipline", "goal", "productive", "strong"],
    "neutral": ["okay", "normal", "fine", "average", "regular", "neutral"],
}


RISK_KEYWORDS = [
    "self harm",
    "suicide",
    "kill myself",
    "die",
    "end my life",
    "overdose",
    "hopeless",
    "can't go on",
]


DATASET_OPTIONS = {
    "FER2025 (auto-download enabled)": {
        "hf_id": "nateraw/fer2013",
        "split": "train",
        "auto": True,
    },
    "EmoNet-Face-Big (auto-download enabled)": {
        "hf_id": "nateraw/fer2013",
        "split": "train",
        "auto": True,
    },
    "MER2024 (manual only)": {
        "hf_id": "",
        "split": "train",
        "auto": False,
    },
    "MER2023 (manual only)": {
        "hf_id": "",
        "split": "train",
        "auto": False,
    },
    "Custom Hugging Face dataset": {
        "hf_id": "",
        "split": "train",
        "auto": True,
    },
    "Synthetic emotion dataset (offline fallback)": {
        "hf_id": "",
        "split": "train",
        "auto": True,
    },
}


def ensure_project_dirs() -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / "train").mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / "val").mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / "test").mkdir(parents=True, exist_ok=True)


def now_ts() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def normalize_label(raw_label: str) -> str:
    label = str(raw_label).strip().lower()
    synonyms = {
        "joy": "happy",
        "happiness": "happy",
        "positive": "happy",
        "anger": "angry",
        "disgust": "angry",
        "negative": "sad",
        "fearful": "fear",
        "surprised": "surprise",
        "calmness": "calm",
        "neutrality": "neutral",
    }
    return synonyms.get(label, label.replace(" ", "_"))


def append_csv(path: Path, row: Dict[str, Any], headers: List[str]) -> None:
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def detect_ethics_risk(text: str) -> Tuple[bool, str]:
    text_lower = (text or "").lower()
    for keyword in RISK_KEYWORDS:
        if keyword in text_lower:
            return (
                True,
                "Potential mental-health risk signal detected. This app is not emergency care. "
                "Please connect with local emergency support, a trusted person, or a certified counselor immediately.",
            )
    return False, "No explicit high-risk keyword detected."


def infer_text_emotion(text: str) -> Tuple[str, float, Dict[str, float]]:
    cleaned = (text or "").strip().lower()
    if not cleaned:
        return "neutral", 0.0, {emotion: 0.0 for emotion in DEFAULT_EMOTIONS}

    scores = {emotion: 0.0 for emotion in DEFAULT_EMOTIONS}
    for emotion, words in TEXT_EMOTION_LEXICON.items():
        score = 0
        for token in words:
            if token in cleaned:
                score += 1
        scores[emotion] = float(score)

    max_emotion = max(scores, key=scores.get)
    max_score = scores[max_emotion]
    if max_score <= 0:
        return "neutral", 0.3, scores

    norm = max_score / (sum(scores.values()) + 1e-9)
    return max_emotion, float(min(1.0, 0.45 + norm)), scores


def infer_emoji_emotion(emoji_text: str) -> Tuple[str, float]:
    if not emoji_text:
        return "neutral", 0.0
    emotion = EMOJI_TO_EMOTION.get(emoji_text.strip(), "neutral")
    return emotion, 0.75


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def infer_face_emotion(image: Image.Image) -> Tuple[str, float, Dict[str, float]]:
    bgr = pil_to_bgr(image)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(40, 40))
    face_count = len(faces)
    smile_count = 0
    if face_count > 0:
        for (x, y, w, h) in faces:
            roi_gray = gray[y : y + h, x : x + w]
            smiles = smile_cascade.detectMultiScale(
                roi_gray, scaleFactor=1.7, minNeighbors=20, minSize=(20, 20)
            )
            smile_count += len(smiles)

    scores = {emotion: 0.0 for emotion in DEFAULT_EMOTIONS}
    if smile_count > 0:
        scores["happy"] += 0.8
        scores["motivated"] += 0.2
    if brightness < 85:
        scores["sad"] += 0.7
    elif brightness > 175:
        scores["surprise"] += 0.6
        scores["happy"] += 0.2
    if sharpness > 450:
        scores["angry"] += 0.45
        scores["motivated"] += 0.2
    if face_count == 0:
        scores["neutral"] += 0.35
        scores["calm"] += 0.2
    else:
        scores["neutral"] += 0.25

    emotion = max(scores, key=scores.get)
    confidence = min(1.0, 0.35 + scores[emotion])
    return emotion, float(confidence), {
        "brightness": brightness,
        "sharpness": sharpness,
        "faces_detected": face_count,
        "smiles_detected": smile_count,
    }


def transcribe_audio(audio_bytes: bytes) -> Tuple[str, str]:
    if not audio_bytes:
        return "", "No audio bytes available."
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
        return text, ""
    except sr.UnknownValueError:
        return "", "Voice was not clear enough to transcribe."
    except Exception as error:
        return "", f"Voice transcription failed: {error}"


def fuse_modalities(
    modality_results: Dict[str, Tuple[str, float]],
    modality_weights: Dict[str, float],
) -> Tuple[str, float, Dict[str, float]]:
    score_map: Dict[str, float] = {}
    total_weight = 0.0
    for modality_name, (emotion, confidence) in modality_results.items():
        weight = modality_weights.get(modality_name, 0.0)
        total_weight += weight
        score_map.setdefault(emotion, 0.0)
        score_map[emotion] += confidence * weight

    if not score_map or total_weight <= 0:
        return "neutral", 0.0, {"neutral": 0.0}

    fused_emotion = max(score_map, key=score_map.get)
    max_value = score_map[fused_emotion]
    confidence = max_value / (total_weight + 1e-9)
    return fused_emotion, float(min(1.0, confidence)), score_map


def build_resource_catalog() -> pd.DataFrame:
    mood_prompts = {
        "happy": ["joy playlist", "feel good pop", "sunny day music"],
        "sad": ["healing songs", "gentle acoustic comfort", "soft piano calm"],
        "angry": ["release energy rock", "focus workout beats", "power metal focus"],
        "fear": ["anxiety relief sounds", "deep breathing music", "safe ambient tones"],
        "surprise": ["fresh discoveries", "unexpected indie mix", "experimental electronic"],
        "neutral": ["balanced background music", "productive instrumentals", "daily routine mix"],
        "calm": ["meditation music", "nature relaxation", "sleep wind down"],
        "motivated": ["high focus playlist", "study motivation tracks", "goal setting anthems"],
        "romantic": ["romantic classics", "date night smooth", "love songs acoustic"],
        "nostalgic": ["retro hits", "90s throwback", "old school memories"],
    }
    video_ids = [
        "5qap5aO4i9A",
        "DWcJFNfaw9c",
        "jfKfPfyJRdk",
        "lTRiuFIWV54",
        "1ZYbU82GVz4",
        "hHW1oY26kxQ",
    ]

    rows: List[Dict[str, Any]] = []
    index = 0
    for mood, prompts in mood_prompts.items():
        for prompt in prompts:
            encoded = urllib.parse.quote_plus(prompt)
            for suffix in ["mix", "playlist", "focus", "relaxation"]:
                title = f"{mood.title()} {prompt.title()} {suffix.title()}"
                rows.append(
                    {
                        "mood": mood,
                        "title": title,
                        "url": f"https://www.youtube.com/watch?v={video_ids[index % len(video_ids)]}",
                        "source": "YouTube",
                        "type": "direct",
                        "offline_fallback_guidance": "Use locally saved wellness playlist in VLC/Windows Media Player.",
                    }
                )
                rows.append(
                    {
                        "mood": mood,
                        "title": f"{title} Search",
                        "url": f"https://www.youtube.com/results?search_query={encoded}+{suffix}",
                        "source": "YouTube",
                        "type": "search",
                        "offline_fallback_guidance": "Search in offline downloaded catalog if internet is unavailable.",
                    }
                )
                rows.append(
                    {
                        "mood": mood,
                        "title": f"{title} Spotify",
                        "url": f"https://open.spotify.com/search/{encoded}%20{suffix}",
                        "source": "Spotify",
                        "type": "search",
                        "offline_fallback_guidance": "Use local MP3 playlist with similar mood tags (without auto-download).",
                    }
                )
                rows.append(
                    {
                        "mood": mood,
                        "title": f"{title} YT Music",
                        "url": f"https://music.youtube.com/search?q={encoded}+{suffix}",
                        "source": "YouTube Music",
                        "type": "search",
                        "offline_fallback_guidance": "Use audio files already present on your system and mood tags.",
                    }
                )
                index += 1

    catalog = pd.DataFrame(rows)
    return catalog


def export_catalog(path: Path = CATALOG_CSV) -> int:
    catalog = build_resource_catalog()
    catalog.to_csv(path, index=False)
    return len(catalog)


def recommend_resources(
    catalog: pd.DataFrame,
    mood: str,
    history_urls: set,
    top_k: int = 8,
) -> pd.DataFrame:
    if mood not in catalog["mood"].unique():
        mood = "neutral"

    mood_df = catalog[catalog["mood"] == mood].copy()
    unseen_df = mood_df[~mood_df["url"].isin(history_urls)].copy()

    if unseen_df.empty:
        history_urls.clear()
        unseen_df = mood_df.copy()

    sample_size = min(top_k, len(unseen_df))
    if sample_size <= 0:
        return mood_df.head(0)

    recommended = unseen_df.sample(n=sample_size, random_state=random.randint(0, 100000))
    for url in recommended["url"].tolist():
        history_urls.add(url)
    return recommended.reset_index(drop=True)


def spotify_recommendations(
    mood: str,
    client_id: str,
    client_secret: str,
    limit: int = 5,
) -> List[Dict[str, str]]:
    if not SPOTIFY_AVAILABLE:
        return [{"name": "spotipy not installed", "artist": "", "url": ""}]
    if not client_id or not client_secret:
        return [{"name": "Spotify credentials not configured", "artist": "", "url": ""}]
    try:
        client_credentials = SpotifyClientCredentials(
            client_id=client_id, client_secret=client_secret
        )
        sp = spotipy.Spotify(client_credentials_manager=client_credentials)
        result = sp.search(q=f"{mood} music", type="track", limit=limit)
        output = []
        for item in result.get("tracks", {}).get("items", []):
            output.append(
                {
                    "name": item.get("name", ""),
                    "artist": ", ".join([artist["name"] for artist in item.get("artists", [])]),
                    "url": item.get("external_urls", {}).get("spotify", ""),
                }
            )
        return output if output else [{"name": "No Spotify tracks found", "artist": "", "url": ""}]
    except Exception as error:
        return [{"name": f"Spotify API error: {error}", "artist": "", "url": ""}]


def build_emotional_prompt(
    question: str,
    mood: str,
    user_context: str,
    safety_note: str,
) -> str:
    return textwrap.dedent(
        f"""
        You are an Emotional AI assistant for a Cognitive Emotion Intelligence project.
        User mood: {mood}
        User context: {user_context}
        Safety note: {safety_note}
        Task:
        1) Reply empathetically.
        2) Ask 2 short important follow-up questions.
        3) Give 3 practical actions for the next 24 hours.
        4) Keep response concise, clear, and supportive.
        User question:
        {question}
        """
    ).strip()


def local_emotional_chat(question: str, mood: str, context: str) -> str:
    base_reply = {
        "happy": "Your positive mood is a strong resource today.",
        "sad": "I hear emotional heaviness. Small, structured steps can help stabilize your day.",
        "angry": "High emotional energy can be redirected into safe, constructive actions.",
        "fear": "Anxiety often reduces when we define one safe immediate step.",
        "surprise": "Unexpected moments can feel intense; grounding helps regain clarity.",
        "calm": "Calmness is an opportunity to build healthy momentum.",
        "motivated": "Your motivation can be turned into focused execution.",
        "neutral": "A neutral state is useful for choosing intentional habits.",
    }.get(mood, "Your emotional state is important and valid.")

    return textwrap.dedent(
        f"""
        {base_reply}

        You asked: "{question}"
        Based on your context ("{context[:180]}"), here are the next steps:
        1) Pause for 90 seconds and rate your current emotional intensity from 1 to 10.
        2) Choose one micro-action aligned with your mood (music, walk, hydration, short journaling).
        3) Re-check mood after 20 minutes and adjust your activity.

        Important follow-up questions:
        - What is the biggest trigger affecting your mood right now?
        - Which support (friend, mentor, or routine) helps you recover fastest?
        """
    ).strip()


def hf_chat_completion(prompt: str, hf_token: str, model: str) -> str:
    if not hf_token:
        return "Hugging Face token is missing. Add HF_TOKEN in environment or sidebar."
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 300, "temperature": 0.7, "return_full_text": False},
    }
    response = requests.post(url, headers=headers, json=payload, timeout=90)
    if response.status_code != 200:
        return f"Hugging Face API error {response.status_code}: {response.text[:400]}"
    data = response.json()
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data[0].get("generated_text", "").strip() or str(data[0])
    if isinstance(data, dict) and "generated_text" in data:
        return str(data["generated_text"]).strip()
    return str(data)[:2000]


def openai_chat_completion(prompt: str, api_key: str, model: str = "gpt-4o-mini") -> str:
    if not api_key:
        return "OPENAI_API_KEY missing. Set environment variable or sidebar key."
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a supportive emotional wellness assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 500,
    }
    response = requests.post(url, headers=headers, json=payload, timeout=90)
    if response.status_code != 200:
        return f"OpenAI API error {response.status_code}: {response.text[:400]}"
    data = response.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return str(data)[:2000]


def groq_chat_completion(prompt: str, api_key: str, model: str = "llama-3.1-8b-instant") -> str:
    if not api_key:
        return "GROQ_API_KEY missing. Set environment variable or sidebar key."
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a supportive emotional wellness assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 500,
    }
    response = requests.post(url, headers=headers, json=payload, timeout=90)
    if response.status_code != 200:
        return f"Groq API error {response.status_code}: {response.text[:400]}"
    data = response.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return str(data)[:2000]


def detect_image_and_label_columns(ds_split) -> Tuple[str, str]:
    sample = ds_split[0]
    image_col = ""
    label_col = ""
    for key, value in sample.items():
        if image_col:
            break
        if isinstance(value, Image.Image):
            image_col = key
            break
        if isinstance(value, dict) and ("bytes" in value or "path" in value):
            image_col = key
            break
        if isinstance(value, np.ndarray):
            image_col = key
            break
    for key, value in sample.items():
        if key == image_col:
            continue
        if isinstance(value, (int, np.integer, str)):
            label_col = key
            break
    if not image_col or not label_col:
        raise ValueError("Could not auto-detect image/label columns.")
    return image_col, label_col


def extract_pil_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, np.ndarray):
        return Image.fromarray(value).convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes"):
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path"):
            return Image.open(value["path"]).convert("RGB")
    raise ValueError("Unsupported image format in dataset row.")


def clear_dataset_folders() -> None:
    ensure_project_dirs()
    for split in ["train", "val", "test"]:
        split_path = DATASET_DIR / split
        for child in split_path.glob("*"):
            if child.is_dir():
                for file_path in child.glob("*"):
                    file_path.unlink(missing_ok=True)
                child.rmdir()


def save_split_items(
    rows: List[Tuple[Image.Image, str]],
    split_name: str,
    image_size: int = 224,
) -> Dict[str, int]:
    split_path = DATASET_DIR / split_name
    split_path.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, int] = {}

    for idx, (image, label) in enumerate(rows):
        label_norm = normalize_label(label)
        label_dir = split_path / label_norm
        label_dir.mkdir(parents=True, exist_ok=True)
        save_path = label_dir / f"{split_name}_{idx:05d}.jpg"
        image.convert("RGB").resize((image_size, image_size)).save(save_path, format="JPEG", quality=92)
        counts[label_norm] = counts.get(label_norm, 0) + 1
    return counts


def create_synthetic_emotion_image(label: str, size: int = 224) -> Image.Image:
    color_map = {
        "happy": (255, 223, 90),
        "sad": (100, 149, 237),
        "angry": (240, 96, 96),
        "fear": (160, 120, 200),
        "surprise": (255, 170, 120),
        "neutral": (180, 180, 180),
    }
    background = color_map.get(label, (180, 180, 180))
    image = Image.new("RGB", (size, size), color=background)
    draw = ImageDraw.Draw(image)

    cx, cy = size // 2, size // 2
    radius = size // 3
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=(0, 0, 0), width=4)
    eye_offset_x = size // 9
    eye_offset_y = size // 10
    draw.ellipse(
        (cx - eye_offset_x - 8, cy - eye_offset_y - 8, cx - eye_offset_x + 8, cy - eye_offset_y + 8),
        fill=(0, 0, 0),
    )
    draw.ellipse(
        (cx + eye_offset_x - 8, cy - eye_offset_y - 8, cx + eye_offset_x + 8, cy - eye_offset_y + 8),
        fill=(0, 0, 0),
    )

    if label == "happy":
        draw.arc((cx - 50, cy - 20, cx + 50, cy + 50), start=20, end=160, fill=(0, 0, 0), width=4)
    elif label == "sad":
        draw.arc((cx - 50, cy + 10, cx + 50, cy + 70), start=200, end=340, fill=(0, 0, 0), width=4)
    elif label == "angry":
        draw.line((cx - 50, cy + 45, cx + 50, cy + 45), fill=(0, 0, 0), width=5)
    elif label == "surprise":
        draw.ellipse((cx - 18, cy + 10, cx + 18, cy + 50), outline=(0, 0, 0), width=4)
    elif label == "fear":
        draw.arc((cx - 40, cy + 10, cx + 40, cy + 60), start=180, end=360, fill=(0, 0, 0), width=3)
    else:
        draw.line((cx - 40, cy + 40, cx + 40, cy + 40), fill=(0, 0, 0), width=4)
    return image


def prepare_synthetic_dataset(sample_size: int = 900, seed: int = 42) -> Dict[str, Any]:
    ensure_project_dirs()
    clear_dataset_folders()
    random.seed(seed)
    np.random.seed(seed)

    labels = ["happy", "sad", "angry", "fear", "surprise", "neutral"]
    rows: List[Tuple[Image.Image, str]] = []
    for _ in range(sample_size):
        label = random.choice(labels)
        image = create_synthetic_emotion_image(label)
        rows.append((image, label))

    random.shuffle(rows)
    n = len(rows)
    train_rows = rows[: int(0.7 * n)]
    val_rows = rows[int(0.7 * n) : int(0.85 * n)]
    test_rows = rows[int(0.85 * n) :]

    train_counts = save_split_items(train_rows, "train")
    val_counts = save_split_items(val_rows, "val")
    test_counts = save_split_items(test_rows, "test")

    manifest = {
        "timestamp": now_ts(),
        "source": "synthetic_offline",
        "sample_size": sample_size,
        "splits": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
        },
        "labels": sorted(set([label for _, label in rows])),
        "counts_per_split": {"train": train_counts, "val": val_counts, "test": test_counts},
        "notes": "Synthetic fallback dataset generated locally for demo/training compatibility.",
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def prepare_dataset_from_hf(
    dataset_id: str,
    split_name: str = "train",
    sample_size: int = 900,
    seed: int = 42,
) -> Dict[str, Any]:
    ensure_project_dirs()
    clear_dataset_folders()
    random.seed(seed)
    np.random.seed(seed)

    ds = load_dataset(dataset_id, split=split_name)
    if len(ds) == 0:
        raise ValueError("Dataset split is empty.")

    image_col, label_col = detect_image_and_label_columns(ds)
    indices = list(range(len(ds)))
    random.shuffle(indices)
    picked = indices[: min(sample_size, len(indices))]

    rows: List[Tuple[Image.Image, str]] = []
    label_decoder = None
    if label_col in ds.features and isinstance(ds.features[label_col], ClassLabel):
        label_decoder = ds.features[label_col]

    for idx in picked:
        row = ds[idx]
        image = extract_pil_image(row[image_col])
        label_raw = row[label_col]
        if label_decoder is not None and isinstance(label_raw, (int, np.integer)):
            label_text = label_decoder.int2str(int(label_raw))
        else:
            label_text = str(label_raw)
        rows.append((image, normalize_label(label_text)))

    random.shuffle(rows)
    n = len(rows)
    train_rows = rows[: int(0.7 * n)]
    val_rows = rows[int(0.7 * n) : int(0.85 * n)]
    test_rows = rows[int(0.85 * n) :]

    train_counts = save_split_items(train_rows, "train")
    val_counts = save_split_items(val_rows, "val")
    test_counts = save_split_items(test_rows, "test")

    manifest = {
        "timestamp": now_ts(),
        "source": dataset_id,
        "split": split_name,
        "sample_size": len(rows),
        "columns": {"image": image_col, "label": label_col},
        "splits": {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)},
        "labels": sorted(set([label for _, label in rows])),
        "counts_per_split": {"train": train_counts, "val": val_counts, "test": test_counts},
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_transfer_model(num_classes: int, image_size: int = 224):
    if not TF_AVAILABLE:
        raise RuntimeError(f"TensorFlow is not available: {TF_IMPORT_ERROR}")

    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="image_input")
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    base_model = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(image_size, image_size, 3),
        name="base_mobilenetv2",
    )
    base_model.trainable = False
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dropout(0.25, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="cei_mobilenet_transfer")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_dataset_tf(data_dir: Path, image_size: int, batch_size: int):
    if not TF_AVAILABLE:
        raise RuntimeError(f"TensorFlow is not available: {TF_IMPORT_ERROR}")
    ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=True,
    )
    return ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)


def evaluate_model(model, test_ds, class_names: List[str]) -> Dict[str, Any]:
    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[np.ndarray] = []
    for batch_images, batch_labels in test_ds:
        probs = model.predict(batch_images, verbose=0)
        preds = np.argmax(probs, axis=1)
        y_true.extend(batch_labels.numpy().tolist())
        y_pred.extend(preds.tolist())
        y_prob.extend(probs.tolist())

    y_prob_arr = np.array(y_prob)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    roc_auc_value = None
    try:
        if len(class_names) == 2:
            roc_auc_value = roc_auc_score(y_true, y_prob_arr[:, 1])
        elif len(class_names) > 2:
            y_true_oh = tf.keras.utils.to_categorical(y_true, num_classes=len(class_names))
            roc_auc_value = roc_auc_score(
                y_true_oh,
                y_prob_arr,
                multi_class="ovr",
                average="weighted",
            )
    except Exception:
        roc_auc_value = None

    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc_value) if roc_auc_value is not None else None,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def train_and_evaluate(
    image_size: int = 224,
    batch_size: int = 8,
    epochs: int = 2,
) -> Dict[str, Any]:
    ensure_project_dirs()
    train_dir = DATASET_DIR / "train"
    val_dir = DATASET_DIR / "val"
    test_dir = DATASET_DIR / "test"
    if not train_dir.exists():
        raise FileNotFoundError("dataset/train not found. Run dataset preparation first.")

    train_ds = build_dataset_tf(train_dir, image_size=image_size, batch_size=batch_size)
    class_names = train_ds.class_names
    val_ds = None
    if val_dir.exists() and any(val_dir.iterdir()):
        val_ds = build_dataset_tf(val_dir, image_size=image_size, batch_size=batch_size)

    model = build_transfer_model(num_classes=len(class_names), image_size=image_size)
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=1)

    metrics_payload = {}
    if test_dir.exists() and any(test_dir.iterdir()):
        test_ds = build_dataset_tf(test_dir, image_size=image_size, batch_size=batch_size)
        metrics_payload = evaluate_model(model, test_ds, class_names=class_names)

    model.save(MODEL_FILE)
    metadata = {
        "timestamp": now_ts(),
        "image_size": image_size,
        "batch_size": batch_size,
        "epochs": epochs,
        "class_names": class_names,
        "history": {k: [float(x) for x in v] for k, v in history.history.items()},
        "metrics": {
            "precision": metrics_payload.get("precision"),
            "recall": metrics_payload.get("recall"),
            "f1": metrics_payload.get("f1"),
            "roc_auc": metrics_payload.get("roc_auc"),
        },
    }
    MODEL_META_FILE.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "model_path": str(MODEL_FILE),
        "metadata_path": str(MODEL_META_FILE),
        "class_names": class_names,
        "history": history.history,
        "metrics": metrics_payload,
    }


def load_model_and_classes():
    if not TF_AVAILABLE:
        raise RuntimeError(f"TensorFlow is not available: {TF_IMPORT_ERROR}")
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Trained model not found at {MODEL_FILE}")
    if not MODEL_META_FILE.exists():
        raise FileNotFoundError(f"Model metadata not found at {MODEL_META_FILE}")

    model = tf.keras.models.load_model(MODEL_FILE)
    metadata = json.loads(MODEL_META_FILE.read_text(encoding="utf-8"))
    class_names = metadata.get("class_names", [])
    image_size = int(metadata.get("image_size", 224))
    return model, class_names, image_size


def preprocess_for_model(image: Image.Image, image_size: int) -> np.ndarray:
    resized = image.convert("RGB").resize((image_size, image_size))
    arr = np.array(resized).astype("float32")
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict_emotion_model(image: Image.Image) -> Tuple[str, float, Dict[str, float]]:
    model, class_names, image_size = load_model_and_classes()
    arr = preprocess_for_model(image, image_size=image_size)
    probs = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(probs))
    emotion = class_names[idx]
    confidence = float(probs[idx])
    distribution = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    return emotion, confidence, distribution


def make_gradcam_overlay(image: Image.Image) -> Image.Image:
    model, _, image_size = load_model_and_classes()
    arr = preprocess_for_model(image, image_size=image_size)

    base_layer = model.get_layer("base_mobilenetv2")
    grad_model = tf.keras.models.Model([model.inputs], [base_layer.output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(arr)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    gradients = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(gradients, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-9)
    heatmap_np = heatmap.numpy()

    rgb = np.array(image.convert("RGB"))
    heatmap_resized = cv2.resize(heatmap_np, (rgb.shape[1], rgb.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), 0.6, colored, 0.4, 0)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return Image.fromarray(overlay_rgb)


def log_twin_event(
    user_id: str,
    fused_emotion: str,
    confidence: float,
    face_emotion: str,
    emoji_emotion: str,
    text_emotion: str,
    voice_text: str,
    user_text: str,
    recommendation_titles: str,
    chat_question: str,
    chat_answer: str,
    ethics_flag: bool,
) -> None:
    headers = [
        "timestamp",
        "user_id",
        "fused_emotion",
        "confidence",
        "face_emotion",
        "emoji_emotion",
        "text_emotion",
        "voice_text",
        "user_text",
        "recommendation_titles",
        "chat_question",
        "chat_answer",
        "ethics_flag",
    ]
    row = {
        "timestamp": now_ts(),
        "user_id": user_id,
        "fused_emotion": fused_emotion,
        "confidence": round(confidence, 4),
        "face_emotion": face_emotion,
        "emoji_emotion": emoji_emotion,
        "text_emotion": text_emotion,
        "voice_text": voice_text,
        "user_text": user_text,
        "recommendation_titles": recommendation_titles,
        "chat_question": chat_question,
        "chat_answer": chat_answer,
        "ethics_flag": ethics_flag,
    }
    append_csv(TWIN_LOG_CSV, row=row, headers=headers)


def log_recommender_stats(
    user_id: str,
    mood: str,
    recommended_count: int,
    catalog_size: int,
    unique_seen: int,
) -> None:
    headers = [
        "timestamp",
        "user_id",
        "mood",
        "recommended_count",
        "catalog_size",
        "unique_seen",
    ]
    row = {
        "timestamp": now_ts(),
        "user_id": user_id,
        "mood": mood,
        "recommended_count": recommended_count,
        "catalog_size": catalog_size,
        "unique_seen": unique_seen,
    }
    append_csv(RECOMMENDER_STATS_CSV, row=row, headers=headers)


def render_confusion_matrix(cm: np.ndarray, class_names: List[str]) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def init_state() -> None:
    if "history_urls" not in st.session_state:
        st.session_state.history_urls = set()
    if "last_fused_emotion" not in st.session_state:
        st.session_state.last_fused_emotion = "neutral"
    if "last_fused_confidence" not in st.session_state:
        st.session_state.last_fused_confidence = 0.0
    if "last_voice_text" not in st.session_state:
        st.session_state.last_voice_text = ""
    if "last_user_text" not in st.session_state:
        st.session_state.last_user_text = ""
    if "last_face_emotion" not in st.session_state:
        st.session_state.last_face_emotion = "neutral"
    if "last_emoji_emotion" not in st.session_state:
        st.session_state.last_emoji_emotion = "neutral"
    if "last_text_emotion" not in st.session_state:
        st.session_state.last_text_emotion = "neutral"


def run_streamlit_app() -> None:
    ensure_project_dirs()
    init_state()

    st.set_page_config(page_title="Cognitive Emotion Intelligence System", layout="wide")
    st.title("Cognitive Emotion Intelligence & Adaptive Lifestyle System")
    st.caption(
        "Single-file Streamlit implementation: multimodal emotion fusion, Digital Emotional Twin logging, "
        "music/lifestyle recommendation, optional API chat integration, dataset prep, transfer learning, and Grad-CAM."
    )

    with st.sidebar:
        st.subheader("Runtime Configuration")
        user_id = st.text_input("User ID", value="demo_user")
        text_weight = st.slider("Text Weight", 0.0, 1.0, 0.40, 0.05)
        face_weight = st.slider("Face Weight", 0.0, 1.0, 0.30, 0.05)
        emoji_weight = st.slider("Emoji Weight", 0.0, 1.0, 0.20, 0.05)
        voice_weight = st.slider("Voice Weight", 0.0, 1.0, 0.10, 0.05)

        st.markdown("---")
        st.subheader("Optional API Keys")
        openai_key = st.text_input(
            "OPENAI_API_KEY",
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password",
            help="Paid API option.",
        )
        hf_token = st.text_input(
            "HF_TOKEN",
            value=os.getenv("HF_TOKEN", ""),
            type="password",
            help="Free tier option from Hugging Face account token.",
        )
        groq_key = st.text_input(
            "GROQ_API_KEY",
            value=os.getenv("GROQ_API_KEY", ""),
            type="password",
            help="Recommended free chat API alternative.",
        )
        spotify_id = st.text_input(
            "SPOTIPY_CLIENT_ID",
            value=os.getenv("SPOTIPY_CLIENT_ID", ""),
            type="password",
        )
        spotify_secret = st.text_input(
            "SPOTIPY_CLIENT_SECRET",
            value=os.getenv("SPOTIPY_CLIENT_SECRET", ""),
            type="password",
        )
        st.info(
            "Security note: APIs require token-based auth (keys/secrets), not raw account passwords. "
            "Never hardcode keys into source files."
        )

    tabs = st.tabs(
        [
            "1) Emotion Fusion",
            "2) Emotional AI Chat",
            "3) No-Repetition Recommender",
            "4) Dataset Preparation",
            "5) CNN Training + Evaluation",
            "6) Grad-CAM Explainability",
            "7) Deployment & Free API Guide",
        ]
    )

    with tabs[0]:
        st.subheader("Multi-Modal Emotion Input and Fusion")
        col1, col2 = st.columns(2)
        with col1:
            face_upload = st.file_uploader("Upload Face Image", type=["jpg", "jpeg", "png"])
            selected_emoji = st.selectbox(
                "Select Emoji",
                options=["", "😀", "😊", "😢", "😠", "😨", "😱", "😐", "💪", "🔥"],
            )
        with col2:
            audio_input = st.audio_input("Voice input (optional)")
            text_input = st.text_area(
                "Text context",
                placeholder="Describe how you feel, what happened, and what help you need.",
                height=130,
            )

        if st.button("Analyze Emotion", type="primary"):
            modality_results: Dict[str, Tuple[str, float]] = {}
            face_emotion = "neutral"
            emoji_emotion = "neutral"
            text_emotion = "neutral"
            voice_text = ""

            if face_upload is not None:
                image = Image.open(face_upload).convert("RGB")
                face_emotion, face_conf, face_debug = infer_face_emotion(image)
                modality_results["face"] = (face_emotion, face_conf)
                st.image(image, caption="Face input", width=280)
                st.write(f"Face emotion: **{face_emotion}** (confidence: {face_conf:.2f})")
                st.json(face_debug)

            if selected_emoji:
                emoji_emotion, emoji_conf = infer_emoji_emotion(selected_emoji)
                modality_results["emoji"] = (emoji_emotion, emoji_conf)
                st.write(f"Emoji emotion: **{emoji_emotion}** (confidence: {emoji_conf:.2f})")

            if audio_input is not None:
                voice_text, voice_error = transcribe_audio(audio_input.read())
                if voice_error:
                    st.warning(voice_error)
                if voice_text:
                    voice_emotion, voice_conf, _ = infer_text_emotion(voice_text)
                    modality_results["voice"] = (voice_emotion, voice_conf)
                    st.write(f"Voice transcript: `{voice_text}`")
                    st.write(f"Voice emotion: **{voice_emotion}** (confidence: {voice_conf:.2f})")

            if text_input.strip():
                text_emotion, text_conf, text_scores = infer_text_emotion(text_input)
                modality_results["text"] = (text_emotion, text_conf)
                st.write(f"Text emotion: **{text_emotion}** (confidence: {text_conf:.2f})")
                st.json(text_scores)

            if not modality_results:
                st.warning("Please provide at least one modality (face/emoji/voice/text).")
            else:
                weights = {
                    "text": text_weight,
                    "face": face_weight,
                    "emoji": emoji_weight,
                    "voice": voice_weight,
                }
                fused_emotion, fused_confidence, fusion_scores = fuse_modalities(modality_results, weights)
                st.session_state.last_fused_emotion = fused_emotion
                st.session_state.last_fused_confidence = fused_confidence
                st.session_state.last_voice_text = voice_text
                st.session_state.last_user_text = text_input
                st.session_state.last_face_emotion = face_emotion
                st.session_state.last_emoji_emotion = emoji_emotion
                st.session_state.last_text_emotion = text_emotion

                st.success(
                    f"Fused Emotion: **{fused_emotion}** | Confidence: **{fused_confidence:.2f}**"
                )
                st.json(fusion_scores)

                ethics_flag, ethics_msg = detect_ethics_risk((text_input or "") + " " + (voice_text or ""))
                if ethics_flag:
                    st.error(ethics_msg)
                else:
                    st.info(ethics_msg)

                log_twin_event(
                    user_id=user_id,
                    fused_emotion=fused_emotion,
                    confidence=fused_confidence,
                    face_emotion=face_emotion,
                    emoji_emotion=emoji_emotion,
                    text_emotion=text_emotion,
                    voice_text=voice_text,
                    user_text=text_input,
                    recommendation_titles="",
                    chat_question="",
                    chat_answer="",
                    ethics_flag=ethics_flag,
                )
                st.caption(f"Digital Emotional Twin log updated: {TWIN_LOG_CSV.name}")

        st.markdown(
            f"Current fused emotion state: **{st.session_state.last_fused_emotion}** "
            f"({st.session_state.last_fused_confidence:.2f})"
        )

    with tabs[1]:
        st.subheader("Chat with Emotional AI")
        chat_provider = st.selectbox(
            "Provider",
            [
                "Local (Free Offline)",
                "Hugging Face (Free Tier)",
                "Groq (Recommended Free Option)",
                "OpenAI (Paid)",
            ],
        )
        question = st.text_area("Ask an important life/mood question", height=120)
        user_context = st.text_area(
            "Optional extra context",
            value=st.session_state.last_user_text,
            height=90,
        )

        if st.button("Ask Emotional AI"):
            mood = st.session_state.last_fused_emotion
            ethics_flag, ethics_message = detect_ethics_risk((question or "") + " " + (user_context or ""))
            if ethics_flag:
                st.error(ethics_message)
            prompt = build_emotional_prompt(
                question=question,
                mood=mood,
                user_context=user_context,
                safety_note="Prioritize safe and non-harmful guidance.",
            )

            if chat_provider == "Local (Free Offline)":
                answer = local_emotional_chat(question=question, mood=mood, context=user_context)
            elif chat_provider == "Hugging Face (Free Tier)":
                answer = hf_chat_completion(
                    prompt=prompt,
                    hf_token=hf_token,
                    model="HuggingFaceH4/zephyr-7b-beta",
                )
            elif chat_provider == "Groq (Recommended Free Option)":
                answer = groq_chat_completion(prompt=prompt, api_key=groq_key)
            else:
                answer = openai_chat_completion(prompt=prompt, api_key=openai_key)

            st.markdown("#### Emotional AI Response")
            st.write(answer)
            log_twin_event(
                user_id=user_id,
                fused_emotion=mood,
                confidence=st.session_state.last_fused_confidence,
                face_emotion=st.session_state.last_face_emotion,
                emoji_emotion=st.session_state.last_emoji_emotion,
                text_emotion=st.session_state.last_text_emotion,
                voice_text=st.session_state.last_voice_text,
                user_text=user_context,
                recommendation_titles="",
                chat_question=question,
                chat_answer=answer,
                ethics_flag=ethics_flag,
            )

    with tabs[2]:
        st.subheader("History-Aware No-Repetition Recommender")
        catalog = build_resource_catalog()
        mood = st.selectbox(
            "Mood for recommendation",
            options=sorted(catalog["mood"].unique().tolist()),
            index=sorted(catalog["mood"].unique().tolist()).index(
                st.session_state.last_fused_emotion
                if st.session_state.last_fused_emotion in set(catalog["mood"])
                else "neutral"
            ),
        )
        top_k = st.slider("Number of recommendations", min_value=3, max_value=15, value=8)

        if st.button("Generate Recommendations", type="primary"):
            recommended_df = recommend_resources(
                catalog=catalog,
                mood=mood,
                history_urls=st.session_state.history_urls,
                top_k=top_k,
            )
            if recommended_df.empty:
                st.warning("No recommendations found.")
            else:
                st.dataframe(recommended_df, use_container_width=True)
                st.markdown("#### Quick links")
                for _, row in recommended_df.iterrows():
                    st.markdown(f"- [{row['title']}]({row['url']}) ({row['source']}, {row['type']})")

                recommendation_titles = " | ".join(recommended_df["title"].tolist())
                log_twin_event(
                    user_id=user_id,
                    fused_emotion=mood,
                    confidence=st.session_state.last_fused_confidence,
                    face_emotion=st.session_state.last_face_emotion,
                    emoji_emotion=st.session_state.last_emoji_emotion,
                    text_emotion=st.session_state.last_text_emotion,
                    voice_text=st.session_state.last_voice_text,
                    user_text=st.session_state.last_user_text,
                    recommendation_titles=recommendation_titles,
                    chat_question="",
                    chat_answer="",
                    ethics_flag=False,
                )
                log_recommender_stats(
                    user_id=user_id,
                    mood=mood,
                    recommended_count=len(recommended_df),
                    catalog_size=len(catalog),
                    unique_seen=len(st.session_state.history_urls),
                )
                st.success("Recommendations generated and logs updated.")

        st.markdown("#### Spotify live search (optional)")
        if st.button("Fetch Spotify Suggestions"):
            spotify_rows = spotify_recommendations(
                mood=mood,
                client_id=spotify_id,
                client_secret=spotify_secret,
                limit=5,
            )
            st.dataframe(pd.DataFrame(spotify_rows), use_container_width=True)

        if st.button("Export Resource Catalog CSV"):
            count = export_catalog(path=CATALOG_CSV)
            st.success(f"Exported {count} catalog entries to {CATALOG_CSV.name}")

    with tabs[3]:
        st.subheader("Automatic Dataset Preparation")
        source_choice = st.selectbox("Dataset option", options=list(DATASET_OPTIONS.keys()))
        sample_size = st.slider("Sample size", min_value=300, max_value=1500, value=900, step=100)
        split_name = st.selectbox("Source split", options=["train", "validation", "test"], index=0)
        custom_hf = st.text_input("Custom Hugging Face dataset ID", value="nateraw/fer2013")
        seed = st.number_input("Random seed", value=42, step=1)

        if st.button("Prepare Dataset", type="primary"):
            info = DATASET_OPTIONS[source_choice]
            if not info["auto"] and "MER" in source_choice:
                st.warning(
                    "MER2023/MER2024 are marked as manual-only due to potential access terms. "
                    "Use approved local ZIP import into dataset/train, dataset/val, dataset/test."
                )
            else:
                with st.spinner("Preparing dataset..."):
                    try:
                        if "Synthetic" in source_choice:
                            manifest = prepare_synthetic_dataset(sample_size=int(sample_size), seed=int(seed))
                        else:
                            hf_id = custom_hf if "Custom" in source_choice else info["hf_id"]
                            manifest = prepare_dataset_from_hf(
                                dataset_id=hf_id,
                                split_name=split_name,
                                sample_size=int(sample_size),
                                seed=int(seed),
                            )
                        st.success("Dataset prepared successfully.")
                        st.json(manifest)
                    except Exception as prep_error:
                        st.error(f"Auto-download failed: {prep_error}")
                        st.info("Switching to synthetic fallback dataset for guaranteed compatibility.")
                        manifest = prepare_synthetic_dataset(sample_size=int(sample_size), seed=int(seed))
                        st.success("Synthetic dataset prepared successfully.")
                        st.json(manifest)

        if MANIFEST_JSON.exists():
            st.markdown("#### Current dataset_manifest.json")
            st.code(MANIFEST_JSON.read_text(encoding="utf-8"), language="json")

    with tabs[4]:
        st.subheader("Transfer Learning with MobileNetV2")
        if not TF_AVAILABLE:
            st.error(
                "TensorFlow unavailable in this environment. "
                f"Import error: {TF_IMPORT_ERROR}"
            )
        else:
            st.info(
                "For Intel i5 + 4GB RAM, keep sample size around 600-1200, batch size 4 or 8, "
                "epochs 1-3 for demo runs."
            )
            image_size = st.selectbox("Image size", [160, 192, 224], index=2)
            batch_size = st.selectbox("Batch size", [4, 8, 16], index=1)
            epochs = st.slider("Epochs", min_value=1, max_value=6, value=2)

            if st.button("Train and Evaluate CNN", type="primary"):
                with st.spinner("Training model..."):
                    try:
                        result = train_and_evaluate(
                            image_size=int(image_size),
                            batch_size=int(batch_size),
                            epochs=int(epochs),
                        )
                        st.success("Training complete and model saved.")
                        st.write(f"Model: `{result['model_path']}`")
                        st.write(f"Metadata: `{result['metadata_path']}`")
                        metrics = result.get("metrics", {})
                        if metrics:
                            st.markdown("#### Evaluation Metrics")
                            st.write(f"Precision: **{metrics.get('precision', 0):.4f}**")
                            st.write(f"Recall: **{metrics.get('recall', 0):.4f}**")
                            st.write(f"F1 Score: **{metrics.get('f1', 0):.4f}**")
                            if metrics.get("roc_auc") is not None:
                                st.write(f"ROC-AUC: **{metrics['roc_auc']:.4f}**")
                            else:
                                st.write("ROC-AUC: Not available for current configuration.")
                            st.text(metrics.get("classification_report", ""))
                            cm = metrics.get("confusion_matrix")
                            if cm is not None:
                                render_confusion_matrix(cm, class_names=result["class_names"])
                    except Exception as train_error:
                        st.error(f"Training failed: {train_error}")

    with tabs[5]:
        st.subheader("Grad-CAM Explainability")
        st.caption("Requires trained model at models/mobile_transfer.keras")
        infer_upload = st.file_uploader(
            "Upload image for model inference + Grad-CAM",
            type=["jpg", "jpeg", "png"],
            key="gradcam_upload",
        )
        if st.button("Run Inference + Grad-CAM"):
            if infer_upload is None:
                st.warning("Please upload an image first.")
            else:
                image = Image.open(infer_upload).convert("RGB")
                st.image(image, caption="Input image", width=300)
                try:
                    pred_emotion, pred_conf, distribution = predict_emotion_model(image)
                    st.success(f"Predicted emotion: **{pred_emotion}** ({pred_conf:.2f})")
                    st.bar_chart(pd.DataFrame({"probability": distribution}))
                    overlay = make_gradcam_overlay(image)
                    st.image(overlay, caption="Grad-CAM overlay", width=300)
                except Exception as infer_error:
                    st.error(f"Inference/Grad-CAM failed: {infer_error}")

    with tabs[6]:
        st.subheader("Step-by-step deployment guide (VS Code + Windows 11)")
        st.markdown(
            """
            ### Required installs in terminal
            1. `python -m venv .venv`
            2. `\\.venv\\Scripts\\activate`
            3. `pip install -r requirements.txt`
            4. `streamlit run app.py`

            ### Essential files in project folder
            - `app.py` (single executable source code)
            - `requirements.txt`
            - `.gitignore`
            - `README.md`
            - Runtime-generated files: `cei_twin_log.csv`, `recommender_stats.csv`, `resource_catalog.csv`, `dataset/`, `models/`

            ### Free API integration guidance
            - **Hugging Face**: Free tier token from account settings (recommended for budget usage with rate limits).
            - **Groq**: OpenAI-compatible free API key with generous starter tier (recommended practical free upgrade).
            - **OpenAI**: Paid, optional.
            - **Spotify**: Paid/commercial usage constraints; optional.

            You cannot securely authenticate production APIs using only plain user ID/password in app logic.
            Use account-generated API tokens/secrets and keep them in environment variables.

            ### Why use/avoid Hugging Face
            - Use when you want free experimentation and open models.
            - Limitations: rate limits, cold start delays, and occasional model queue time.
            - If those limits impact UX, use **Groq free API** as the best practical upgrade path.

            ### FREE APK conversion trick
            - Host Streamlit app on a URL.
            - Android Chrome -> "Add to Home Screen" for PWA-like install.
            - Optional free wrapper: create a simple WebView app using Android Studio (no paid tool needed).
            - For Windows 11: Edge/Chrome -> "Install app".
            """
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CEI major project single-file application.")
    parser.add_argument(
        "--export-catalog",
        action="store_true",
        help="Export 100+ resource catalog to resource_catalog.csv and exit.",
    )
    parser.add_argument(
        "--prepare-dataset",
        action="store_true",
        help="Prepare dataset directly from CLI and exit.",
    )
    parser.add_argument(
        "--hf-dataset-id",
        type=str,
        default="nateraw/fer2013",
        help="Hugging Face dataset ID for CLI dataset preparation.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split name for CLI dataset preparation.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=900,
        help="Sample size for dataset preparation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Force synthetic dataset generation in CLI mode.",
    )
    args, _unknown = parser.parse_known_args()
    return args


def main() -> None:
    ensure_project_dirs()
    args = parse_args()
    if args.export_catalog:
        count = export_catalog()
        print(f"Exported {count} rows to {CATALOG_CSV}")
        return

    if args.prepare_dataset:
        if args.synthetic:
            manifest = prepare_synthetic_dataset(sample_size=args.sample_size, seed=args.seed)
        else:
            try:
                manifest = prepare_dataset_from_hf(
                    dataset_id=args.hf_dataset_id,
                    split_name=args.split,
                    sample_size=args.sample_size,
                    seed=args.seed,
                )
            except Exception as error:
                print(f"HF dataset failed ({error}), switching to synthetic fallback...")
                manifest = prepare_synthetic_dataset(sample_size=args.sample_size, seed=args.seed)
        print(json.dumps(manifest, indent=2))
        return

    run_streamlit_app()


if __name__ == "__main__":
    main()
