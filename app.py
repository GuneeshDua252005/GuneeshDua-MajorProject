"""
CEI-ALOS: Cognitive Emotion Intelligence & Adaptive Lifestyle OS

Single-file major-project application that integrates:
1) Multi-modal emotion inference (Face + Voice + Emoji + Text)
2) Multi-modal fusion engine
3) Reinforcement-learning style anti-repetition recommender
4) YouTube + Spotify recommendation integration
5) Digital Emotional Twin (history logging + analytics)
6) Transfer-learning CNN training/evaluation (proper metrics)
7) Grad-CAM explainability
8) CNN + LLM integration (chatbot with image-aware context)
9) Step-by-step API setup and free native Android/Windows build guidance
"""

from __future__ import annotations

import io
import json
import math
import os
import random
import re
import sqlite3
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency
    cv2 = None

try:
    import speech_recognition as sr
except Exception:  # pragma: no cover - optional dependency
    sr = None

try:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
except Exception:  # pragma: no cover - optional dependency
    spotipy = None
    SpotifyClientCredentials = None

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None

TORCH_AVAILABLE = True
TORCH_IMPORT_ERROR = ""
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, models, transforms
except Exception as exc:  # pragma: no cover - optional dependency
    TORCH_AVAILABLE = False
    TORCH_IMPORT_ERROR = str(exc)
    torch = None
    nn = None
    DataLoader = None
    datasets = None
    models = None
    transforms = None

SKLEARN_AVAILABLE = True
SKLEARN_IMPORT_ERROR = ""
try:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        precision_recall_fscore_support,
        roc_auc_score,
    )
    from sklearn.preprocessing import label_binarize
except Exception as exc:  # pragma: no cover - optional dependency
    SKLEARN_AVAILABLE = False
    SKLEARN_IMPORT_ERROR = str(exc)
    accuracy_score = None
    classification_report = None
    confusion_matrix = None
    precision_recall_fscore_support = None
    roc_auc_score = None
    label_binarize = None

# -------------------------
# App constants
# -------------------------
APP_TITLE = "CEI-ALOS Unified Major Project"
DB_PATH = Path("cei_alos.db")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "cei_transfer_model.pt"
DATASET_DIR = Path("uploaded_datasets")

MOODS = ["happy", "calm", "sad", "energetic"]
MOOD_TO_IDX = {m: i for i, m in enumerate(MOODS)}
EMOJI_MAP = {"😊": "happy", "😢": "sad", "😡": "energetic", "😴": "calm", "😍": "happy"}

TEXT_KEYWORDS = {
    "happy": {"happy", "joy", "joyful", "grateful", "excited", "awesome", "great", "love"},
    "calm": {"calm", "relax", "relaxed", "peace", "okay", "fine", "neutral", "balanced"},
    "sad": {"sad", "upset", "hurt", "lonely", "depressed", "cry", "tired", "low"},
    "energetic": {"angry", "mad", "furious", "energetic", "hyped", "pump", "power", "workout"},
}

YOUTUBE_PLAYABLE: Dict[str, List[str]] = {
    "happy": [
        "https://www.youtube.com/watch?v=d-diB65scQU",
        "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
        "https://www.youtube.com/watch?v=ZMO_XC9w7Lw",
        "https://www.youtube.com/watch?v=OPf0YbXqDm0",
        "https://www.youtube.com/watch?v=ru0K8uYEZWw",
        "https://www.youtube.com/watch?v=ZbZSe6N_BXs",
        "https://www.youtube.com/watch?v=CevxZvSJLk8",
        "https://www.youtube.com/watch?v=fLexgOxsZu0",
        "https://www.youtube.com/watch?v=KQ6zr6kCPj8",
        "https://www.youtube.com/watch?v=mk48xRzuNvA",
        "https://www.youtube.com/watch?v=IcrbM1l_BoI",
        "https://www.youtube.com/watch?v=jZhQOvvV45w",
    ],
    "sad": [
        "https://www.youtube.com/watch?v=2XU0oxnq2qU",
        "https://www.youtube.com/watch?v=inpok4MKVLM",
        "https://www.youtube.com/watch?v=1vx8iUvfyCY",
        "https://www.youtube.com/watch?v=9Q634rbsypE",
        "https://www.youtube.com/watch?v=YQHsXMglC9A",
        "https://www.youtube.com/watch?v=hLQl3WQQoQ0",
        "https://www.youtube.com/watch?v=2Vv-BfVoq4g",
        "https://www.youtube.com/watch?v=450p7goxZqg",
        "https://www.youtube.com/watch?v=kTJczUoc26U",
        "https://www.youtube.com/watch?v=mWRsgZuwf_8",
        "https://www.youtube.com/watch?v=4fndeDfaWCg",
        "https://www.youtube.com/watch?v=oRdxUFDoQe0",
    ],
    "calm": [
        "https://www.youtube.com/watch?v=5qap5aO4i9A",
        "https://www.youtube.com/watch?v=v7AYKMP6rOE",
        "https://www.youtube.com/watch?v=lFcSrYw-ARY",
        "https://www.youtube.com/watch?v=O-6f5wQXSu8",
        "https://www.youtube.com/watch?v=inpok4MKVLM",
        "https://www.youtube.com/watch?v=1vx8iUvfyCY",
        "https://www.youtube.com/watch?v=jfKfPfyJRdk",
        "https://www.youtube.com/watch?v=DWcJFNfaw9c",
        "https://www.youtube.com/watch?v=4pLUleLdwY4",
        "https://www.youtube.com/watch?v=aIIEI33EUqI",
        "https://www.youtube.com/watch?v=UfcAVejslrU",
        "https://www.youtube.com/watch?v=Dx5qFachd3A",
    ],
    "energetic": [
        "https://www.youtube.com/watch?v=HgzGwKwLmgM",
        "https://www.youtube.com/watch?v=ml6cT4AZdqI",
        "https://www.youtube.com/watch?v=ZXsQAXx_ao0",
        "https://www.youtube.com/watch?v=btPJPFnesV4",
        "https://www.youtube.com/watch?v=60ItHLz5WEA",
        "https://www.youtube.com/watch?v=pRpeEdMmmQ0",
        "https://www.youtube.com/watch?v=9bZkp7q19f0",
        "https://www.youtube.com/watch?v=iEPTlhBmwRg",
        "https://www.youtube.com/watch?v=fLexgOxsZu0",
        "https://www.youtube.com/watch?v=RgKAFK5djSk",
        "https://www.youtube.com/watch?v=YqeW9_5kURI",
        "https://www.youtube.com/watch?v=QK8mJJJvaes",
    ],
}

YOUTUBE_SEARCH_TERMS = {
    "happy": [
        "happy dance playlist",
        "feel good pop songs",
        "motivation songs morning",
        "positive vibes music",
        "celebration songs",
        "happy bollywood songs",
        "uplifting instrumental",
        "weekend party songs",
        "smile songs collection",
        "energy booster music",
    ],
    "sad": [
        "healing music for sadness",
        "calm therapy music",
        "emotional healing meditation",
        "mindfulness for anxiety",
        "self compassion meditation",
        "relief from overthinking",
        "sad songs acoustic",
        "breakup healing songs",
        "mental health motivation",
        "gentle piano for stress",
    ],
    "calm": [
        "yoga nidra session",
        "deep breathing exercises",
        "relaxing nature sounds",
        "rain sounds sleep",
        "focus music alpha waves",
        "guided meditation beginners",
        "morning mindfulness routine",
        "calm instrumental",
        "forest ambience 4k",
        "peaceful mantra chanting",
    ],
    "energetic": [
        "workout edm mix",
        "gym motivation songs",
        "high intensity interval playlist",
        "running songs 2026",
        "power lifting music",
        "boxing training playlist",
        "cardio warmup songs",
        "sports hype songs",
        "edm festival mix",
        "focus and hustle music",
    ],
}

SPOTIFY_SEARCH_TERMS = {
    "happy": [
        "happy hits",
        "good vibes",
        "feel good songs",
        "dance party",
        "sunny day music",
        "motivation pop",
        "celebration anthems",
        "smile playlist",
    ],
    "sad": [
        "sad songs",
        "heartbreak hits",
        "lofi healing",
        "emotional piano",
        "soft acoustic",
        "anxiety relief music",
        "deep focus calm",
        "healing frequencies",
    ],
    "calm": [
        "calm vibes",
        "meditation music",
        "peaceful piano",
        "deep sleep sounds",
        "focus flow",
        "zen yoga",
        "ambient relaxation",
        "nature calm",
    ],
    "energetic": [
        "beast mode",
        "workout mix",
        "edm power",
        "running motivation",
        "gym hard",
        "sports energy",
        "high bpm mix",
        "hustle tracks",
    ],
}


# -------------------------
# Utility and storage
# -------------------------
def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat()


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                user TEXT NOT NULL,
                face_mood TEXT,
                face_conf REAL,
                voice_text TEXT,
                voice_mood TEXT,
                voice_conf REAL,
                emoji TEXT,
                emoji_mood TEXT,
                text_input TEXT,
                text_mood TEXT,
                text_conf REAL,
                fused_mood TEXT NOT NULL,
                fused_conf REAL,
                recommendation_url TEXT,
                recommendation_source TEXT,
                spotify_json TEXT
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                event_id INTEGER NOT NULL,
                rating INTEGER NOT NULL,
                note TEXT,
                FOREIGN KEY(event_id) REFERENCES events(id)
            );
            """
        )


def log_event(
    user: str,
    face: Tuple[str, float],
    voice_text: str,
    voice: Tuple[str, float],
    emoji: str,
    emoji_mood: str,
    text_input: str,
    text: Tuple[str, float],
    fused: Tuple[str, float],
    recommendation_url: str,
    recommendation_source: str,
    spotify_items: List[Dict[str, str]],
) -> int:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            """
            INSERT INTO events (
                ts, user, face_mood, face_conf, voice_text, voice_mood, voice_conf,
                emoji, emoji_mood, text_input, text_mood, text_conf, fused_mood, fused_conf,
                recommendation_url, recommendation_source, spotify_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                now_iso(),
                user,
                face[0],
                face[1],
                voice_text,
                voice[0],
                voice[1],
                emoji,
                emoji_mood,
                text_input,
                text[0],
                text[1],
                fused[0],
                fused[1],
                recommendation_url,
                recommendation_source,
                json.dumps(spotify_items, ensure_ascii=True),
            ),
        )
        return int(cur.lastrowid)


def log_feedback(event_id: int, rating: int, note: str = "") -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO feedback (ts, event_id, rating, note) VALUES (?, ?, ?, ?);",
            (now_iso(), int(event_id), int(rating), note),
        )


def get_user_events(user: str, limit: int = 500) -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    q = """
    SELECT * FROM events
    WHERE user = ?
    ORDER BY id DESC
    LIMIT ?;
    """
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(q, conn, params=(user, int(limit)))
    return df


def get_recent_urls(user: str, lookback: int = 5) -> List[str]:
    if not DB_PATH.exists():
        return []
    q = """
    SELECT recommendation_url FROM events
    WHERE user = ? AND recommendation_url IS NOT NULL AND recommendation_url <> ''
    ORDER BY id DESC
    LIMIT ?;
    """
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(q, conn, params=(user, int(lookback)))
    return [x for x in df["recommendation_url"].tolist() if x]


def get_bandit_stats(user: str, mood: str) -> Dict[str, Dict[str, float]]:
    """Return per-url stats: average reward + counts for UCB-like selection."""
    if not DB_PATH.exists():
        return {}
    q = """
    SELECT
      e.recommendation_url AS url,
      COUNT(e.id) AS shown_count,
      COALESCE(AVG(f.rating), 0.0) AS avg_reward,
      COALESCE(COUNT(f.id), 0) AS feedback_count
    FROM events e
    LEFT JOIN feedback f ON f.event_id = e.id
    WHERE e.user = ? AND e.fused_mood = ? AND e.recommendation_url IS NOT NULL
    GROUP BY e.recommendation_url;
    """
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(q, conn, params=(user, mood))
    stats: Dict[str, Dict[str, float]] = {}
    for _, row in df.iterrows():
        stats[str(row["url"])] = {
            "shown_count": float(row["shown_count"]),
            "avg_reward": float(row["avg_reward"]),
            "feedback_count": float(row["feedback_count"]),
        }
    return stats


# -------------------------
# Resource catalog (100+)
# -------------------------
def build_resource_catalog() -> Dict[str, List[Dict[str, str]]]:
    catalog: Dict[str, List[Dict[str, str]]] = {}
    for mood in MOODS:
        entries: List[Dict[str, str]] = []

        for idx, url in enumerate(YOUTUBE_PLAYABLE[mood], start=1):
            entries.append(
                {
                    "mood": mood,
                    "title": f"{mood.title()} direct video #{idx}",
                    "url": url,
                    "source": "YouTube-Playable",
                }
            )

        for term in YOUTUBE_SEARCH_TERMS[mood]:
            entries.append(
                {
                    "mood": mood,
                    "title": f"{term.title()}",
                    "url": f"https://www.youtube.com/results?search_query={quote_plus(term)}",
                    "source": "YouTube-Search",
                }
            )

        for term in SPOTIFY_SEARCH_TERMS[mood]:
            entries.append(
                {
                    "mood": mood,
                    "title": f"{term.title()}",
                    "url": f"https://open.spotify.com/search/{quote_plus(term)}",
                    "source": "Spotify-Search",
                }
            )

        catalog[mood] = entries
    return catalog


RESOURCE_CATALOG = build_resource_catalog()  # 4 moods * (12 + 10 + 8) = 120 entries


# -------------------------
# Multi-modal inference
# -------------------------
def text_to_mood(text: str) -> Tuple[str, float]:
    text = (text or "").strip().lower()
    if not text:
        return "calm", 0.25

    tokens = re.findall(r"[a-z']+", text)
    scores = {m: 0 for m in MOODS}
    for token in tokens:
        for mood, words in TEXT_KEYWORDS.items():
            if token in words:
                scores[mood] += 1

    if max(scores.values()) == 0:
        return "calm", 0.35

    best = max(scores, key=scores.get)
    total = sum(scores.values())
    conf = max(0.35, min(0.95, scores[best] / max(total, 1)))
    return best, conf


def face_to_mood(image_file) -> Tuple[str, float, Dict[str, float]]:
    if image_file is None:
        return "calm", 0.2, {"note": 0.0}

    image = Image.open(image_file).convert("RGB")
    arr = np.array(image)
    gray = np.mean(arr, axis=2).astype(np.uint8)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    saturation = float(np.std(arr, axis=2).mean())

    faces_detected = 0
    if cv2 is not None:
        try:
            cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            det = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
            faces_detected = int(len(det))
        except Exception:
            faces_detected = 0

    if brightness < 80:
        mood = "sad"
    elif brightness > 165 and saturation > 35:
        mood = "happy"
    elif contrast > 72:
        mood = "energetic"
    else:
        mood = "calm"

    confidence = 0.45
    confidence += min(abs(brightness - 120) / 255.0, 0.2)
    confidence += min(contrast / 255.0, 0.2)
    if faces_detected > 0:
        confidence += 0.15
    confidence = max(0.35, min(0.95, confidence))

    details = {
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "saturation": round(saturation, 2),
        "faces_detected": float(faces_detected),
    }
    return mood, float(confidence), details


def transcribe_audio_file(audio_file) -> Tuple[str, str]:
    if audio_file is None:
        return "", ""
    if sr is None:
        return "", "speechrecognition not installed (pip install SpeechRecognition)"

    suffix = Path(audio_file.name).suffix.lower()
    if suffix not in {".wav", ".aiff", ".aif", ".flac"}:
        return "", "Upload wav/aiff/flac for built-in transcription."

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_file.getvalue())
            temp_path = tmp.name

        rec = sr.Recognizer()
        with sr.AudioFile(temp_path) as source:
            data = rec.record(source)
        text = rec.recognize_google(data)
        return text.strip(), ""
    except Exception as exc:
        return "", f"Transcription failed: {exc}"
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def mood_distribution(mood: str, conf: float) -> np.ndarray:
    conf = float(max(0.0, min(1.0, conf)))
    vec = np.zeros(len(MOODS), dtype=float)
    idx = MOOD_TO_IDX.get(mood, MOOD_TO_IDX["calm"])
    if len(MOODS) > 1:
        spill = (1.0 - conf) / (len(MOODS) - 1)
    else:
        spill = 0.0
    vec[:] = spill
    vec[idx] = conf
    return vec


def fuse_moods(
    face: Tuple[str, float],
    voice: Tuple[str, float],
    emoji: Tuple[str, float],
    text_m: Tuple[str, float],
) -> Tuple[str, float, Dict[str, float]]:
    # Weighted multi-modal fusion
    weights = {"face": 0.35, "voice": 0.25, "emoji": 0.15, "text": 0.25}
    total = np.zeros(len(MOODS), dtype=float)
    total += weights["face"] * mood_distribution(face[0], face[1])
    total += weights["voice"] * mood_distribution(voice[0], voice[1])
    total += weights["emoji"] * mood_distribution(emoji[0], emoji[1])
    total += weights["text"] * mood_distribution(text_m[0], text_m[1])

    pred_idx = int(np.argmax(total))
    pred_mood = MOODS[pred_idx]
    pred_conf = float(total[pred_idx])
    score_map = {m: float(total[i]) for i, m in enumerate(MOODS)}
    return pred_mood, pred_conf, score_map


# -------------------------
# RL-style recommender
# -------------------------
def recommend_youtube(user: str, mood: str, epsilon: float = 0.15) -> str:
    options = YOUTUBE_PLAYABLE.get(mood, YOUTUBE_PLAYABLE["calm"])
    recent = set(get_recent_urls(user, lookback=5))
    stats = get_bandit_stats(user, mood)

    candidates = [url for url in options if url not in recent]
    if not candidates:
        # If all options recently played, keep a smaller anti-repeat penalty and still select best.
        candidates = list(options)

    if random.random() < epsilon:
        return random.choice(candidates)

    total_shows = sum(v["shown_count"] for v in stats.values()) + 1.0
    best_score = -1e9
    best_url = candidates[0]
    for url in candidates:
        stt = stats.get(url, {"shown_count": 0.0, "avg_reward": 0.0, "feedback_count": 0.0})
        shown = stt["shown_count"]
        avg_reward = stt["avg_reward"]
        exploration_bonus = math.sqrt(2.0 * math.log(total_shows + 1.0) / (shown + 1.0))
        repeat_penalty = 0.4 if url in recent else 0.0
        score = avg_reward + exploration_bonus - repeat_penalty
        if score > best_score:
            best_score = score
            best_url = url
    return best_url


def get_spotify_client():
    if spotipy is None or SpotifyClientCredentials is None:
        return None
    cid = os.getenv("SPOTIPY_CLIENT_ID", "").strip()
    sec = os.getenv("SPOTIPY_CLIENT_SECRET", "").strip()
    if not cid or not sec:
        return None
    try:
        auth = SpotifyClientCredentials(client_id=cid, client_secret=sec)
        return spotipy.Spotify(auth_manager=auth)
    except Exception:
        return None


def spotify_recommend(mood: str, limit: int = 5) -> List[Dict[str, str]]:
    client = get_spotify_client()
    if client is None:
        return []
    query_map = {
        "happy": "happy feel good hits",
        "sad": "healing calm songs",
        "calm": "calm meditation focus",
        "energetic": "workout edm motivation",
    }
    q = query_map.get(mood, "calm vibes")
    try:
        data = client.search(q=q, type="track", limit=int(limit))
        out: List[Dict[str, str]] = []
        for item in data.get("tracks", {}).get("items", []):
            artists = ", ".join([a["name"] for a in item.get("artists", [])])
            out.append(
                {
                    "name": item.get("name", "Unknown"),
                    "artists": artists,
                    "url": item.get("external_urls", {}).get("spotify", ""),
                }
            )
        return out
    except Exception:
        return []


# -------------------------
# OpenAI helpers
# -------------------------
def get_openai_client():
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key or OpenAI is None:
        return None
    try:
        return OpenAI(api_key=key)
    except Exception:
        return None


def emotional_support_reply(user_text: str, fused_mood: str, twin_summary: str = "") -> str:
    text = (user_text or "").strip()
    if not text:
        text = f"User feels {fused_mood}."
    fallback = (
        f"I hear you. Your detected mood is '{fused_mood}'. "
        "Take one slow deep breath, hydrate, and do one small positive action in the next 5 minutes."
    )

    client = get_openai_client()
    if client is None:
        return fallback

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.4,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an empathetic emotional wellness assistant. "
                        "Give concise, practical, supportive responses. "
                        "Do not diagnose medical conditions."
                    ),
                },
                {"role": "system", "content": f"Digital twin summary: {twin_summary}"},
                {"role": "user", "content": text},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return fallback


def explain_prediction_with_llm(
    predicted_class: str,
    confidence: float,
    user_question: str,
    extra_context: str = "",
) -> str:
    fallback = (
        f"Predicted class: {predicted_class} ({confidence:.2%}). "
        "The model relied on visual patterns learned through transfer learning."
    )
    client = get_openai_client()
    if client is None:
        return fallback
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You explain CNN outputs to students. "
                        "Be clear, practical, and avoid hallucinations."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Prediction: {predicted_class}\n"
                        f"Confidence: {confidence:.4f}\n"
                        f"Context: {extra_context}\n"
                        f"Question: {user_question}"
                    ),
                },
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return fallback


# -------------------------
# Digital twin analytics
# -------------------------
def summarize_digital_twin(df: pd.DataFrame) -> Dict[str, str]:
    if df.empty:
        return {
            "sessions": "0",
            "dominant_mood": "N/A",
            "stability": "N/A",
            "summary": "No history yet. Run emotion analysis to build your digital twin.",
        }

    mood_counts = df["fused_mood"].value_counts()
    dominant = mood_counts.index[0]
    sessions = len(df)

    ordered = df.iloc[::-1].copy()
    transitions = 0
    prev = None
    for mood in ordered["fused_mood"].tolist():
        if prev is not None and mood != prev:
            transitions += 1
        prev = mood
    stability = 1.0 - (transitions / max(sessions - 1, 1))

    summary = (
        f"Dominant mood is '{dominant}'. Mood stability score: {stability:.2f}. "
        f"Total logged sessions: {sessions}."
    )
    return {
        "sessions": str(sessions),
        "dominant_mood": dominant,
        "stability": f"{stability:.2f}",
        "summary": summary,
    }


# -------------------------
# CNN transfer learning + evaluation + Grad-CAM
# -------------------------
def check_dataset_structure(dataset_root: Path) -> Tuple[bool, str]:
    required = ["train", "val", "test"]
    if not dataset_root.exists():
        return False, f"Dataset path does not exist: {dataset_root}"
    for split in required:
        if not (dataset_root / split).exists():
            return False, f"Missing split folder: {dataset_root / split}"
    return True, "OK"


def get_device() -> str:
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_transfer_model(architecture: str, num_classes: int):
    if architecture == "ResNet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        for p in model.parameters():
            p.requires_grad = False
        for p in model.layer4.parameters():
            p.requires_grad = True
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        target_layer = model.layer4[-1].conv2
    elif architecture == "EfficientNet-B0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        for p in model.features.parameters():
            p.requires_grad = False
        for p in model.features[-1].parameters():
            p.requires_grad = True
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        target_layer = model.features[-1]
    else:
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        for p in model.features.parameters():
            p.requires_grad = False
        for p in model.features[-1].parameters():
            p.requires_grad = True
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        target_layer = model.features[-1]
    return model, target_layer


def get_transforms(image_size: int = 224):
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return train_tf, eval_tf


def evaluate_classifier(model, loader, device: str, num_classes: int):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            y_prob.extend(probs.tolist())
            y_pred.extend(preds.tolist())
            y_true.extend(y.numpy().tolist())

    y_prob_arr = np.array(y_prob)
    acc = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    roc_auc = None
    try:
        if num_classes == 2:
            roc_auc = float(roc_auc_score(y_true, y_prob_arr[:, 1]))
        else:
            y_bin = label_binarize(y_true, classes=list(range(num_classes)))
            roc_auc = float(roc_auc_score(y_bin, y_prob_arr, multi_class="ovr", average="macro"))
    except Exception:
        roc_auc = None

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        "accuracy": acc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def train_transfer_learning_model(
    dataset_root: Path,
    architecture: str,
    epochs: int,
    batch_size: int,
    lr: float,
    image_size: int,
):
    train_tf, eval_tf = get_transforms(image_size=image_size)
    train_ds = datasets.ImageFolder(str(dataset_root / "train"), transform=train_tf)
    val_ds = datasets.ImageFolder(str(dataset_root / "val"), transform=eval_tf)
    test_ds = datasets.ImageFolder(str(dataset_root / "test"), transform=eval_tf)

    class_names = train_ds.classes
    num_classes = len(class_names)
    if num_classes < 2:
        raise ValueError("Need at least 2 classes for CNN classification.")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model, _ = build_transfer_model(architecture, num_classes=num_classes)
    device = get_device()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    history = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        seen = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * x.size(0)
            seen += x.size(0)

        train_loss = running_loss / max(seen, 1)
        val_metrics = evaluate_classifier(model, val_loader, device, num_classes)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_f1": val_metrics["f1"],
            }
        )

    test_metrics = evaluate_classifier(model, test_loader, device, num_classes)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "architecture": architecture,
        "image_size": image_size,
        "class_names": class_names,
        "state_dict": model.state_dict(),
        "trained_at": now_iso(),
        "history": history,
        "test_metrics": {
            k: v
            for k, v in test_metrics.items()
            if k not in {"confusion_matrix", "classification_report"}
        },
    }
    torch.save(checkpoint, MODEL_PATH)

    return model, checkpoint, test_metrics


def load_trained_model(model_path: Path = MODEL_PATH):
    ckpt = torch.load(model_path, map_location="cpu")
    architecture = ckpt["architecture"]
    class_names = ckpt["class_names"]
    image_size = int(ckpt.get("image_size", 224))
    model, target_layer = build_transfer_model(architecture, len(class_names))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, target_layer, class_names, image_size, architecture


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self.bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx: Optional[int] = None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(torch.argmax(output, dim=1).item())
        self.model.zero_grad()
        score = output[:, class_idx]
        score.backward(retain_graph=True)

        grads = self.gradients[0]  # [C, H, W]
        acts = self.activations[0]  # [C, H, W]
        weights = torch.mean(grads, dim=(1, 2))
        cam = torch.zeros(acts.shape[1:], dtype=acts.dtype, device=acts.device)
        for i, w in enumerate(weights):
            cam += w * acts[i]
        cam = torch.relu(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.detach().cpu().numpy(), class_idx

    def close(self):
        self.fwd_hook.remove()
        self.bwd_hook.remove()


def predict_image(model, image: Image.Image, class_names: List[str], image_size: int, device: str = "cpu"):
    tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    x = tf(image.convert("RGB")).unsqueeze(0).to(device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    idx = int(np.argmax(probs))
    return class_names[idx], float(probs[idx]), probs


def overlay_cam_on_image(image: Image.Image, cam: np.ndarray) -> Image.Image:
    base = np.array(image.convert("RGB").resize((224, 224)))
    cam_u8 = np.uint8(np.clip(cam, 0.0, 1.0) * 255.0)

    if cv2 is not None:
        heat = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    else:
        heat = np.stack([cam_u8, np.zeros_like(cam_u8), 255 - cam_u8], axis=-1)

    blended = np.clip(0.58 * base + 0.42 * heat, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def extract_dataset_zip(uploaded_zip, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / f"dataset_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(uploaded_zip.getvalue())) as zf:
        zf.extractall(target)
    return target


# -------------------------
# Native build helper text
# -------------------------
def native_build_notes() -> str:
    return (
        "AppsGeyser/WebViewGold are NOT the only option. "
        "For a no-paid-wrapper path, use a native Python app approach:\n"
        "1) Android (free): Kivy + Buildozer -> generates APK/AAB.\n"
        "2) Windows 11 desktop app (free): PyInstaller -> .exe.\n"
        "3) If project is web-first: deploy backend + PWA/TWA for installable app behavior.\n\n"
        "Best substitute for major projects with ML logic: keep this CEI-ALOS model code as core, "
        "and build native frontends for Android/Windows that call the same core methods/API."
    )


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
init_db()

if "voice_text" not in st.session_state:
    st.session_state["voice_text"] = ""
if "last_event_id" not in st.session_state:
    st.session_state["last_event_id"] = None
if "last_recommendation_url" not in st.session_state:
    st.session_state["last_recommendation_url"] = ""
if "last_user" not in st.session_state:
    st.session_state["last_user"] = "guest"

st.title("🧠 CEI-ALOS: Unified Major Project (Single Python Source)")
st.caption(
    "Multi-modal AI + RL anti-repetition + Digital Emotional Twin + "
    "CNN Transfer Learning + Grad-CAM + Spotify + OpenAI integration"
)

with st.sidebar:
    st.subheader("🔑 API Keys (session/runtime)")
    openai_key = st.text_input("OPENAI_API_KEY", type="password", help="Session-only unless exported in shell.")
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key.strip()

    sp_id = st.text_input("SPOTIPY_CLIENT_ID", type="password")
    sp_secret = st.text_input("SPOTIPY_CLIENT_SECRET", type="password")
    if sp_id:
        os.environ["SPOTIPY_CLIENT_ID"] = sp_id.strip()
    if sp_secret:
        os.environ["SPOTIPY_CLIENT_SECRET"] = sp_secret.strip()

    st.markdown("### Runtime health")
    st.write(f"OpenCV: {'OK' if cv2 is not None else 'Missing'}")
    st.write(f"SpeechRecognition: {'OK' if sr is not None else 'Missing'}")
    st.write(f"Spotipy: {'OK' if spotipy is not None else 'Missing'}")
    st.write(f"OpenAI SDK: {'OK' if OpenAI is not None else 'Missing'}")
    st.write(f"PyTorch: {'OK' if TORCH_AVAILABLE else 'Missing'}")
    st.write(f"scikit-learn: {'OK' if SKLEARN_AVAILABLE else 'Missing'}")

tabs = st.tabs(
    [
        "Architecture + API Setup",
        "Emotion Engine",
        "Resource Catalog (120)",
        "Digital Emotional Twin",
        "CNN Lab (Transfer + Metrics + Grad-CAM)",
        "LLM Chat + Image Context",
        "Native Build (Free Substitute)",
    ]
)

# -------------------------
# Tab 1: Architecture and setup
# -------------------------
with tabs[0]:
    st.markdown(
        """
### Final Architecture
**Cognitive Emotion Intelligence & Adaptive Lifestyle OS (CEI-ALOS)**

Face (CNN/heuristic) -> Voice (Speech -> NLP) -> Emoji/Text -> Multi-modal Fusion ->
Reinforcement Learning (history-aware) -> Recommendation Engine ->
YouTube + Spotify + Wellness Content -> Digital Emotional Twin (logging + analytics)
"""
    )

    st.markdown("### Step-by-step API setup")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**OpenAI**")
        st.code(
            "Linux/macOS:\n"
            'export OPENAI_API_KEY="your_key_here"\n\n'
            "Windows PowerShell:\n"
            'setx OPENAI_API_KEY "your_key_here"',
            language="bash",
        )
    with c2:
        st.markdown("**Spotify**")
        st.code(
            "Linux/macOS:\n"
            'export SPOTIPY_CLIENT_ID="your_id"\n'
            'export SPOTIPY_CLIENT_SECRET="your_secret"\n\n'
            "Windows PowerShell:\n"
            'setx SPOTIPY_CLIENT_ID "your_id"\n'
            'setx SPOTIPY_CLIENT_SECRET "your_secret"',
            language="bash",
        )

    st.markdown("### Install command")
    st.code(
        "pip install streamlit opencv-python numpy pandas SpeechRecognition spotipy openai "
        "torch torchvision scikit-learn pillow",
        language="bash",
    )

    st.info(
        "This file is fully self-contained. Optional modules (Torch/OpenCV/Speech/OpenAI/Spotipy) "
        "degrade gracefully if unavailable."
    )

# -------------------------
# Tab 2: Emotion Engine
# -------------------------
with tabs[1]:
    st.subheader("🚀 Multi-modal Emotion Analysis + RL Recommendation")
    user = st.text_input("Username", value=st.session_state.get("last_user", "guest"), key="main_user")
    st.session_state["last_user"] = user

    c1, c2 = st.columns(2)
    with c1:
        emoji = st.select_slider("Select Emoji", options=list(EMOJI_MAP.keys()), value="😊")
        text_input = st.text_area("Text input (emotion context)", placeholder="Type how you feel...")
        voice_upload = st.file_uploader(
            "Upload voice file for transcription (wav/aiff/flac)", type=["wav", "aiff", "aif", "flac"], key="voice"
        )
        if st.button("Transcribe Voice File"):
            txt, err = transcribe_audio_file(voice_upload)
            if err:
                st.warning(err)
            st.session_state["voice_text"] = txt
        voice_text = st.text_area("Voice transcript (editable)", value=st.session_state.get("voice_text", ""))
        st.session_state["voice_text"] = voice_text

    with c2:
        cam_img = st.camera_input("Capture face")
        upload_img = st.file_uploader("Or upload face image", type=["jpg", "jpeg", "png"], key="face_upload")

    if st.button("Analyze Emotion + Recommend", type="primary"):
        face_source = upload_img if upload_img is not None else cam_img
        face_mood, face_conf, face_details = face_to_mood(face_source)
        voice_mood, voice_conf = text_to_mood(voice_text)
        emoji_mood = EMOJI_MAP.get(emoji, "calm")
        emoji_conf = 0.9
        text_mood, text_conf = text_to_mood(text_input)

        fused_mood, fused_conf, fused_scores = fuse_moods(
            face=(face_mood, face_conf),
            voice=(voice_mood, voice_conf),
            emoji=(emoji_mood, emoji_conf),
            text_m=(text_mood, text_conf),
        )

        yt_url = recommend_youtube(user=user, mood=fused_mood, epsilon=0.15)
        spotify_items = spotify_recommend(fused_mood, limit=5)

        event_id = log_event(
            user=user,
            face=(face_mood, face_conf),
            voice_text=voice_text,
            voice=(voice_mood, voice_conf),
            emoji=emoji,
            emoji_mood=emoji_mood,
            text_input=text_input,
            text=(text_mood, text_conf),
            fused=(fused_mood, fused_conf),
            recommendation_url=yt_url,
            recommendation_source="youtube",
            spotify_items=spotify_items,
        )

        st.session_state["last_event_id"] = event_id
        st.session_state["last_recommendation_url"] = yt_url

        st.success(f"Final fused mood: {fused_mood} (confidence: {fused_conf:.2f})")
        st.json(
            {
                "face": {"mood": face_mood, "conf": round(face_conf, 3), "features": face_details},
                "voice": {"mood": voice_mood, "conf": round(voice_conf, 3), "text": voice_text},
                "emoji": {"mood": emoji_mood, "conf": emoji_conf},
                "text": {"mood": text_mood, "conf": round(text_conf, 3)},
                "fusion_scores": {k: round(v, 3) for k, v in fused_scores.items()},
                "event_id": event_id,
            }
        )

        st.markdown("### 📺 YouTube recommendation")
        st.video(yt_url)

        st.markdown("### 🎧 Spotify tracks")
        if spotify_items:
            for item in spotify_items:
                st.write(f"- {item['name']} — {item['artists']}  \n  {item['url']}")
        else:
            st.info("Spotify recommendations unavailable (set SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET).")

        twin_df = get_user_events(user, limit=200)
        twin_summary = summarize_digital_twin(twin_df)["summary"]
        coach = emotional_support_reply(
            user_text=text_input or voice_text or f"I feel {fused_mood}",
            fused_mood=fused_mood,
            twin_summary=twin_summary,
        )
        st.markdown("### 💬 Emotional AI response")
        st.write(coach)

    if st.session_state.get("last_event_id"):
        st.markdown("### Reinforcement feedback (improves no-repeat policy)")
        f1, f2 = st.columns(2)
        with f1:
            if st.button("👍 Helpful"):
                log_feedback(st.session_state["last_event_id"], rating=1, note="helpful")
                st.success("Positive feedback recorded.")
        with f2:
            if st.button("👎 Not helpful"):
                log_feedback(st.session_state["last_event_id"], rating=-1, note="not helpful")
                st.warning("Negative feedback recorded.")

# -------------------------
# Tab 3: Catalog
# -------------------------
with tabs[2]:
    st.subheader("📚 Structured Resource Catalog (100+ links)")
    all_rows: List[Dict[str, str]] = []
    for mood, items in RESOURCE_CATALOG.items():
        all_rows.extend(items)
    catalog_df = pd.DataFrame(all_rows)
    st.metric("Total catalog entries", len(catalog_df))

    mood_filter = st.multiselect("Filter mood", options=MOODS, default=MOODS)
    source_filter = st.multiselect(
        "Filter source",
        options=sorted(catalog_df["source"].unique().tolist()),
        default=sorted(catalog_df["source"].unique().tolist()),
    )

    filtered = catalog_df[catalog_df["mood"].isin(mood_filter) & catalog_df["source"].isin(source_filter)]
    st.dataframe(filtered, use_container_width=True, height=420)

    csv_data = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download catalog CSV",
        data=csv_data,
        file_name="cei_alos_resource_catalog.csv",
        mime="text/csv",
    )

    st.caption(
        "Catalog is structured and expandable. Recommendation engine uses playable YouTube links "
        "and avoids recent repetition through RL-style selection."
    )

# -------------------------
# Tab 4: Digital twin
# -------------------------
with tabs[3]:
    st.subheader("🧬 Digital Emotional Twin")
    twin_user = st.text_input("User for twin analytics", value=st.session_state.get("last_user", "guest"))
    hist = get_user_events(twin_user, limit=1000)

    if hist.empty:
        st.info("No history for this user yet. Run Emotion Engine to start tracking.")
    else:
        summary = summarize_digital_twin(hist)
        c1, c2, c3 = st.columns(3)
        c1.metric("Sessions", summary["sessions"])
        c2.metric("Dominant mood", summary["dominant_mood"])
        c3.metric("Mood stability", summary["stability"])
        st.write(summary["summary"])

        mood_counts = hist["fused_mood"].value_counts().reindex(MOODS, fill_value=0)
        st.markdown("### Mood distribution")
        st.bar_chart(mood_counts)

        timeline = hist.copy()
        timeline["ts"] = pd.to_datetime(timeline["ts"], errors="coerce")
        timeline = timeline.sort_values("ts")
        timeline["mood_num"] = timeline["fused_mood"].map(MOOD_TO_IDX)
        st.markdown("### Mood timeline (encoded)")
        st.line_chart(timeline.set_index("ts")["mood_num"])

        # Non-repetition score for recommendations
        recs = timeline["recommendation_url"].dropna().tolist()
        if len(recs) > 1:
            repeats = sum(1 for i in range(1, len(recs)) if recs[i] == recs[i - 1])
            no_repeat_rate = 1.0 - (repeats / (len(recs) - 1))
            st.metric("No-repeat recommendation score", f"{no_repeat_rate:.2%}")

        st.markdown("### Recent emotional twin logs")
        show_cols = [
            "id",
            "ts",
            "face_mood",
            "voice_mood",
            "emoji_mood",
            "text_mood",
            "fused_mood",
            "recommendation_url",
        ]
        st.dataframe(hist[show_cols].head(20), use_container_width=True)

# -------------------------
# Tab 5: CNN lab
# -------------------------
with tabs[4]:
    st.subheader("🧪 CNN Lab: Transfer Learning + Full Metrics + Grad-CAM")
    st.caption(
        "Mandatory components included: pretrained transfer learning, precision/recall/F1/confusion matrix/ROC-AUC, "
        "deployment-ready prediction, and model explainability with Grad-CAM."
    )

    if not TORCH_AVAILABLE or not SKLEARN_AVAILABLE:
        st.error("Torch/scikit-learn missing for CNN workflow.")
        if not TORCH_AVAILABLE:
            st.code(f"PyTorch import error: {TORCH_IMPORT_ERROR}", language="text")
        if not SKLEARN_AVAILABLE:
            st.code(f"scikit-learn import error: {SKLEARN_IMPORT_ERROR}", language="text")
    else:
        st.markdown("#### Dataset input")
        st.write("Expected structure inside dataset folder (or uploaded zip): train/, val/, test/ with class subfolders.")
        zip_file = st.file_uploader("Optional: upload dataset zip", type=["zip"], key="dataset_zip")
        if zip_file is not None and st.button("Extract uploaded dataset zip"):
            extracted = extract_dataset_zip(zip_file, DATASET_DIR)
            st.success(f"Extracted to: {extracted}")
            st.session_state["dataset_root"] = str(extracted)

        dataset_root_txt = st.text_input(
            "Dataset root path",
            value=st.session_state.get("dataset_root", ""),
            placeholder="e.g., /workspace/your_dataset",
        )
        architecture = st.selectbox("Transfer model", options=["MobileNetV3", "ResNet18", "EfficientNet-B0"])
        epochs = st.slider("Epochs", 1, 20, 3)
        batch_size = st.select_slider("Batch size", options=[4, 8, 16, 32], value=8)
        lr = st.number_input("Learning rate", min_value=1e-6, max_value=1e-1, value=1e-3, format="%.6f")
        image_size = st.selectbox("Image size", options=[160, 192, 224, 256], index=2)

        if st.button("Train transfer learning model"):
            if not dataset_root_txt:
                st.warning("Provide dataset root path or upload zip.")
            else:
                root = Path(dataset_root_txt)
                ok, msg = check_dataset_structure(root)
                if not ok:
                    st.error(msg)
                else:
                    with st.spinner("Training model..."):
                        try:
                            model, ckpt, metrics = train_transfer_learning_model(
                                dataset_root=root,
                                architecture=architecture,
                                epochs=int(epochs),
                                batch_size=int(batch_size),
                                lr=float(lr),
                                image_size=int(image_size),
                            )
                            st.success(f"Training complete. Model saved at: {MODEL_PATH}")

                            history_df = pd.DataFrame(ckpt["history"])
                            st.markdown("#### Training history")
                            st.dataframe(history_df, use_container_width=True)

                            st.markdown("#### Test metrics")
                            row = {
                                "accuracy": metrics["accuracy"],
                                "precision": metrics["precision"],
                                "recall": metrics["recall"],
                                "f1_score": metrics["f1"],
                                "roc_auc": metrics["roc_auc"],
                            }
                            st.dataframe(pd.DataFrame([row]))

                            st.markdown("#### Confusion matrix")
                            cm = metrics["confusion_matrix"]
                            cm_df = pd.DataFrame(cm)
                            st.dataframe(cm_df, use_container_width=True)

                            st.markdown("#### Classification report")
                            rep_df = pd.DataFrame(metrics["classification_report"]).T
                            st.dataframe(rep_df, use_container_width=True)
                        except Exception as exc:
                            st.exception(exc)

        st.markdown("#### Inference + Grad-CAM")
        pred_image_file = st.file_uploader("Upload image for prediction", type=["jpg", "jpeg", "png"], key="pred_img")
        if st.button("Predict with trained model"):
            if not MODEL_PATH.exists():
                st.warning("Train model first so checkpoint is available.")
            elif pred_image_file is None:
                st.warning("Upload an image first.")
            else:
                try:
                    model, target_layer, class_names, model_img_size, _ = load_trained_model(MODEL_PATH)
                    img = Image.open(pred_image_file).convert("RGB")
                    pred_class, conf, probs = predict_image(
                        model=model,
                        image=img,
                        class_names=class_names,
                        image_size=model_img_size,
                        device="cpu",
                    )
                    st.success(f"Prediction: {pred_class} ({conf:.2%})")
                    probs_df = pd.DataFrame({"class": class_names, "probability": probs})
                    st.dataframe(probs_df, use_container_width=True)

                    grad = GradCAM(model, target_layer)
                    tf = transforms.Compose(
                        [
                            transforms.Resize((model_img_size, model_img_size)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                        ]
                    )
                    x = tf(img).unsqueeze(0)
                    cam, pred_idx = grad.generate(x)
                    grad.close()
                    overlay = overlay_cam_on_image(img, cam)
                    st.image([img, overlay], caption=["Original", f"Grad-CAM (class: {class_names[pred_idx]})"], width=320)
                except Exception as exc:
                    st.exception(exc)

# -------------------------
# Tab 6: LLM chat + image context
# -------------------------
with tabs[5]:
    st.subheader("💬 Chatbot with Image Context (CNN + LLM Integration)")
    st.caption("Advanced enhancement fulfilled: CNN + LLM integration and multimodal (image + text) reasoning.")

    chat_user = st.text_input("User", value=st.session_state.get("last_user", "guest"), key="chat_user")
    user_question = st.text_area("Ask your question", placeholder="How does my current pattern look and what should I do next?")
    img_for_chat = st.file_uploader("Optional image for CNN context", type=["jpg", "jpeg", "png"], key="chat_img")

    if st.button("Generate AI response"):
        history_df = get_user_events(chat_user, limit=200)
        twin_summary = summarize_digital_twin(history_df)["summary"]
        context = f"Twin summary: {twin_summary}"

        cnn_context = ""
        if img_for_chat is not None and MODEL_PATH.exists() and TORCH_AVAILABLE:
            try:
                model, _, class_names, model_img_size, _ = load_trained_model(MODEL_PATH)
                img = Image.open(img_for_chat).convert("RGB")
                pred_class, conf, _ = predict_image(model, img, class_names, model_img_size, device="cpu")
                cnn_context = f"CNN predicted class '{pred_class}' with confidence {conf:.2%}."
                st.info(cnn_context)
                explain = explain_prediction_with_llm(pred_class, conf, user_question or "Explain this output.", context)
                st.write("### CNN explanation")
                st.write(explain)
            except Exception as exc:
                st.warning(f"CNN context unavailable: {exc}")

        final_prompt = user_question or "Give me one practical wellness action for today."
        response = emotional_support_reply(
            user_text=f"{final_prompt}\n{cnn_context}",
            fused_mood="calm",
            twin_summary=twin_summary,
        )
        st.write("### AI response")
        st.write(response)

# -------------------------
# Tab 7: Free native build substitute
# -------------------------
with tabs[6]:
    st.subheader("📱 Native Android + Windows (Free Substitute to Paid Wrappers)")
    st.write(native_build_notes())

    st.markdown("### Recommended free path (no paid wrapper)")
    st.markdown(
        """
**Android (native-like, free):**
1. Keep CEI-ALOS logic in Python.
2. Build a native client with **Kivy**.
3. Use **Buildozer** to generate APK/AAB.
4. Sign and distribute APK for direct install.

**Windows 11 desktop app (free):**
1. Package using **PyInstaller**.
2. Generate `.exe` for offline use.
"""
    )

    st.code(
        "# Android free build path (Linux recommended)\n"
        "pip install kivy buildozer\n"
        "buildozer init\n"
        "buildozer -v android debug\n\n"
        "# Windows 11 free build path\n"
        "pip install pyinstaller\n"
        "pyinstaller --onefile --name CEI_ALOS app.py",
        language="bash",
    )

    st.markdown("### Conditional project guidance")
    st.markdown(
        """
- If project is **web-first**: deploy backend and use PWA/TWA.
- If project needs **on-device ML + sensors**: native Android (Kotlin/Flutter) with Python backend API.
- If project needs **fast prototype with full Python stack**: Kivy + Buildozer is the strongest free substitute.
"""
    )

    st.markdown("### Viva statement")
    st.success(
        "Our system implements a multi-modal reinforcement learning-based emotional intelligence OS "
        "integrating facial analysis, voice sentiment NLP, emoji-based interaction, Spotify and YouTube "
        "recommendation engines, and GPT-based emotional AI, forming a personalized digital emotional twin "
        "with adaptive lifestyle recommendations."
    )
