from __future__ import annotations

import argparse
import csv
import html
import io
import importlib.util
import json
import logging
import os
import random
import re
import textwrap
import urllib.error
import urllib.parse
import urllib.request
import warnings
import wave
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets as tv_datasets
from torchvision import transforms
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s

os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "true")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*Accessing `__path__`.*")
logging.getLogger("transformers").setLevel(logging.ERROR)

try:
    import cv2
except Exception:
    cv2 = None

try:
    import nltk
    from nltk.corpus import wordnet as wn

    NLTK_AVAILABLE = True
except Exception:
    nltk = None
    wn = None
    NLTK_AVAILABLE = False

try:
    import pyttsx3
except Exception:
    pyttsx3 = None

try:
    import speech_recognition as sr
except Exception:
    sr = None

try:
    import streamlit as st
except Exception:
    st = None

TRANSFORMERS_AVAILABLE = importlib.util.find_spec("transformers") is not None


APP_TITLE = "Emotionally Intelligent Animated Mascot Chatbot"
APP_SUBTITLE = (
    "Single-file PyTorch + Streamlit project with EfficientNetV2-S, Grad-CAM, "
    "free Hugging Face chat, and digital emotional twin logging."
)

BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
MODEL_DIR = BASE_DIR / "models"
DATASET_DIR = BASE_DIR / "dataset"
MODEL_PATH = MODEL_DIR / "emotion_efficientnet_v2_s.pth"
MODEL_META_PATH = MODEL_DIR / "emotion_efficientnet_v2_s.json"
TWIN_LOG_PATH = BASE_DIR / "cei_twin_log.csv"
RECOMMENDER_STATS_PATH = BASE_DIR / "recommender_stats.csv"
CATALOG_EXPORT_PATH = BASE_DIR / "resource_catalog.csv"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

HF_TOKEN = os.getenv("HF_TOKEN", "").strip() or os.getenv("HUGGINGFACEHUB_API_TOKEN", "").strip()
HF_CHAT_MODEL = os.getenv("HF_CHAT_MODEL", "google/flan-t5-small").strip() or "google/flan-t5-small"
TEXT_EMOTION_MODEL = (
    os.getenv("TEXT_EMOTION_MODEL", "j-hartmann/emotion-english-distilroberta-base").strip()
    or "j-hartmann/emotion-english-distilroberta-base"
)

MOOD_CHOICES = ["happy", "sad", "calm", "energetic"]
EMOJI_MAP = {
    "😊": "happy",
    "😄": "happy",
    "😍": "happy",
    "😢": "sad",
    "😭": "sad",
    "😔": "sad",
    "😌": "calm",
    "😴": "calm",
    "🙂": "calm",
    "😡": "energetic",
    "😤": "energetic",
    "🤩": "energetic",
}
EMOTION_TO_MOOD = {
    "happy": "happy",
    "joy": "happy",
    "love": "happy",
    "surprise": "energetic",
    "excited": "energetic",
    "anger": "energetic",
    "angry": "energetic",
    "fear": "sad",
    "sadness": "sad",
    "sad": "sad",
    "neutral": "calm",
    "calm": "calm",
    "relaxed": "calm",
}
DEFAULT_FACE_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

TONE_GUIDES = {
    "Therapist": "Warm, reflective, practical, and grounded support.",
    "Friendly": "Human, caring, positive, and conversational support.",
    "Motivational": "Energetic and action-focused support with discipline.",
}

FALLBACK_ACTIONS = {
    "happy": "Write one win, one gratitude, and one next meaningful goal.",
    "sad": "Take two minutes of slow breathing and send one message to a trusted person.",
    "calm": "Protect focus with one low-noise task and a 20-minute work sprint.",
    "energetic": "Channel your energy into one priority task or a short movement break.",
}

MOOD_QUESTION_BANK = {
    "happy": [
        "What triggered this positive shift today?",
        "How can you repeat one small action that helped?",
        "Who can you share this progress with?",
    ],
    "sad": [
        "What feels heaviest right now?",
        "What helped even a little last time?",
        "What is one tiny step that makes the next hour easier?",
    ],
    "calm": [
        "What is helping you stay balanced now?",
        "Which routine is protecting your focus today?",
        "What do you want to maintain before stress builds?",
    ],
    "energetic": [
        "Is this energy helping progress or creating overload?",
        "Where can this intensity be used productively?",
        "What boundary will keep this energy healthy?",
    ],
}

DIRECT_VIDEO_URLS = {
    "happy": [
        "https://www.youtube.com/watch?v=d-diB65scQU",
        "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
        "https://www.youtube.com/watch?v=OPf0YbXqDm0",
    ],
    "sad": [
        "https://www.youtube.com/watch?v=inpok4MKVLM",
        "https://www.youtube.com/watch?v=1vx8iUvfyCY",
        "https://www.youtube.com/watch?v=6p_yaNFSYao",
    ],
    "calm": [
        "https://www.youtube.com/watch?v=5qap5aO4i9A",
        "https://www.youtube.com/watch?v=v7AYKMP6rOE",
        "https://www.youtube.com/watch?v=lFcSrYw-ARY",
    ],
    "energetic": [
        "https://www.youtube.com/watch?v=HgzGwKwLmgM",
        "https://www.youtube.com/watch?v=ml6cT4AZdqI",
        "https://www.youtube.com/watch?v=fLexgOxsZu0",
    ],
}

YOUTUBE_SEARCH_QUERIES = {
    "happy": ["feel good playlist", "gratitude meditation", "upbeat study break songs"],
    "sad": ["comfort songs", "self compassion meditation", "gentle piano mood"],
    "calm": ["lofi focus session", "rain sounds for studying", "box breathing guide"],
    "energetic": ["power workout mix", "motivation speech short", "focus sprint music"],
}

SPOTIFY_SEARCH_QUERIES = {
    "happy": ["happy playlist", "feel good hits", "good vibes only"],
    "sad": ["comfort songs", "gentle piano", "self care songs"],
    "calm": ["lofi beats", "peaceful piano", "deep focus"],
    "energetic": ["workout hits", "high energy mix", "motivation songs"],
}


@dataclass(frozen=True)
class CatalogEntry:
    id: str
    mood: str
    title: str
    url: str
    source: str
    resource_type: str
    playable: bool
    tags: str
    offline_fallback: str


def require_streamlit() -> None:
    if st is None:
        raise RuntimeError("Streamlit is not installed. Install requirements and run streamlit run app.py.")


def slugify(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower())
    return text.strip("_") or "unknown"


def utc_now() -> str:
    return datetime.utcnow().isoformat()


def ensure_csv_file(path: Path, headers: list[str]) -> None:
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def append_csv_row(path: Path, fieldnames: list[str], row: dict[str, Any]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def ensure_runtime_files() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    ensure_csv_file(
        TWIN_LOG_PATH,
        [
            "User",
            "TimeUTC",
            "FaceLabel",
            "FaceMood",
            "VoiceMood",
            "TextMood",
            "EmojiMood",
            "FusedMood",
            "StateScore",
            "ToneMode",
            "RecommendedId",
            "RecommendedTitle",
            "RecommendedUrl",
            "RecommendedSource",
            "ChatQuery",
            "ChatReply",
            "Feedback",
        ],
    )
    ensure_csv_file(
        RECOMMENDER_STATS_PATH,
        ["ItemId", "Exposures", "Likes", "Skips", "LastShown", "LastFeedback"],
    )


@lru_cache(maxsize=1)
def build_resource_catalog() -> tuple[CatalogEntry, ...]:
    entries: list[CatalogEntry] = []
    for mood in MOOD_CHOICES:
        fallback = FALLBACK_ACTIONS[mood]
        for idx, url in enumerate(DIRECT_VIDEO_URLS[mood], start=1):
            entries.append(
                CatalogEntry(
                    id=f"{mood}_direct_{idx}",
                    mood=mood,
                    title=f"{mood.title()} direct video {idx}",
                    url=url,
                    source="YouTube",
                    resource_type="youtube_video",
                    playable=True,
                    tags=f"{mood},video,direct",
                    offline_fallback=fallback,
                )
            )
        for idx, query in enumerate(YOUTUBE_SEARCH_QUERIES[mood], start=1):
            entries.append(
                CatalogEntry(
                    id=f"{mood}_yt_{idx}",
                    mood=mood,
                    title=f"{mood.title()} YouTube search: {query.title()}",
                    url=f"https://www.youtube.com/results?search_query={urllib.parse.quote_plus(query)}",
                    source="YouTube",
                    resource_type="youtube_search",
                    playable=False,
                    tags=f"{mood},youtube,{slugify(query)}",
                    offline_fallback=fallback,
                )
            )
        for idx, query in enumerate(SPOTIFY_SEARCH_QUERIES[mood], start=1):
            entries.append(
                CatalogEntry(
                    id=f"{mood}_spotify_{idx}",
                    mood=mood,
                    title=f"{mood.title()} Spotify search: {query.title()}",
                    url=f"https://open.spotify.com/search/{urllib.parse.quote_plus(query)}",
                    source="Spotify",
                    resource_type="spotify_search",
                    playable=False,
                    tags=f"{mood},spotify,{slugify(query)}",
                    offline_fallback=fallback,
                )
            )
    return tuple(entries)


def catalog_rows() -> list[dict[str, Any]]:
    return [asdict(item) for item in build_resource_catalog()]


def export_resource_catalog(path: Path = CATALOG_EXPORT_PATH) -> None:
    rows = catalog_rows()
    if not rows:
        return
    write_csv_rows(path, list(rows[0].keys()), rows)


def get_user_history(user: str) -> list[str]:
    rows = read_csv_rows(TWIN_LOG_PATH)
    history: list[str] = []
    for row in rows:
        if str(row.get("User", "")) == str(user):
            item_id = str(row.get("RecommendedId", "")).strip()
            if item_id:
                history.append(item_id)
    return history


def recommender_rows() -> list[dict[str, str]]:
    ensure_runtime_files()
    return read_csv_rows(RECOMMENDER_STATS_PATH)


def upsert_recommender_feedback(item_id: str, feedback: str = "shown") -> None:
    rows = recommender_rows()
    fields = ["ItemId", "Exposures", "Likes", "Skips", "LastShown", "LastFeedback"]
    target = None
    for row in rows:
        if str(row.get("ItemId", "")) == str(item_id):
            target = row
            break
    if target is None:
        target = {
            "ItemId": item_id,
            "Exposures": "0",
            "Likes": "0",
            "Skips": "0",
            "LastShown": "",
            "LastFeedback": "",
        }
        rows.append(target)
    exposures = int(str(target.get("Exposures", "0") or "0"))
    likes = int(str(target.get("Likes", "0") or "0"))
    skips = int(str(target.get("Skips", "0") or "0"))
    if feedback == "shown":
        exposures += 1
        target["LastShown"] = utc_now()
    elif feedback == "liked":
        likes += 1
        target["LastFeedback"] = "liked"
    elif feedback == "skipped":
        skips += 1
        target["LastFeedback"] = "skipped"
    target["Exposures"] = str(exposures)
    target["Likes"] = str(likes)
    target["Skips"] = str(skips)
    write_csv_rows(RECOMMENDER_STATS_PATH, fields, rows)


def recommend_resource(user: str, mood: str, last_n: int = 5) -> dict[str, Any]:
    catalog = catalog_rows()
    candidates = [row for row in catalog if row["mood"] == mood] or catalog[:]
    recent = set(get_user_history(user)[-last_n:])
    stats_map = {row.get("ItemId", ""): row for row in recommender_rows()}
    scored: list[tuple[float, dict[str, Any]]] = []
    for item in candidates:
        stats = stats_map.get(item["id"], {})
        exposures = int(str(stats.get("Exposures", "0") or "0"))
        likes = int(str(stats.get("Likes", "0") or "0"))
        skips = int(str(stats.get("Skips", "0") or "0"))
        like_ratio = likes / max(exposures, 1)
        skip_ratio = skips / max(exposures, 1)
        novelty = 1.0 / (exposures + 1.0)
        recent_penalty = 0.50 if item["id"] in recent else 0.0
        playable_bonus = 0.12 if item.get("playable") else 0.0
        score = 0.45 * like_ratio + 0.35 * novelty + playable_bonus - 0.20 * skip_ratio - recent_penalty
        score += random.uniform(0.0, 0.05)
        scored.append((score, item))
    pool = [(score, item) for score, item in scored if item["id"] not in recent] or scored
    pool.sort(key=lambda pair: pair[0], reverse=True)
    top_pool = pool[: min(8, len(pool))]
    weights = np.array([max(score, 0.01) for score, _ in top_pool], dtype=np.float64)
    weights = weights / weights.sum()
    selected = top_pool[int(np.random.choice(np.arange(len(top_pool)), p=weights))][1]
    upsert_recommender_feedback(selected["id"], "shown")
    return selected


def update_last_feedback(user: str, item_id: str, feedback: str) -> None:
    rows = read_csv_rows(TWIN_LOG_PATH)
    if not rows:
        return
    idx = None
    for i in range(len(rows) - 1, -1, -1):
        row = rows[i]
        if str(row.get("User", "")) == str(user) and str(row.get("RecommendedId", "")) == str(item_id):
            idx = i
            break
    if idx is None:
        return
    rows[idx]["Feedback"] = feedback
    write_csv_rows(TWIN_LOG_PATH, list(rows[0].keys()), rows)


def log_interaction(
    user: str,
    face_label: str,
    face_mood: str,
    voice_mood: str,
    text_mood: str,
    emoji_mood: str,
    fused_mood: str,
    state_score: float,
    tone_mode: str,
    recommendation: dict[str, Any],
    chat_query: str = "",
    chat_reply: str = "",
    feedback: str = "",
) -> None:
    append_csv_row(
        TWIN_LOG_PATH,
        [
            "User",
            "TimeUTC",
            "FaceLabel",
            "FaceMood",
            "VoiceMood",
            "TextMood",
            "EmojiMood",
            "FusedMood",
            "StateScore",
            "ToneMode",
            "RecommendedId",
            "RecommendedTitle",
            "RecommendedUrl",
            "RecommendedSource",
            "ChatQuery",
            "ChatReply",
            "Feedback",
        ],
        {
            "User": user,
            "TimeUTC": utc_now(),
            "FaceLabel": face_label,
            "FaceMood": face_mood,
            "VoiceMood": voice_mood,
            "TextMood": text_mood,
            "EmojiMood": emoji_mood,
            "FusedMood": fused_mood,
            "StateScore": f"{state_score:.4f}",
            "ToneMode": tone_mode,
            "RecommendedId": recommendation.get("id", ""),
            "RecommendedTitle": recommendation.get("title", ""),
            "RecommendedUrl": recommendation.get("url", ""),
            "RecommendedSource": recommendation.get("source", ""),
            "ChatQuery": chat_query,
            "ChatReply": chat_reply,
            "Feedback": feedback,
        },
    )


def get_user_twin_summary(user: str, limit: int = 12) -> str:
    rows = [row for row in read_csv_rows(TWIN_LOG_PATH) if str(row.get("User", "")) == str(user)]
    if not rows:
        return "No digital twin history is available yet."
    recent = rows[-limit:]
    moods = [str(row.get("FusedMood", "calm") or "calm") for row in recent]
    dominant = Counter(moods).most_common(1)[0][0] if moods else "calm"
    recent_seq = ", ".join(moods[-5:]) if moods else "none"
    last_title = next((row.get("RecommendedTitle", "") for row in reversed(recent) if row.get("RecommendedTitle")), "")
    last_feedback = next((row.get("Feedback", "") for row in reversed(recent) if row.get("Feedback")), "")
    return (
        f"Dominant mood: {dominant}. "
        f"Recent mood sequence: {recent_seq or 'none'}. "
        f"Last recommendation: {last_title or 'not available'}. "
        f"Latest feedback: {last_feedback or 'not recorded'}."
    )


def normalize_label_to_mood(label: str) -> str:
    token = slugify(label).replace("_", " ")
    for key, mood in EMOTION_TO_MOOD.items():
        if key in token:
            return mood
    return "calm"


def tokenize_words(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def ensure_nltk_resources() -> None:
    if not NLTK_AVAILABLE:
        return
    for resource in ["wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                return


@lru_cache(maxsize=256)
def synonyms_for_word(word: str) -> tuple[str, ...]:
    if not NLTK_AVAILABLE:
        return tuple()
    ensure_nltk_resources()
    found: list[str] = []
    try:
        for synset in wn.synsets(word):
            for lemma in synset.lemmas():
                item = lemma.name().replace("_", " ").lower().strip()
                if item and item != word.lower() and item not in found:
                    found.append(item)
                if len(found) >= 8:
                    return tuple(found)
    except Exception:
        return tuple()
    return tuple(found)


INTENT_SEEDS = {
    "stress_relief": {"stress", "overwhelmed", "pressure", "tense", "burnout", "anxious"},
    "motivation": {"motivation", "discipline", "goal", "progress", "improve", "focus"},
    "study_focus": {"study", "exam", "assignment", "college", "project"},
    "loneliness": {"alone", "lonely", "isolated", "empty", "miss"},
    "confidence": {"confidence", "nervous", "presentation", "interview", "fear"},
}


def infer_intent(text: str) -> dict[str, Any]:
    tokens = set(tokenize_words(text))
    intent_scores: dict[str, int] = {}
    matched_terms: dict[str, list[str]] = {}
    for intent, seeds in INTENT_SEEDS.items():
        expanded = set(seeds)
        for seed in list(seeds):
            expanded.update(tokenize_words(" ".join(synonyms_for_word(seed))))
        hits = sorted(token for token in tokens if token in expanded)
        intent_scores[intent] = len(hits)
        matched_terms[intent] = hits
    best = max(intent_scores.items(), key=lambda item: item[1])[0] if intent_scores else "general_support"
    if intent_scores.get(best, 0) == 0:
        best = "general_support"
    preview = sorted({s for t in list(tokens)[:6] for s in synonyms_for_word(t)[:2]})[:8]
    return {"intent": best, "matched_terms": matched_terms.get(best, []), "synonyms_preview": preview}


def simple_text_mood(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ["sad", "cry", "lonely", "grief", "hurt", "down"]):
        return "sad"
    if any(token in lowered for token in ["happy", "joy", "great", "love", "awesome", "good"]):
        return "happy"
    if any(token in lowered for token in ["angry", "mad", "stress", "frustrat", "rage"]):
        return "energetic"
    if any(token in lowered for token in ["calm", "peace", "relax", "stable", "focus"]):
        return "calm"
    return "calm"


@lru_cache(maxsize=1)
def load_text_emotion_model():
    if not TRANSFORMERS_AVAILABLE:
        return None, None
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(TEXT_EMOTION_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(TEXT_EMOTION_MODEL)
        model.eval()
        return tokenizer, model
    except Exception:
        return None, None


def transformer_text_emotion(text: str) -> tuple[str, float, dict[str, float]] | None:
    tokenizer, model = load_text_emotion_model()
    if tokenizer is None or model is None:
        return None
    try:
        encoded = tokenizer(text[:512], truncation=True, return_tensors="pt")
        with torch.no_grad():
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        id2label = model.config.id2label or {}
        mood_scores = {mood: 0.0 for mood in MOOD_CHOICES}
        top_idx = int(np.argmax(probs))
        top_label = id2label.get(top_idx, str(top_idx))
        for idx, prob in enumerate(probs):
            label = str(id2label.get(idx, idx))
            mood_scores[normalize_label_to_mood(label)] += float(prob)
        return str(top_label), float(probs[top_idx]), mood_scores
    except Exception:
        return None


def analyze_text_emotion(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    intent = infer_intent(text) if text else {"intent": "general_support", "matched_terms": [], "synonyms_preview": []}
    mood_scores = {mood: 0.0 for mood in MOOD_CHOICES}
    if not text:
        mood_scores["calm"] = 1.0
        return {
            "text": "",
            "emotion_label": "neutral",
            "emotion_score": 0.50,
            "mood": "calm",
            "confidence": 0.50,
            "intent": intent["intent"],
            "matched_terms": intent["matched_terms"],
            "synonyms_preview": intent["synonyms_preview"],
            "mood_scores": mood_scores,
            "method": "empty_text",
        }
    model_output = transformer_text_emotion(text)
    if model_output is not None:
        label, score, score_map = model_output
        mood_scores.update(score_map)
        mood = max(mood_scores.items(), key=lambda item: item[1])[0]
        return {
            "text": text,
            "emotion_label": label,
            "emotion_score": score,
            "mood": mood,
            "confidence": max(float(score), float(max(mood_scores.values()))),
            "intent": intent["intent"],
            "matched_terms": intent["matched_terms"],
            "synonyms_preview": intent["synonyms_preview"],
            "mood_scores": mood_scores,
            "method": "transformers",
        }
    heuristic_mood = simple_text_mood(text)
    mood_scores[heuristic_mood] = 1.0
    return {
        "text": text,
        "emotion_label": heuristic_mood,
        "emotion_score": 0.60,
        "mood": heuristic_mood,
        "confidence": 0.60,
        "intent": intent["intent"],
        "matched_terms": intent["matched_terms"],
        "synonyms_preview": intent["synonyms_preview"],
        "mood_scores": mood_scores,
        "method": "heuristic",
    }


def transcribe_audio_bytes(audio_bytes: bytes | None) -> tuple[str, str]:
    if not audio_bytes:
        return "", "no_audio"
    if sr is None:
        return "", "speechrecognition_missing"
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
        return text.strip(), "google_web_speech"
    except Exception:
        return "", "transcription_failed"


def estimate_wav_energy(audio_bytes: bytes | None) -> float:
    if not audio_bytes:
        return 0.0
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
            sample_width = wav_file.getsampwidth()
            channels = wav_file.getnchannels()
            frame_count = wav_file.getnframes()
            frames = wav_file.readframes(frame_count)
        if sample_width == 1:
            data = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0
            denom = 128.0
        elif sample_width == 2:
            data = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
            denom = 32768.0
        elif sample_width == 4:
            data = np.frombuffer(frames, dtype=np.int32).astype(np.float32)
            denom = float(2**31)
        else:
            return 0.0
        if channels > 1:
            data = data.reshape(-1, channels).mean(axis=1)
        return float(np.mean(np.abs(data)) / denom)
    except Exception:
        return 0.0


def analyze_voice_emotion(audio_bytes: bytes | None, username: str) -> dict[str, Any]:
    transcript, method = transcribe_audio_bytes(audio_bytes)
    text_result = analyze_text_emotion(transcript) if transcript else analyze_text_emotion("")
    energy = estimate_wav_energy(audio_bytes)
    mood_scores = dict(text_result["mood_scores"])
    if energy > 0.14:
        mood_scores["energetic"] += 0.15
    if energy < 0.04:
        mood_scores["calm"] += 0.08
    if text_result["mood"] == "sad" and energy < 0.05:
        mood_scores["sad"] += 0.10
    mood = max(mood_scores.items(), key=lambda item: item[1])[0]
    spoken_name_detected = bool(username and transcript and username.lower() in transcript.lower())
    return {
        "transcript": transcript,
        "transcription_method": method,
        "energy": energy,
        "mood": mood,
        "confidence": float(max(mood_scores.values()) if mood_scores else 0.0),
        "spoken_name_detected": spoken_name_detected,
        "text_result": text_result,
    }


def decode_image_bytes(image_bytes: bytes | None) -> np.ndarray | None:
    if not image_bytes or cv2 is None:
        return None
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


@lru_cache(maxsize=1)
def get_face_cascade():
    if cv2 is None:
        return None
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return cascade if not cascade.empty() else None


@lru_cache(maxsize=1)
def get_smile_cascade():
    if cv2 is None:
        return None
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
    return cascade if not cascade.empty() else None


def detect_largest_face(image_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    cascade = get_face_cascade()
    if cascade is None or cv2 is None:
        return None
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    return max(faces, key=lambda box: int(box[2] * box[3]))


def heuristic_face_label(face_bgr: np.ndarray) -> tuple[str, float]:
    if cv2 is None:
        return "neutral", 0.40
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    texture = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    smile_found = False
    smile_cascade = get_smile_cascade()
    if smile_cascade is not None:
        try:
            smiles = smile_cascade.detectMultiScale(gray, scaleFactor=1.7, minNeighbors=20, minSize=(25, 25))
            smile_found = len(smiles) > 0
        except Exception:
            smile_found = False
    if smile_found or brightness > 150:
        return "happy", 0.58
    if texture > 450 and brightness < 135:
        return "angry", 0.47
    if brightness < 95:
        return "sad", 0.46
    return "neutral", 0.44


def build_efficientnet_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    try:
        weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        model = efficientnet_v2_s(weights=weights)
    except Exception:
        model = efficientnet_v2_s(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(p=0.30), nn.Linear(in_features, num_classes))
    return model


def save_model_checkpoint(model: nn.Module, class_names: list[str], extra: dict[str, Any] | None = None) -> None:
    checkpoint = {
        "architecture": "efficientnet_v2_s",
        "class_names": class_names,
        "saved_at_utc": utc_now(),
        "model_state_dict": model.state_dict(),
    }
    if extra:
        checkpoint.update(extra)
    torch.save(checkpoint, MODEL_PATH)
    metadata = {
        "architecture": "efficientnet_v2_s",
        "class_names": class_names,
        "saved_at_utc": checkpoint.get("saved_at_utc", utc_now()),
        "model_path": str(MODEL_PATH),
    }
    if extra:
        metadata.update(extra)
    MODEL_META_PATH.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")


@lru_cache(maxsize=1)
def load_saved_model() -> tuple[nn.Module | None, dict[str, Any]]:
    if not MODEL_PATH.exists():
        return None, {}
    try:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
        class_names = checkpoint.get("class_names") or DEFAULT_FACE_CLASSES
        model = build_efficientnet_model(len(class_names), pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        model.eval()
        metadata = {"class_names": class_names}
        if MODEL_META_PATH.exists():
            try:
                metadata.update(json.loads(MODEL_META_PATH.read_text(encoding="utf-8")))
            except Exception:
                pass
        return model, metadata
    except Exception:
        return None, {}


def get_inference_transform():
    try:
        weights = EfficientNet_V2_S_Weights.DEFAULT
        mean = list(weights.meta.get("mean", [0.485, 0.456, 0.406]))
        std = list(weights.meta.get("std", [0.229, 0.224, 0.225]))
    except Exception:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def get_last_conv_module(model: nn.Module):
    last_name = None
    last_module = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_name = name
            last_module = module
    return last_name, last_module


def compute_gradcam(model: nn.Module, image_tensor: torch.Tensor, target_idx: int | None = None) -> np.ndarray | None:
    _last_name, target_module = get_last_conv_module(model)
    if target_module is None:
        return None
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def forward_hook(_module, _inputs, output):
        activations.append(output.detach())

    def backward_hook(_module, _grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    h1 = target_module.register_forward_hook(forward_hook)
    h2 = target_module.register_full_backward_hook(backward_hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(image_tensor)
        if target_idx is None:
            target_idx = int(torch.argmax(logits, dim=1).item())
        score = logits[:, target_idx].sum()
        score.backward()
        if not activations or not gradients:
            return None
        acts = activations[-1][0]
        grads = gradients[-1][0]
        weights = grads.mean(dim=(1, 2))
        cam = torch.relu((weights[:, None, None] * acts).sum(dim=0))
        if float(cam.max()) <= 0:
            return None
        cam = cam / cam.max()
        return np.uint8(255 * cam.cpu().numpy())
    except Exception:
        return None
    finally:
        h1.remove()
        h2.remove()


def overlay_heatmap(face_bgr: np.ndarray, heatmap: np.ndarray | None) -> np.ndarray | None:
    if heatmap is None or cv2 is None:
        return None
    resized = cv2.resize(heatmap, (face_bgr.shape[1], face_bgr.shape[0]))
    colored = cv2.applyColorMap(resized, cv2.COLORMAP_JET)
    return cv2.addWeighted(colored, 0.35, face_bgr, 0.65, 0)


def predict_face_emotion(image_bytes: bytes | None) -> dict[str, Any]:
    if not image_bytes:
        return {
            "label": "neutral",
            "mood": "calm",
            "confidence": 0.0,
            "method": "no_image",
            "overlay_rgb": None,
            "face_found": False,
        }
    image_bgr = decode_image_bytes(image_bytes)
    if image_bgr is None or cv2 is None:
        return {
            "label": "neutral",
            "mood": "calm",
            "confidence": 0.0,
            "method": "opencv_missing",
            "overlay_rgb": None,
            "face_found": False,
        }
    face_box = detect_largest_face(image_bgr)
    display = image_bgr.copy()
    if face_box is None:
        face_crop = image_bgr
        x, y, w, h = 0, 0, image_bgr.shape[1], image_bgr.shape[0]
    else:
        x, y, w, h = map(int, face_box)
        face_crop = image_bgr[y : y + h, x : x + w]
        cv2.rectangle(display, (x, y), (x + w, y + h), (70, 255, 140), 2)
    model, metadata = load_saved_model()
    if model is not None:
        try:
            transform = get_inference_transform()
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            tensor = transform(face_rgb).unsqueeze(0)
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            class_names = metadata.get("class_names") or DEFAULT_FACE_CLASSES
            idx = int(np.argmax(probs))
            label = class_names[idx] if idx < len(class_names) else f"class_{idx}"
            confidence = float(probs[idx])
            heatmap = compute_gradcam(model, tensor, idx)
            overlay = overlay_heatmap(face_crop, heatmap)
            if overlay is not None and face_box is not None:
                display[y : y + h, x : x + w] = cv2.resize(overlay, (w, h))
                cv2.rectangle(display, (x, y), (x + w, y + h), (70, 255, 140), 2)
            elif overlay is not None:
                display = overlay
            return {
                "label": str(label),
                "mood": normalize_label_to_mood(str(label)),
                "confidence": confidence,
                "method": "trained_efficientnet_v2_s",
                "overlay_rgb": cv2.cvtColor(display, cv2.COLOR_BGR2RGB),
                "face_found": face_box is not None,
            }
        except Exception:
            pass
    label, conf = heuristic_face_label(face_crop)
    return {
        "label": label,
        "mood": normalize_label_to_mood(label),
        "confidence": conf,
        "method": "opencv_heuristic",
        "overlay_rgb": cv2.cvtColor(display, cv2.COLOR_BGR2RGB),
        "face_found": face_box is not None,
    }


def dataset_has_images(path: Path) -> bool:
    if not path.exists():
        return False
    for item in path.rglob("*"):
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS:
            return True
    return False


def dataset_summary(root: Path = DATASET_DIR) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ["train", "val", "test"]:
        split_dir = root / split
        if not split_dir.exists():
            continue
        for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            count = sum(
                1
                for item in class_dir.iterdir()
                if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
            )
            rows.append({"split": split, "class_name": class_dir.name, "count": count})
    return rows


def build_dataloaders(root: Path, batch_size: int):
    train_dir = root / "train"
    val_dir = root / "val"
    test_dir = root / "test"
    if not dataset_has_images(train_dir):
        raise RuntimeError("No training images found in dataset/train.")
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_tfms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(8),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    train_ds = tv_datasets.ImageFolder(train_dir, transform=train_tfms)
    class_names = list(train_ds.classes)
    val_ds = tv_datasets.ImageFolder(val_dir, transform=eval_tfms) if dataset_has_images(val_dir) else None
    test_ds = tv_datasets.ImageFolder(test_dir, transform=eval_tfms) if dataset_has_images(test_dir) else None
    if val_ds is not None and list(val_ds.classes) != class_names:
        raise RuntimeError("Validation classes must match training classes.")
    if test_ds is not None and list(test_ds.classes) != class_names:
        raise RuntimeError("Test classes must match training classes.")
    kwargs = {"batch_size": batch_size, "num_workers": 0, "pin_memory": torch.cuda.is_available()}
    train_loader = DataLoader(train_ds, shuffle=True, **kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **kwargs) if val_ds is not None else None
    test_loader = DataLoader(test_ds, shuffle=False, **kwargs) if test_ds is not None else None
    return train_loader, val_loader, test_loader, class_names


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer, criterion, device: torch.device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * labels.size(0)
        total_correct += int((torch.argmax(logits, dim=1) == labels).sum().item())
        total_count += int(labels.size(0))
    return total_loss / max(total_count, 1), total_correct / max(total_count, 1)


def eval_one_epoch(model: nn.Module, loader: DataLoader, criterion, device: torch.device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += float(loss.item()) * labels.size(0)
            total_correct += int((torch.argmax(logits, dim=1) == labels).sum().item())
            total_count += int(labels.size(0))
    return total_loss / max(total_count, 1), total_correct / max(total_count, 1)


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            matrix[t, p] += 1
    return matrix


def classification_report_np(matrix: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    total = int(matrix.sum())
    report: dict[str, Any] = {}
    precisions = []
    recalls = []
    f1s = []
    weights = []
    for i, name in enumerate(class_names):
        tp = int(matrix[i, i])
        fp = int(matrix[:, i].sum() - tp)
        fn = int(matrix[i, :].sum() - tp)
        support = int(matrix[i, :].sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        report[name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1-score": round(f1, 4),
            "support": support,
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        weights.append(support)
    report["accuracy"] = round(float(np.trace(matrix) / max(total, 1)), 4)
    report["macro avg"] = {
        "precision": round(float(np.mean(precisions)) if precisions else 0.0, 4),
        "recall": round(float(np.mean(recalls)) if recalls else 0.0, 4),
        "f1-score": round(float(np.mean(f1s)) if f1s else 0.0, 4),
        "support": total,
    }
    if sum(weights) > 0:
        report["weighted avg"] = {
            "precision": round(float(np.average(precisions, weights=weights)), 4),
            "recall": round(float(np.average(recalls, weights=weights)), 4),
            "f1-score": round(float(np.average(f1s, weights=weights)), 4),
            "support": total,
        }
    return report


def multiclass_roc_auc_macro(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int) -> float | None:
    def _auc_binary(y_bin: np.ndarray, scores: np.ndarray) -> float | None:
        pos = int(y_bin.sum())
        neg = int(len(y_bin) - pos)
        if pos == 0 or neg == 0:
            return None
        order = np.argsort(scores)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(scores) + 1)
        rank_sum = float(ranks[y_bin == 1].sum())
        return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)

    aucs: list[float] = []
    for i in range(num_classes):
        auc = _auc_binary((y_true == i).astype(np.int32), y_prob[:, i])
        if auc is not None:
            aucs.append(auc)
    if not aucs:
        return None
    return float(np.mean(aucs))


def evaluate_model(model: nn.Module, loader: DataLoader | None, class_names: list[str], device: torch.device) -> dict[str, Any]:
    if loader is None:
        return {"message": "No validation or test split found for evaluation."}
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[list[float]] = []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(preds.tolist())
            y_prob.extend(probs.tolist())
    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_pred_np = np.asarray(y_pred, dtype=np.int64)
    y_prob_np = np.asarray(y_prob, dtype=np.float64)
    matrix = confusion_matrix_np(y_true_np, y_pred_np, len(class_names))
    report = classification_report_np(matrix, class_names)
    roc_auc = multiclass_roc_auc_macro(y_true_np, y_prob_np, len(class_names))
    return {"report": report, "confusion_matrix": matrix.tolist(), "roc_auc_macro": roc_auc}


def train_emotion_model(
    dataset_root: Path,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    freeze_backbone: bool,
    progress_callback=None,
):
    train_loader, val_loader, test_loader, class_names = build_dataloaders(dataset_root, batch_size=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_efficientnet_model(len(class_names), pretrained=True)
    if freeze_backbone:
        for p in model.features.parameters():
            p.requires_grad = False
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        if val_loader is not None:
            va_loss, va_acc = eval_one_epoch(model, val_loader, criterion, device)
        else:
            va_loss, va_acc = 0.0, 0.0
        metrics = {
            "epoch": float(epoch),
            "train_loss": round(tr_loss, 4),
            "train_accuracy": round(tr_acc, 4),
            "val_loss": round(va_loss, 4),
            "val_accuracy": round(va_acc, 4),
        }
        history.append(metrics)
        if progress_callback is not None:
            progress_callback(epoch / epochs, metrics)
    save_model_checkpoint(
        model.cpu(),
        class_names,
        extra={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "freeze_backbone": freeze_backbone,
        },
    )
    load_saved_model.cache_clear()
    evaluation = evaluate_model(model.to(device), test_loader or val_loader, class_names, device)
    return history, evaluation, class_names


def emotion_fusion(face: dict[str, Any], voice: dict[str, Any], text: dict[str, Any], emoji_mood: str):
    scores = {mood: 0.0 for mood in MOOD_CHOICES}
    scores[face.get("mood", "calm")] += 0.35 * max(float(face.get("confidence") or 0.45), 0.30)
    scores[voice.get("mood", "calm")] += 0.25 * max(float(voice.get("confidence") or 0.45), 0.30)
    scores[text.get("mood", "calm")] += 0.25 * max(float(text.get("confidence") or 0.45), 0.30)
    scores[emoji_mood] += 0.15
    fused = max(scores.items(), key=lambda item: item[1])[0]
    state_score = scores[fused] / (sum(scores.values()) or 1.0)
    return fused, scores, float(state_score)


def ethical_monitor(face: dict[str, Any], voice: dict[str, Any], text: dict[str, Any], scores: dict[str, float]):
    warnings_list: list[str] = []
    used_modalities = 0
    if face.get("method") != "no_image":
        used_modalities += 1
    if voice.get("transcription_method") != "no_audio":
        used_modalities += 1
    if text.get("text"):
        used_modalities += 1
    if face.get("method") == "opencv_heuristic":
        warnings_list.append("Face analysis is currently heuristic. Train and save a .pth model for better accuracy.")
    if used_modalities < 2:
        warnings_list.append("Add at least two modalities for stronger fusion confidence.")
    peak = max(scores.values()) if scores else 0.0
    if peak < 0.25:
        warnings_list.append("Signals are mixed. The confidence is moderate.")
    warnings_list.append("This chatbot is supportive guidance, not medical diagnosis.")
    band = "High" if peak >= 0.45 else "Medium" if peak >= 0.28 else "Low"
    return {"confidence_band": band, "modalities_used": used_modalities, "warnings": warnings_list}


def get_chat_provider_name() -> str:
    if HF_TOKEN:
        return f"Hugging Face Inference API ({HF_CHAT_MODEL})"
    return "Local rule-based coach"


def huggingface_chat(system_prompt: str, user_prompt: str) -> str | None:
    if not HF_TOKEN:
        return None
    payload = {
        "inputs": f"Instruction:\n{system_prompt}\n\nUser:\n{user_prompt}\n\nAssistant:\n",
        "parameters": {
            "max_new_tokens": 220,
            "temperature": 0.7,
            "top_p": 0.9,
            "return_full_text": False,
        },
        "options": {"wait_for_model": True},
    }
    req = urllib.request.Request(
        url=f"https://api-inference.huggingface.co/models/{HF_CHAT_MODEL}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as response:
            body = response.read().decode("utf-8")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, Exception):
        return None
    try:
        parsed = json.loads(body)
    except Exception:
        return body.strip() or None
    if isinstance(parsed, dict):
        if parsed.get("error"):
            return None
        if isinstance(parsed.get("generated_text"), str):
            return parsed["generated_text"].strip()
    if isinstance(parsed, list) and parsed:
        first = parsed[0]
        if isinstance(first, dict) and isinstance(first.get("generated_text"), str):
            return first["generated_text"].strip()
        if isinstance(first, str):
            return first.strip()
    return None


def local_chat_response(prompt: str, mood: str, tone_mode: str, twin_summary: str, intent: str) -> str:
    questions = MOOD_QUESTION_BANK.get(mood, MOOD_QUESTION_BANK["calm"])
    lead = {
        "Therapist": "I hear you. Let us process this carefully and practically.",
        "Friendly": "I am with you. Let us keep this simple and helpful.",
        "Motivational": "You have momentum. Let us direct it toward a clear next step.",
    }.get(tone_mode, "I am here to support you.")
    return textwrap.dedent(
        f"""
        {lead}

        Mood context: {mood}. Detected intent: {intent}.
        Twin memory: {twin_summary}

        Reflection:
        - Your message carries a valid emotional signal and deserves a focused response.

        Three important questions:
        1. {questions[0]}
        2. {questions[1]}
        3. {questions[2]}

        One next action (10-20 mins):
        - {FALLBACK_ACTIONS.get(mood, FALLBACK_ACTIONS["calm"])}
        """
    ).strip()


def generate_chat_reply(username: str, message: str, current_mood: str, tone_mode: str, analysis: dict[str, Any] | None):
    text_result = analyze_text_emotion(message)
    mood = current_mood if current_mood in MOOD_CHOICES else text_result["mood"]
    twin_summary = get_user_twin_summary(username)
    snapshot = "No multimodal analysis available yet."
    if analysis:
        snapshot = (
            f"Face={analysis['face_result']['mood']}, Voice={analysis['voice_result']['mood']}, "
            f"Text={analysis['text_result']['mood']}, Emoji={analysis['emoji_mood']}, "
            f"Fused={analysis['fused_mood']}, StateScore={analysis['state_score']:.2f}"
        )
    system_prompt = (
        f"You are an emotionally intelligent therapeutic chatbot.\n"
        f"User: {username}\nMood: {mood}\nTone: {tone_mode}\n"
        f"Tone guide: {TONE_GUIDES.get(tone_mode, TONE_GUIDES['Therapist'])}\n"
        f"Twin summary: {twin_summary}\nAnalysis snapshot: {snapshot}\n"
        "Rules: Give one reflection, exactly three meaningful questions, and one practical action. "
        "Avoid generic FAQ replies and avoid medical diagnosis."
    )
    user_prompt = (
        f"User message: {message}\n"
        f"Detected intent: {text_result['intent']}\n"
        f"Matched terms: {', '.join(text_result['matched_terms']) or 'none'}\n"
        f"Synonym hints: {', '.join(text_result['synonyms_preview']) or 'none'}"
    )
    hf = huggingface_chat(system_prompt, user_prompt)
    if hf:
        return hf, get_chat_provider_name()
    return local_chat_response(message, mood, tone_mode, twin_summary, text_result["intent"]), get_chat_provider_name()


def speak_text(text: str) -> str:
    if pyttsx3 is None:
        return "pyttsx3 is not installed, so text-to-speech is unavailable."
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 170)
        engine.say(text)
        engine.runAndWait()
        return "Reply spoken on local machine."
    except Exception:
        return "Text-to-speech failed on this environment."


def render_mascot(mood: str, username: str, tone_mode: str, subtitle: str) -> None:
    require_streamlit()
    mood = mood if mood in MOOD_CHOICES else "calm"
    palette = {
        "happy": {"bg": "#fff9db", "face": "#ffd866", "accent": "#ff922b", "mouth": "smile", "eye": "open"},
        "sad": {"bg": "#eef2ff", "face": "#a5b4fc", "accent": "#4c6ef5", "mouth": "sad", "eye": "soft"},
        "calm": {"bg": "#ecfeff", "face": "#99f6e4", "accent": "#0f766e", "mouth": "calm", "eye": "soft"},
        "energetic": {"bg": "#fff1f2", "face": "#fda4af", "accent": "#e11d48", "mouth": "grin", "eye": "sharp"},
    }[mood]
    safe_subtitle = html.escape(subtitle[:200] if subtitle else f"{username}, I am tuned to your {mood} state.")
    mouth_html = {
        "smile": '<div class="cei-mouth cei-mouth-smile"></div>',
        "sad": '<div class="cei-mouth cei-mouth-sad"></div>',
        "calm": '<div class="cei-mouth cei-mouth-calm"></div>',
        "grin": '<div class="cei-mouth cei-mouth-grin"></div>',
    }[palette["mouth"]]
    eye_class = {
        "open": "cei-eye-open",
        "soft": "cei-eye-soft",
        "sharp": "cei-eye-sharp",
    }[palette["eye"]]
    block = f"""
    <style>
    .cei-card {{
      background: {palette["bg"]};
      border-radius: 20px;
      padding: 16px;
      border: 1px solid rgba(0,0,0,0.08);
      box-shadow: 0 8px 16px rgba(15,23,42,0.08);
    }}
    .cei-bubble {{
      background: #ffffff;
      border-radius: 14px;
      padding: 10px 12px;
      margin-bottom: 10px;
      font-size: 0.95rem;
    }}
    .cei-stage {{
      min-height: 230px;
      display: flex;
      justify-content: center;
      align-items: center;
      position: relative;
    }}
    .cei-glow {{
      position: absolute;
      width: 220px; height: 220px;
      background: radial-gradient(circle, {palette["accent"]}44 0%, transparent 70%);
      animation: ceiPulse 2.6s ease-in-out infinite;
    }}
    .cei-avatar {{
      position: relative;
      width: 170px; height: 170px;
      border-radius: 999px;
      background: {palette["face"]};
      animation: ceiFloat 2.4s ease-in-out infinite;
      box-shadow: 0 16px 24px rgba(15,23,42,0.16);
    }}
    .cei-eye {{
      position: absolute;
      top: 64px;
      width: 16px; height: 16px;
      border-radius: 999px;
      background: #111827;
      animation: ceiBlink 4.2s infinite;
    }}
    .cei-eye-left {{ left: 46px; }}
    .cei-eye-right {{ right: 46px; }}
    .cei-eye-soft {{ height: 12px; top: 67px; }}
    .cei-eye-sharp {{ transform: skewX(-12deg); }}
    .cei-mouth {{ position: absolute; left: 50%; transform: translateX(-50%); bottom: 42px; }}
    .cei-mouth-smile {{ width: 52px; height: 26px; border-bottom: 6px solid #7f1d1d; border-radius: 0 0 70px 70px; }}
    .cei-mouth-sad {{ width: 52px; height: 26px; border-top: 6px solid #1e293b; border-radius: 70px 70px 0 0; }}
    .cei-mouth-calm {{ width: 42px; border-top: 5px solid #0f172a; border-radius: 20px; }}
    .cei-mouth-grin {{ width: 62px; height: 14px; border-bottom: 6px solid #7f1d1d; border-radius: 0 0 80px 80px; }}
    .cei-badge {{
      background: #fff;
      border-radius: 999px;
      display: inline-block;
      padding: 7px 12px;
      margin-top: 8px;
      font-weight: 600;
      box-shadow: 0 5px 12px rgba(15,23,42,0.08);
    }}
    @keyframes ceiFloat {{ 0%,100%{{transform:translateY(0)}} 50%{{transform:translateY(-8px)}} }}
    @keyframes ceiPulse {{ 0%,100%{{transform:scale(0.92)}} 50%{{transform:scale(1.05)}} }}
    @keyframes ceiBlink {{ 0%,92%,100%{{transform:scaleY(1)}} 94%,96%{{transform:scaleY(0.1)}} }}
    </style>
    <div class="cei-card">
      <div class="cei-bubble">{safe_subtitle}</div>
      <div class="cei-stage">
        <div class="cei-glow"></div>
        <div class="cei-avatar">
          <div class="cei-eye cei-eye-left {eye_class}"></div>
          <div class="cei-eye cei-eye-right {eye_class}"></div>
          {mouth_html}
        </div>
      </div>
      <div class="cei-badge">Mascot mode: {html.escape(tone_mode)} | Mood: {html.escape(mood.title())}</div>
    </div>
    """
    st.markdown(block, unsafe_allow_html=True)


def render_recommendation(item: dict[str, Any]) -> None:
    st.subheader(item["title"])
    st.caption(f"{item['mood'].title()} | {item['source']} | {item['resource_type']}")
    if item.get("playable") and "youtube.com/watch" in item.get("url", ""):
        st.video(item["url"])
    else:
        st.markdown(f"[Open resource]({item['url']})")
    st.caption(item.get("offline_fallback", ""))


def render_mood_scores(scores: dict[str, float]) -> None:
    total = sum(scores.values()) or 1.0
    for mood in MOOD_CHOICES:
        ratio = float(scores.get(mood, 0.0) / total)
        st.write(f"{mood.title()}: {ratio:.2%}")
        st.progress(min(max(ratio, 0.0), 1.0))


def perform_live_analysis(
    user: str,
    emoji: str,
    image_bytes: bytes | None,
    audio_bytes: bytes | None,
    text_input: str,
    tone_mode: str,
):
    face = predict_face_emotion(image_bytes)
    voice = analyze_voice_emotion(audio_bytes, username=user)
    text = analyze_text_emotion(text_input)
    emoji_mood = EMOJI_MAP.get(emoji, "calm")
    fused, scores, state_score = emotion_fusion(face, voice, text, emoji_mood)
    recommendation = recommend_resource(user, fused, last_n=5)
    monitor = ethical_monitor(face, voice, text, scores)
    result = {
        "face_result": face,
        "voice_result": voice,
        "text_result": text,
        "emoji_mood": emoji_mood,
        "fused_mood": fused,
        "fusion_scores": scores,
        "state_score": state_score,
        "recommendation": recommendation,
        "ethical_report": monitor,
        "tone_mode": tone_mode,
    }
    log_interaction(
        user=user,
        face_label=str(face.get("label", "neutral")),
        face_mood=str(face.get("mood", "calm")),
        voice_mood=str(voice.get("mood", "calm")),
        text_mood=str(text.get("mood", "calm")),
        emoji_mood=emoji_mood,
        fused_mood=fused,
        state_score=state_score,
        tone_mode=tone_mode,
        recommendation=recommendation,
    )
    return result


def run_cli() -> int:
    parser = argparse.ArgumentParser(description="Emotion AI utility")
    parser.add_argument("--export-catalog", action="store_true", help="Export resource catalog csv.")
    parser.add_argument("--catalog-path", default=str(CATALOG_EXPORT_PATH), help="Path for exported catalog csv.")
    args, _ = parser.parse_known_args()
    ensure_runtime_files()
    if args.export_catalog:
        export_resource_catalog(Path(args.catalog_path))
        print(f"Catalog exported to: {args.catalog_path}")
        return 0
    return -1


def render_setup_tab() -> None:
    st.subheader("Complete VS Code execution guide")
    st.markdown(
        textwrap.dedent(
            """
            1. Create project files:
               - `app.py`
               - `requirements.txt`
               - optional `.gitignore`

            2. Create and activate virtual environment (Windows PowerShell):
               ```powershell
               python -m venv .venv
               .\\.venv\\Scripts\\Activate.ps1
               ```

            3. Install dependencies:
               ```powershell
               python -m pip install --upgrade pip
               pip install -r requirements.txt
               ```

            4. Optional free Hugging Face API token:
               - Create free account on huggingface.co
               - Settings -> Access Tokens -> New token (Read)
               - Set token:
               ```powershell
               $env:HF_TOKEN="your_token_here"
               ```

            5. Run the app:
               ```powershell
               streamlit run app.py
               ```
            """
        )
    )
    st.markdown("### Required folder structure")
    st.code(
        textwrap.dedent(
            """
            project/
            ├─ app.py
            ├─ requirements.txt
            ├─ models/                  # auto-created
            ├─ dataset/                 # add train/val/test for training
            ├─ cei_twin_log.csv         # auto-created
            ├─ recommender_stats.csv    # auto-created
            └─ resource_catalog.csv     # auto-created
            """
        ).strip(),
        language="text",
    )
    st.markdown("### requirements.txt content")
    st.code(
        "\n".join(
            [
                "torch",
                "torchvision",
                "transformers",
                "streamlit",
                "opencv-python",
                "numpy",
                "nltk",
                "SpeechRecognition",
                "pyttsx3",
            ]
        ),
        language="text",
    )
    st.info(
        "This app intentionally avoids TensorFlow/Keras. "
        "It uses PyTorch EfficientNetV2-S and free Hugging Face API integration."
    )
    st.caption(
        "On low RAM laptops, keep batch size 4-8 and epochs 1-3 for smooth training demos."
    )


def main() -> None:
    if run_cli() == 0:
        return
    require_streamlit()
    ensure_runtime_files()
    export_resource_catalog()

    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)

    if "analysis_result" not in st.session_state:
        st.session_state["analysis_result"] = None
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "feedback_saved" not in st.session_state:
        st.session_state["feedback_saved"] = None

    with st.sidebar:
        st.header("Inputs")
        username = st.text_input("Username", value="guest_user")
        tone_mode = st.selectbox("Adaptive reply mode", options=list(TONE_GUIDES.keys()), index=0)
        emoji = st.select_slider("Emoji mood input", options=list(EMOJI_MAP.keys()), value="😊")
        st.caption(f"Chat provider: {get_chat_provider_name()}")
        speak_reply = st.checkbox("Speak chatbot replies on local machine", value=False)
        st.caption("No TensorFlow/Keras is used in this app.")

    tab_live, tab_chat, tab_train, tab_twin, tab_setup = st.tabs(
        ["Live Emotion Studio", "Mascot Chatbot", "Training + Grad-CAM", "Digital Twin + Catalog", "Setup Guide"]
    )

    with tab_live:
        st.subheader("Multimodal emotion analysis")
        left, right = st.columns([1.2, 1.0])
        with left:
            image_file = st.camera_input("Capture face image")
            image_bytes = image_file.getvalue() if image_file is not None else None
            if hasattr(st, "audio_input"):
                audio_file = st.audio_input("Record voice")
                audio_bytes = audio_file.getvalue() if audio_file is not None else None
            else:
                audio_bytes = None
                st.info("Upgrade Streamlit for browser microphone capture support (`st.audio_input`).")
            text_input = st.text_area("Context text", placeholder="Describe how you feel and what you need.")
            if st.button("Analyze emotional state", type="primary"):
                with st.spinner("Running multimodal analysis..."):
                    st.session_state["analysis_result"] = perform_live_analysis(
                        user=username,
                        emoji=emoji,
                        image_bytes=image_bytes,
                        audio_bytes=audio_bytes,
                        text_input=text_input,
                        tone_mode=tone_mode,
                    )
                    st.session_state["feedback_saved"] = None
        with right:
            result = st.session_state.get("analysis_result")
            mood = result["fused_mood"] if result else "calm"
            subtitle = (
                f"{username}, I am ready to analyse face, voice, text, and emoji."
                if result is None
                else f"{username}, your current fused emotional state is {mood}."
            )
            render_mascot(mood, username, tone_mode, subtitle)

        result = st.session_state.get("analysis_result")
        if result:
            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Face mood", result["face_result"]["mood"].title())
            c2.metric("Voice mood", result["voice_result"]["mood"].title())
            c3.metric("Text mood", result["text_result"]["mood"].title())
            c4.metric("Fused mood", result["fused_mood"].title())
            st.metric("User Emotional State Score", f"{result['state_score']:.2%}")

            a, b = st.columns([1.2, 1.0])
            with a:
                st.markdown("#### Face analysis + Grad-CAM")
                if result["face_result"].get("overlay_rgb") is not None:
                    st.image(result["face_result"]["overlay_rgb"], caption=result["face_result"]["method"])
                else:
                    st.info("No face image available yet.")
            with b:
                st.markdown("#### Fusion scores")
                render_mood_scores(result["fusion_scores"])
                st.markdown("#### Ethical AI monitor")
                st.write(f"Confidence band: {result['ethical_report']['confidence_band']}")
                st.write(f"Modalities used: {result['ethical_report']['modalities_used']}")
                for line in result["ethical_report"]["warnings"]:
                    st.caption(f"- {line}")

            st.markdown("#### Voice transcript")
            voice = result["voice_result"]
            if voice["transcript"]:
                st.write(voice["transcript"])
                st.caption(f"Method: {voice['transcription_method']} | Energy: {voice['energy']:.3f}")
                if voice["spoken_name_detected"]:
                    st.success("Username was detected in voice transcript.")
            else:
                st.info("No transcript detected from voice input.")

            st.markdown("#### Text emotion details")
            text_result = result["text_result"]
            st.write(
                f"Method: {text_result['method']} | Emotion: {text_result['emotion_label']} "
                f"({text_result['emotion_score']:.2f}) | Intent: {text_result['intent']}"
            )
            if text_result["matched_terms"]:
                st.caption("Matched terms: " + ", ".join(text_result["matched_terms"]))
            if text_result["synonyms_preview"]:
                st.caption("Synonym hints: " + ", ".join(text_result["synonyms_preview"]))

            st.markdown("#### Adaptive recommendation")
            render_recommendation(result["recommendation"])
            lcol, scol = st.columns(2)
            with lcol:
                if st.button("I liked this", disabled=st.session_state["feedback_saved"] is not None):
                    upsert_recommender_feedback(result["recommendation"]["id"], "liked")
                    update_last_feedback(username, result["recommendation"]["id"], "liked")
                    st.session_state["feedback_saved"] = "liked"
                    st.success("Feedback saved.")
            with scol:
                if st.button("Skip this", disabled=st.session_state["feedback_saved"] is not None):
                    upsert_recommender_feedback(result["recommendation"]["id"], "skipped")
                    update_last_feedback(username, result["recommendation"]["id"], "skipped")
                    st.session_state["feedback_saved"] = "skipped"
                    st.info("Feedback saved.")

    with tab_chat:
        st.subheader("Animated mascot chatbot")
        analysis = st.session_state.get("analysis_result")
        current_mood = analysis["fused_mood"] if analysis else "calm"
        render_mascot(
            current_mood,
            username,
            tone_mode,
            f"{username}, ask anything. I will respond according to your emotional context.",
        )
        for role, message in st.session_state["chat_history"]:
            with st.chat_message(role):
                st.write(message)
        prompt = st.chat_input("Ask your question...")
        if prompt:
            with st.chat_message("user"):
                st.write(prompt)
            reply, provider = generate_chat_reply(username, prompt, current_mood, tone_mode, analysis)
            with st.chat_message("assistant"):
                st.write(reply)
                st.caption(f"Provider: {provider}")
            st.session_state["chat_history"].append(("user", prompt))
            st.session_state["chat_history"].append(("assistant", reply))
            rec = analysis["recommendation"] if analysis else {"id": "", "title": "", "url": "", "source": ""}
            log_interaction(
                user=username,
                face_label=analysis["face_result"]["label"] if analysis else "neutral",
                face_mood=analysis["face_result"]["mood"] if analysis else "calm",
                voice_mood=analysis["voice_result"]["mood"] if analysis else "calm",
                text_mood=analysis["text_result"]["mood"] if analysis else simple_text_mood(prompt),
                emoji_mood=analysis["emoji_mood"] if analysis else "calm",
                fused_mood=current_mood,
                state_score=analysis["state_score"] if analysis else 0.0,
                tone_mode=tone_mode,
                recommendation=rec,
                chat_query=prompt,
                chat_reply=reply,
            )
            if speak_reply:
                st.info(speak_text(reply))
        if st.button("Clear chat history"):
            st.session_state["chat_history"] = []
            st.success("Chat history cleared.")

    with tab_train:
        st.subheader("PyTorch EfficientNetV2-S training")
        st.caption("Expected dataset folder: dataset/train, dataset/val, dataset/test with class subfolders.")
        summary = dataset_summary(DATASET_DIR)
        if summary:
            st.dataframe(summary)
        else:
            st.info("No dataset found yet. Add folders in dataset/ before training.")

        batch_size = st.select_slider("Batch size", options=[4, 8, 12, 16], value=8)
        learning_rate = st.select_slider("Learning rate", options=[1e-4, 3e-4, 1e-3], value=3e-4)
        epochs = st.slider("Epochs", min_value=1, max_value=6, value=2)
        freeze_backbone = st.checkbox("Freeze EfficientNet backbone", value=True)

        if st.button("Train model"):
            progress = st.progress(0.0)
            meter = st.empty()

            def cb(frac, metrics):
                progress.progress(float(frac))
                meter.write(metrics)

            try:
                with st.spinner("Training in progress..."):
                    history, evaluation, class_names = train_emotion_model(
                        dataset_root=DATASET_DIR,
                        batch_size=batch_size,
                        epochs=epochs,
                        learning_rate=float(learning_rate),
                        freeze_backbone=freeze_backbone,
                        progress_callback=cb,
                    )
                st.success(f"Training complete. Saved model: {MODEL_PATH.name}")
                st.write("Classes:", ", ".join(class_names))
                st.table(history)
                st.json(evaluation)
            except Exception as exc:
                st.error(f"Training failed: {exc}")

        if MODEL_META_PATH.exists():
            st.markdown("#### Saved model metadata")
            try:
                st.json(json.loads(MODEL_META_PATH.read_text(encoding="utf-8")))
            except Exception:
                st.info("Could not parse model metadata.")

    with tab_twin:
        st.subheader("Digital twin logs and resource catalog")
        twin_rows = read_csv_rows(TWIN_LOG_PATH)
        if twin_rows:
            st.markdown("#### Interaction log")
            st.dataframe(twin_rows)
            st.download_button("Download twin log", TWIN_LOG_PATH.read_bytes(), file_name=TWIN_LOG_PATH.name, mime="text/csv")
        else:
            st.info("No interactions logged yet.")

        st.markdown("#### Recommender stats")
        r_rows = recommender_rows()
        if r_rows:
            st.dataframe(r_rows)
        else:
            st.info("No recommender stats yet.")

        st.markdown("#### Resource catalog")
        catalog = catalog_rows()
        mood_filter = st.selectbox("Mood filter", options=["all"] + MOOD_CHOICES, index=0)
        source_filter = st.selectbox("Source filter", options=["all", "YouTube", "Spotify"], index=0)
        filtered = []
        for item in catalog:
            if mood_filter != "all" and item["mood"] != mood_filter:
                continue
            if source_filter != "all" and item["source"] != source_filter:
                continue
            filtered.append(item)
        st.write(f"Catalog items: {len(filtered)} / {len(catalog)}")
        st.dataframe(filtered)
        st.download_button("Download catalog csv", CATALOG_EXPORT_PATH.read_bytes(), file_name=CATALOG_EXPORT_PATH.name, mime="text/csv")

    with tab_setup:
        render_setup_tab()


if __name__ == "__main__":
    main()
