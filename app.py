from __future__ import annotations

import csv
import html
import io
import json
import os
import random
import re
import textwrap
import urllib.error
import urllib.parse
import urllib.request
import wave
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "true")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

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
    import speech_recognition as sr
except Exception:
    sr = None

try:
    import streamlit as st
except Exception:
    st = None

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import datasets as tv_datasets
    from torchvision import transforms
    from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s

    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    DataLoader = Any
    tv_datasets = None
    transforms = None
    EfficientNet_V2_S_Weights = None
    efficientnet_v2_s = None
    TORCH_AVAILABLE = False

try:
    import pyttsx3
except Exception:
    pyttsx3 = None


APP_TITLE = "Cognitive Emotion Intelligence and Adaptive Lifestyle System"
APP_SUBTITLE = (
    "Single-file PyTorch Streamlit project with EfficientNetV2, Grad-CAM, "
    "voice/text/face/emoji fusion, an animated mascot chatbot, digital twin logging, "
    "and free Hugging Face API support."
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
LOCAL_CHAT_MODEL = os.getenv("LOCAL_CHAT_MODEL", "google/flan-t5-small").strip() or "google/flan-t5-small"
TEXT_EMOTION_MODEL = (
    os.getenv("TEXT_EMOTION_MODEL", "j-hartmann/emotion-english-distilroberta-base").strip()
    or "j-hartmann/emotion-english-distilroberta-base"
)
TEXT_SENTIMENT_MODEL = (
    os.getenv("TEXT_SENTIMENT_MODEL", "distilbert-base-uncased-finetuned-sst-2-english").strip()
    or "distilbert-base-uncased-finetuned-sst-2-english"
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
    "gratitude": "happy",
    "surprise": "energetic",
    "excitement": "energetic",
    "excited": "energetic",
    "anger": "energetic",
    "angry": "energetic",
    "frustration": "energetic",
    "fear": "sad",
    "sadness": "sad",
    "sad": "sad",
    "loneliness": "sad",
    "neutral": "calm",
    "calm": "calm",
    "peace": "calm",
    "relaxed": "calm",
}
DEFAULT_FACE_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

FALLBACK_ACTIONS = {
    "happy": "Write one win, one gratitude, and one next goal while your mood is high.",
    "sad": "Pause, breathe slowly for two minutes, drink water, and contact one trusted person.",
    "calm": "Protect your balance with one low-noise focus task and a small break.",
    "energetic": "Channel the intensity into one useful task or a short movement break.",
}

MOOD_QUESTION_BANK = {
    "happy": [
        "What created this positive shift for you today?",
        "How can you repeat one small action that supported this mood?",
        "Who could you share this progress or gratitude with?",
    ],
    "sad": [
        "What feels heaviest right now?",
        "What helped even a little the last time you felt like this?",
        "What is one tiny step that would make the next hour easier?",
    ],
    "calm": [
        "What is helping you stay balanced right now?",
        "Which routine is protecting your focus today?",
        "What would you like to maintain before stress builds up?",
    ],
    "energetic": [
        "Is this energy helping progress or creating overload?",
        "Where can you channel this intensity most productively?",
        "What boundary would keep this energy healthy?",
    ],
}

TONE_GUIDES = {
    "Therapist": "Warm, grounded, reflective, structured, and non-diagnostic.",
    "Friendly": "Caring, natural, encouraging, and practical.",
    "Motivational": "Energetic, disciplined, action-focused, and respectful.",
}

DIRECT_VIDEO_URLS = {
    "happy": [
        "https://www.youtube.com/watch?v=d-diB65scQU",
        "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
        "https://www.youtube.com/watch?v=OPf0YbXqDm0",
        "https://www.youtube.com/watch?v=ZMO_XC9w7Lw",
    ],
    "sad": [
        "https://www.youtube.com/watch?v=inpok4MKVLM",
        "https://www.youtube.com/watch?v=1vx8iUvfyCY",
        "https://www.youtube.com/watch?v=6p_yaNFSYao",
        "https://www.youtube.com/watch?v=2XU0oxnq2qU",
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

SEARCH_QUERIES = {
    "happy": [
        "feel good playlist",
        "gratitude meditation",
        "upbeat study break songs",
        "confidence booster music",
        "joyful study playlist",
        "positive energy songs",
        "happy workday mix",
        "dance pop mood boost",
    ],
    "sad": [
        "comfort songs",
        "self compassion meditation",
        "gentle piano mood",
        "journaling prompts for grief",
        "calm emotional regulation",
        "soft healing songs",
        "mindfulness for low mood",
        "emotional reset playlist",
    ],
    "calm": [
        "lofi focus session",
        "rain sounds for studying",
        "box breathing guide",
        "ambient nature soundscape",
        "peaceful instrumental reading",
        "reading soundtrack",
        "meditation music",
        "deep focus",
    ],
    "energetic": [
        "power workout mix",
        "motivation speech short",
        "focus sprint music",
        "high energy coding playlist",
        "quick movement break",
        "pump up songs clean",
        "productive soundtrack",
        "hype coding mix",
    ],
}

DATASET_GUIDE = [
    {
        "dataset": "FER2025",
        "year": "2025",
        "type": "Face emotion",
        "best_use": "Primary facial expression training",
        "note": "Keep samples small on 8 GB RAM machines.",
    },
    {
        "dataset": "EmoNet-Face-Big",
        "year": "2025",
        "type": "Synthetic face emotion",
        "best_use": "Experimentation and future scope",
        "note": "Use fewer classes for demo speed.",
    },
    {
        "dataset": "MER2024",
        "year": "2024",
        "type": "Multimodal emotion",
        "best_use": "Research comparison",
        "note": "Often requires manual access approval.",
    },
    {
        "dataset": "MER2023",
        "year": "2023",
        "type": "Multimodal emotion",
        "best_use": "Literature review and extension",
        "note": "Use only with approved local access.",
    },
]


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


def utc_now() -> str:
    return datetime.utcnow().isoformat()


def slugify(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower())
    return text.strip("_") or "unknown"


def ensure_csv_file(path: Path, headers: list[str]) -> None:
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()


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


def safe_dataframe(data: Any) -> None:
    if st is None:
        return
    try:
        st.dataframe(data, width="stretch")
    except TypeError:
        st.dataframe(data, use_container_width=True)


def safe_image(image: Any, caption: str = "") -> None:
    if st is None:
        return
    try:
        st.image(image, caption=caption, width="stretch")
    except TypeError:
        st.image(image, caption=caption, use_container_width=True)


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
                    source="YouTube",
                    resource_type="youtube_video",
                    playable=True,
                    tags=f"{mood},video,direct",
                    offline_fallback=fallback,
                )
            )
        for index, query in enumerate(SEARCH_QUERIES[mood], start=1):
            encoded = urllib.parse.quote_plus(query)
            entries.extend(
                [
                    CatalogEntry(
                        id=f"{mood}_yt_search_{index}",
                        mood=mood,
                        title=f"{mood.title()} YouTube search: {query.title()}",
                        url=f"https://www.youtube.com/results?search_query={encoded}",
                        source="YouTube",
                        resource_type="youtube_search",
                        playable=False,
                        tags=f"{mood},youtube,{slugify(query)}",
                        offline_fallback=fallback,
                    ),
                    CatalogEntry(
                        id=f"{mood}_spotify_search_{index}",
                        mood=mood,
                        title=f"{mood.title()} Spotify search: {query.title()}",
                        url=f"https://open.spotify.com/search/{encoded}",
                        source="Spotify",
                        resource_type="spotify_search",
                        playable=False,
                        tags=f"{mood},spotify,{slugify(query)}",
                        offline_fallback=fallback,
                    ),
                    CatalogEntry(
                        id=f"{mood}_ytmusic_search_{index}",
                        mood=mood,
                        title=f"{mood.title()} YouTube Music search: {query.title()}",
                        url=f"https://music.youtube.com/search?q={encoded}",
                        source="YouTube Music",
                        resource_type="ytmusic_search",
                        playable=False,
                        tags=f"{mood},ytmusic,{slugify(query)}",
                        offline_fallback=fallback,
                    ),
                ]
            )
    return tuple(entries)


def catalog_rows() -> list[dict[str, Any]]:
    return [asdict(item) for item in build_resource_catalog()]


def export_resource_catalog_csv(path: Path = CATALOG_EXPORT_PATH) -> Path:
    rows = catalog_rows()
    write_csv_rows(path, list(rows[0].keys()), rows)
    return path


def get_recommender_stats() -> list[dict[str, str]]:
    ensure_runtime_files()
    return read_csv_rows(RECOMMENDER_STATS_PATH)


def upsert_recommender_feedback(item_id: str, feedback: str = "shown") -> None:
    rows = get_recommender_stats()
    fieldnames = ["ItemId", "Exposures", "Likes", "Skips", "LastShown", "LastFeedback"]
    target = next((row for row in rows if str(row.get("ItemId", "")) == str(item_id)), None)
    if target is None:
        target = {"ItemId": item_id, "Exposures": "0", "Likes": "0", "Skips": "0", "LastShown": "", "LastFeedback": ""}
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
    write_csv_rows(RECOMMENDER_STATS_PATH, fieldnames, rows)


def get_user_history(user: str) -> list[str]:
    history: list[str] = []
    for row in read_csv_rows(TWIN_LOG_PATH):
        if str(row.get("User", "")) == str(user):
            item_id = str(row.get("RecommendedId", "")).strip()
            if item_id:
                history.append(item_id)
    return history


def recommend_resource(user: str, mood: str, last_n: int = 5) -> dict[str, Any]:
    catalog = catalog_rows()
    candidates = [row for row in catalog if row["mood"] == mood] or catalog[:]
    recent_ids = set(get_user_history(user)[-last_n:])
    stats_map = {row["ItemId"]: row for row in get_recommender_stats()}
    scored: list[tuple[float, dict[str, Any]]] = []
    for item in candidates:
        stats = stats_map.get(item["id"], {})
        exposures = int(str(stats.get("Exposures", "0") or "0"))
        likes = int(str(stats.get("Likes", "0") or "0"))
        skips = int(str(stats.get("Skips", "0") or "0"))
        like_ratio = likes / max(exposures, 1)
        skip_ratio = skips / max(exposures, 1)
        novelty_bonus = 1.0 / (exposures + 1.0)
        playable_bonus = 0.12 if item.get("playable") else 0.0
        recent_penalty = 0.50 if item["id"] in recent_ids else 0.0
        score = 0.45 * like_ratio + 0.35 * novelty_bonus + playable_bonus - 0.20 * skip_ratio - recent_penalty
        score += random.uniform(0.0, 0.05)
        scored.append((score, item))
    pool = [(score, item) for score, item in scored if item["id"] not in recent_ids] or scored
    pool.sort(key=lambda pair: pair[0], reverse=True)
    top_pool = pool[: min(8, len(pool))]
    weights = np.array([max(score, 0.01) for score, _ in top_pool], dtype=np.float64)
    weights = weights / weights.sum()
    chosen = top_pool[int(np.random.choice(np.arange(len(top_pool)), p=weights))][1]
    upsert_recommender_feedback(chosen["id"], feedback="shown")
    return chosen


def update_last_feedback(user: str, item_id: str, feedback: str) -> None:
    rows = read_csv_rows(TWIN_LOG_PATH)
    if not rows:
        return
    for index in range(len(rows) - 1, -1, -1):
        row = rows[index]
        if str(row.get("User", "")) == str(user) and str(row.get("RecommendedId", "")) == str(item_id):
            rows[index]["Feedback"] = feedback
            write_csv_rows(TWIN_LOG_PATH, list(rows[0].keys()), rows)
            return


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
        return "No digital emotional twin history is available yet."
    recent = rows[-limit:]
    fused_moods = [str(row.get("FusedMood", "") or "calm") for row in recent]
    dominant_mood = Counter(fused_moods).most_common(1)[0][0] if fused_moods else "calm"
    recent_sequence = ", ".join(fused_moods[-5:]) if fused_moods else "none"
    last_title = next((row.get("RecommendedTitle", "") for row in reversed(recent) if row.get("RecommendedTitle")), "")
    last_feedback = next((row.get("Feedback", "") for row in reversed(recent) if row.get("Feedback")), "")
    return (
        f"Dominant recent mood: {dominant_mood}. Recent mood sequence: {recent_sequence or 'none'}. "
        f"Last recommendation: {last_title or 'not available'}. Latest feedback: {last_feedback or 'not recorded'}."
    )


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
    if not NLTK_AVAILABLE or wn is None:
        return tuple()
    ensure_nltk_resources()
    results: list[str] = []
    try:
        for synset in wn.synsets(word):
            for lemma in synset.lemmas():
                candidate = lemma.name().replace("_", " ").lower().strip()
                if candidate and candidate != word.lower() and candidate not in results:
                    results.append(candidate)
                if len(results) >= 8:
                    return tuple(results)
    except Exception:
        return tuple()
    return tuple(results)


INTENT_SEEDS = {
    "stress_relief": {"stress", "overwhelmed", "pressure", "tense", "burnout", "anxious"},
    "motivation": {"motivation", "discipline", "goal", "win", "progress", "improve"},
    "study_focus": {"study", "exam", "assignment", "focus", "college", "project"},
    "loneliness": {"alone", "lonely", "isolated", "miss", "empty"},
    "gratitude": {"grateful", "gratitude", "blessing", "thankful", "appreciate"},
    "self_reflection": {"reflect", "journal", "understand", "why", "meaning"},
    "confidence": {"confidence", "nervous", "presentation", "interview", "fear"},
}


def infer_user_intent(text: str) -> dict[str, Any]:
    tokens = set(tokenize_words(text))
    scores: dict[str, int] = {}
    matched_terms: dict[str, list[str]] = {}
    for intent_name, seeds in INTENT_SEEDS.items():
        expanded = set(seeds)
        for seed in list(seeds):
            expanded.update(tokenize_words(" ".join(synonyms_for_word(seed))))
        matches = sorted(token for token in tokens if token in expanded)
        scores[intent_name] = len(matches)
        matched_terms[intent_name] = matches
    best_intent = max(scores.items(), key=lambda item: item[1])[0] if scores else "general_support"
    if scores.get(best_intent, 0) == 0:
        best_intent = "general_support"
    preview = sorted({syn for token in list(tokens)[:6] for syn in synonyms_for_word(token)[:2]})[:8]
    return {"intent": best_intent, "matched_terms": matched_terms.get(best_intent, []), "synonyms_preview": preview}


def normalize_label_to_mood(label: str) -> str:
    token = slugify(label).replace("_", " ")
    for emotion, mood in EMOTION_TO_MOOD.items():
        if emotion in token:
            return mood
    return "calm"


def simple_text_mood(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ["sad", "cry", "lonely", "down", "grief", "hurt"]):
        return "sad"
    if any(token in lowered for token in ["happy", "joy", "great", "awesome", "love", "good"]):
        return "happy"
    if any(token in lowered for token in ["angry", "mad", "stressed", "frustrated", "furious", "rage"]):
        return "energetic"
    return "calm"


def get_transformers_pipeline(task: str, model_name: str):
    try:
        from transformers import pipeline
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        return pipeline(task, model=model_name, device=-1)
    except Exception:
        return None


@lru_cache(maxsize=1)
def get_text_emotion_pipeline():
    return get_transformers_pipeline("text-classification", TEXT_EMOTION_MODEL)


@lru_cache(maxsize=1)
def get_text_sentiment_pipeline():
    return get_transformers_pipeline("sentiment-analysis", TEXT_SENTIMENT_MODEL)


@lru_cache(maxsize=1)
def get_local_chat_pipeline():
    return get_transformers_pipeline("text2text-generation", LOCAL_CHAT_MODEL)


def analyze_text_emotion(text: str, use_local_transformers: bool = False) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {
            "text": "",
            "sentiment_label": "NEUTRAL",
            "sentiment_score": 0.50,
            "emotion_label": "neutral",
            "emotion_score": 0.50,
            "mood": "calm",
            "confidence": 0.50,
            "intent": "general_support",
            "matched_terms": [],
            "synonyms_preview": [],
            "mood_scores": {mood: 0.0 for mood in MOOD_CHOICES},
            "method": "empty_text",
        }

    intent = infer_user_intent(text)
    sentiment_label = "NEUTRAL"
    sentiment_score = 0.50
    emotion_label = simple_text_mood(text)
    emotion_score = 0.55
    mood_scores = {mood: 0.0 for mood in MOOD_CHOICES}
    method = "local_heuristic"

    if use_local_transformers:
        sentiment_pipeline = get_text_sentiment_pipeline()
        if sentiment_pipeline is not None:
            try:
                result = sentiment_pipeline(text[:512], truncation=True)[0]
                sentiment_label = str(result.get("label", sentiment_label)).upper()
                sentiment_score = float(result.get("score", sentiment_score))
                method = "local_transformers"
            except Exception:
                pass
        emotion_pipeline = get_text_emotion_pipeline()
        if emotion_pipeline is not None:
            try:
                result = emotion_pipeline(text[:512], truncation=True, top_k=None)
                if result and isinstance(result[0], list):
                    result = result[0]
                if result:
                    top_item = max(result, key=lambda item: float(item.get("score", 0.0)))
                    emotion_label = str(top_item.get("label", emotion_label))
                    emotion_score = float(top_item.get("score", emotion_score))
                    for item in result:
                        mood_scores[normalize_label_to_mood(str(item.get("label", "neutral")))] += float(
                            item.get("score", 0.0)
                        )
                    method = "local_transformers"
            except Exception:
                pass

    if max(mood_scores.values()) <= 0:
        mood_scores[simple_text_mood(text)] = 1.0
    if sentiment_label.startswith("NEG"):
        mood_scores["sad"] += 0.15
    if sentiment_label.startswith("POS"):
        mood_scores["happy"] += 0.15

    mood = max(mood_scores.items(), key=lambda item: item[1])[0]
    confidence = float(max(max(mood_scores.values()), emotion_score, sentiment_score))
    return {
        "text": text,
        "sentiment_label": sentiment_label,
        "sentiment_score": sentiment_score,
        "emotion_label": emotion_label,
        "emotion_score": emotion_score,
        "mood": mood,
        "confidence": confidence,
        "intent": intent["intent"],
        "matched_terms": intent["matched_terms"],
        "synonyms_preview": intent["synonyms_preview"],
        "mood_scores": mood_scores,
        "method": method,
    }


def transcribe_audio_bytes(audio_bytes: bytes | None) -> tuple[str, str]:
    if not audio_bytes:
        return "", "no_audio"
    if sr is None:
        return "", "speechrecognition_not_installed"
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio = recognizer.record(source)
        return recognizer.recognize_google(audio).strip(), "google_web_speech"
    except Exception:
        return "", "transcription_unavailable"


def estimate_wav_energy(audio_bytes: bytes | None) -> float:
    if not audio_bytes:
        return 0.0
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            sample_width = wav_file.getsampwidth()
            channels = wav_file.getnchannels()
        if sample_width == 1:
            data = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0
            max_abs = 128.0
        elif sample_width == 2:
            data = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
            max_abs = 32768.0
        elif sample_width == 4:
            data = np.frombuffer(frames, dtype=np.int32).astype(np.float32)
            max_abs = float(2**31)
        else:
            return 0.0
        if channels > 1:
            data = data.reshape(-1, channels).mean(axis=1)
        return float(np.mean(np.abs(data)) / max_abs)
    except Exception:
        return 0.0


def analyze_voice_emotion(audio_bytes: bytes | None, username: str, use_local_transformers: bool = False) -> dict[str, Any]:
    transcript, method = transcribe_audio_bytes(audio_bytes)
    text_result = analyze_text_emotion(transcript, use_local_transformers=use_local_transformers)
    energy = estimate_wav_energy(audio_bytes)
    mood_scores = dict(text_result["mood_scores"])
    if energy > 0.14:
        mood_scores["energetic"] += 0.15
    if energy < 0.04:
        mood_scores["calm"] += 0.08
    if text_result["sentiment_label"].startswith("NEG") and energy < 0.05:
        mood_scores["sad"] += 0.10
    voice_mood = max(mood_scores.items(), key=lambda item: item[1])[0]
    return {
        "transcript": transcript,
        "transcription_method": method,
        "energy": energy,
        "mood": voice_mood,
        "confidence": float(max(mood_scores.values()) if mood_scores else 0.0),
        "spoken_name_detected": bool(username and transcript and username.lower() in transcript.lower()),
        "text_result": text_result,
    }


@lru_cache(maxsize=1)
def get_face_cascade():
    if cv2 is None:
        return None
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    return cascade if not cascade.empty() else None


@lru_cache(maxsize=1)
def get_smile_cascade():
    if cv2 is None:
        return None
    cascade_path = cv2.data.haarcascades + "haarcascade_smile.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    return cascade if not cascade.empty() else None


def decode_image_bytes(image_bytes: bytes | None) -> np.ndarray | None:
    if not image_bytes:
        return None
    if cv2 is None:
        return None
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


def detect_largest_face(image_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    cascade = get_face_cascade()
    if cascade is None:
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
    smile_cascade = get_smile_cascade()
    smile_found = False
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


def imagenet_mean_std() -> tuple[list[float], list[float]]:
    if not TORCH_AVAILABLE or EfficientNet_V2_S_Weights is None:
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    weights = EfficientNet_V2_S_Weights.DEFAULT
    return list(weights.transforms().mean), list(weights.transforms().std)


def get_inference_transform():
    if not TORCH_AVAILABLE or transforms is None:
        raise RuntimeError("Install torch and torchvision before using EfficientNetV2 inference.")
    mean, std = imagenet_mean_std()
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def build_efficientnet_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    if not TORCH_AVAILABLE or efficientnet_v2_s is None or nn is None:
        raise RuntimeError("Install torch and torchvision before building the EfficientNetV2 model.")
    try:
        weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        model = efficientnet_v2_s(weights=weights)
    except Exception:
        model = efficientnet_v2_s(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(p=0.30), nn.Linear(in_features, num_classes))
    return model


def save_model_metadata(class_names: list[str], extra: dict[str, Any] | None = None) -> None:
    payload = {
        "class_names": class_names,
        "architecture": "efficientnet_v2_s",
        "saved_at_utc": utc_now(),
        "model_path": str(MODEL_PATH),
    }
    if extra:
        payload.update(extra)
    MODEL_META_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_model_checkpoint(model: nn.Module, class_names: list[str], extra: dict[str, Any] | None = None) -> None:
    if not TORCH_AVAILABLE or torch is None:
        raise RuntimeError("Install torch before saving model checkpoints.")
    payload = {
        "architecture": "efficientnet_v2_s",
        "class_names": class_names,
        "model_state_dict": model.state_dict(),
        "saved_at_utc": utc_now(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, MODEL_PATH)
    save_model_metadata(class_names, extra=extra or {})


@lru_cache(maxsize=1)
def load_saved_model() -> tuple[nn.Module | None, dict[str, Any]]:
    if not TORCH_AVAILABLE or torch is None or not MODEL_PATH.exists():
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


def get_last_conv_module(model: nn.Module):
    last_name = None
    last_module = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_name = name
            last_module = module
    return last_name, last_module


def compute_gradcam(model: nn.Module, image_tensor: torch.Tensor, target_index: int | None = None) -> np.ndarray | None:
    if not TORCH_AVAILABLE or torch is None or nn is None:
        return None
    _, target_module = get_last_conv_module(model)
    if target_module is None:
        return None
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def forward_hook(_module, _inputs, output):
        activations.append(output.detach())

    def backward_hook(_module, _grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    handle_forward = target_module.register_forward_hook(forward_hook)
    handle_backward = target_module.register_full_backward_hook(backward_hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(image_tensor)
        if target_index is None:
            target_index = int(torch.argmax(logits, dim=1).item())
        logits[:, target_index].sum().backward()
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
        handle_forward.remove()
        handle_backward.remove()


def overlay_heatmap(face_bgr: np.ndarray, heatmap: np.ndarray | None) -> np.ndarray | None:
    if heatmap is None or cv2 is None:
        return None
    heatmap = cv2.resize(heatmap, (face_bgr.shape[1], face_bgr.shape[0]))
    colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
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
    if image_bgr is None:
        return {
            "label": "neutral",
            "mood": "calm",
            "confidence": 0.0,
            "method": "invalid_image",
            "overlay_rgb": None,
            "face_found": False,
        }
    face_box = detect_largest_face(image_bgr)
    display_bgr = image_bgr.copy()
    if face_box is None:
        face_crop = image_bgr
        x, y, w, h = 0, 0, image_bgr.shape[1], image_bgr.shape[0]
    else:
        x, y, w, h = map(int, face_box)
        face_crop = image_bgr[y : y + h, x : x + w]
        cv2.rectangle(display_bgr, (x, y), (x + w, y + h), (70, 255, 140), 2)

    model, metadata = load_saved_model()
    if model is not None:
        try:
            transform = get_inference_transform()
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            tensor = transform(face_rgb).unsqueeze(0)
            logits = model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
            class_names = metadata.get("class_names") or DEFAULT_FACE_CLASSES
            predicted_index = int(np.argmax(probabilities))
            label = class_names[predicted_index] if predicted_index < len(class_names) else f"class_{predicted_index}"
            heatmap = compute_gradcam(model, tensor, predicted_index)
            overlay = overlay_heatmap(face_crop, heatmap)
            if overlay is not None and face_box is not None:
                display_bgr[y : y + h, x : x + w] = cv2.resize(overlay, (w, h))
                cv2.rectangle(display_bgr, (x, y), (x + w, y + h), (70, 255, 140), 2)
            elif overlay is not None:
                display_bgr = overlay
            return {
                "label": str(label),
                "mood": normalize_label_to_mood(str(label)),
                "confidence": float(probabilities[predicted_index]),
                "method": "trained_efficientnet_v2_s",
                "overlay_rgb": cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB),
                "face_found": face_box is not None,
            }
        except Exception:
            pass

    label, confidence = heuristic_face_label(face_crop)
    return {
        "label": label,
        "mood": normalize_label_to_mood(label),
        "confidence": confidence,
        "method": "opencv_heuristic",
        "overlay_rgb": cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB),
        "face_found": face_box is not None,
    }


def dataset_has_images(split_dir: Path) -> bool:
    if not split_dir.exists():
        return False
    return any(path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS for path in split_dir.rglob("*"))


def dataset_directory_summary(root: Path = DATASET_DIR) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name in ["train", "val", "test"]:
        split_dir = root / split_name
        if not split_dir.exists():
            continue
        for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            count = sum(1 for item in class_dir.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS)
            rows.append({"split": split_name, "class_name": class_dir.name, "count": count})
    return rows


def build_dataloaders(dataset_root: Path, batch_size: int = 8):
    if not TORCH_AVAILABLE or transforms is None or tv_datasets is None or DataLoader is Any:
        raise RuntimeError("Install torch and torchvision before training.")
    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"
    test_dir = dataset_root / "test"
    if not dataset_has_images(train_dir):
        raise RuntimeError("No training images were found under dataset/train.")
    mean, std = imagenet_mean_std()
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    eval_transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    )
    train_ds = tv_datasets.ImageFolder(train_dir, transform=train_transform)
    class_names = list(train_ds.classes)
    val_ds = tv_datasets.ImageFolder(val_dir, transform=eval_transform) if dataset_has_images(val_dir) else None
    test_ds = tv_datasets.ImageFolder(test_dir, transform=eval_transform) if dataset_has_images(test_dir) else None
    for split_name, dataset in [("validation", val_ds), ("test", test_ds)]:
        if dataset is not None and list(dataset.classes) != class_names:
            raise RuntimeError(f"{split_name.title()} classes must match training classes exactly.")
    loader_kwargs = {"batch_size": batch_size, "num_workers": 0, "pin_memory": torch.cuda.is_available()}
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs) if val_ds is not None else None
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs) if test_ds is not None else None
    return train_loader, val_loader, test_loader, class_names


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
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
        total_samples += int(labels.size(0))
    return total_loss / max(total_samples, 1), total_correct / max(total_samples, 1)


def evaluate_loader_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += float(loss.item()) * labels.size(0)
            total_correct += int((torch.argmax(logits, dim=1) == labels).sum().item())
            total_samples += int(labels.size(0))
    return total_loss / max(total_samples, 1), total_correct / max(total_samples, 1)


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred):
        if 0 <= truth < num_classes and 0 <= pred < num_classes:
            matrix[truth, pred] += 1
    return matrix


def classification_report_np(matrix: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    total = int(matrix.sum())
    report: dict[str, Any] = {}
    precisions: list[float] = []
    recalls: list[float] = []
    f1_scores: list[float] = []
    weights: list[int] = []
    for index, class_name in enumerate(class_names):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum() - tp)
        fn = int(matrix[index, :].sum() - tp)
        support = int(matrix[index, :].sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        report[class_name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1-score": round(f1, 4),
            "support": support,
        }
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        weights.append(support)
    report["accuracy"] = round(float(np.trace(matrix) / max(total, 1)), 4)
    report["macro avg"] = {
        "precision": round(float(np.mean(precisions)), 4) if precisions else 0.0,
        "recall": round(float(np.mean(recalls)), 4) if recalls else 0.0,
        "f1-score": round(float(np.mean(f1_scores)), 4) if f1_scores else 0.0,
        "support": total,
    }
    report["weighted avg"] = {
        "precision": round(float(np.average(precisions, weights=weights)), 4) if sum(weights) > 0 else 0.0,
        "recall": round(float(np.average(recalls, weights=weights)), 4) if sum(weights) > 0 else 0.0,
        "f1-score": round(float(np.average(f1_scores, weights=weights)), 4) if sum(weights) > 0 else 0.0,
        "support": total,
    }
    return report


def average_ranks(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(scores)
    ranks = np.zeros_like(scores, dtype=np.float64)
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start
        while end + 1 < len(scores) and sorted_scores[end + 1] == sorted_scores[start]:
            end += 1
        average_rank = (start + end + 2) / 2.0
        ranks[order[start : end + 1]] = average_rank
        start = end + 1
    return ranks


def binary_auc(y_true_binary: np.ndarray, y_scores: np.ndarray) -> float | None:
    positives = int(y_true_binary.sum())
    negatives = int(len(y_true_binary) - positives)
    if positives == 0 or negatives == 0:
        return None
    ranks = average_ranks(y_scores.astype(np.float64))
    positive_rank_sum = float(ranks[y_true_binary == 1].sum())
    return float((positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def multiclass_roc_auc_macro(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int) -> float | None:
    aucs: list[float] = []
    for class_index in range(num_classes):
        auc = binary_auc((y_true == class_index).astype(np.int32), y_prob[:, class_index])
        if auc is not None:
            aucs.append(auc)
    return float(np.mean(aucs)) if aucs else None


def evaluate_model_on_loader(model: nn.Module, loader: DataLoader | None, class_names: list[str], device) -> dict[str, Any]:
    if loader is None:
        return {"message": "No validation/test split available for evaluation."}
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[list[float]] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            y_prob.extend(probs.tolist())
            y_pred.extend(np.argmax(probs, axis=1).tolist())
            y_true.extend(labels.numpy().tolist())
    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_pred_np = np.asarray(y_pred, dtype=np.int64)
    y_prob_np = np.asarray(y_prob, dtype=np.float64)
    matrix = confusion_matrix_np(y_true_np, y_pred_np, len(class_names))
    roc_auc_macro = multiclass_roc_auc_macro(y_true_np, y_prob_np, len(class_names))
    return {
        "report": classification_report_np(matrix, class_names),
        "confusion_matrix": matrix.tolist(),
        "roc_auc_macro": round(roc_auc_macro, 4) if roc_auc_macro is not None else None,
    }


def train_emotion_model(dataset_root: Path, batch_size: int, epochs: int, learning_rate: float, freeze_backbone: bool, progress_callback=None):
    train_loader, val_loader, test_loader, class_names = build_dataloaders(dataset_root, batch_size=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_efficientnet_model(len(class_names), pretrained=True)
    if freeze_backbone:
        for parameter in model.features.parameters():
            parameter.requires_grad = False
    model.to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = (0.0, 0.0)
        if val_loader is not None:
            val_loss, val_acc = evaluate_loader_loss(model, val_loader, criterion, device)
        row = {
            "epoch": float(epoch),
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 4),
        }
        history.append(row)
        if progress_callback is not None:
            progress_callback(epoch / epochs, row)
    save_model_checkpoint(
        model.cpu(),
        class_names,
        extra={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "freeze_backbone": freeze_backbone,
            "dataset_root": str(dataset_root),
        },
    )
    load_saved_model.cache_clear()
    evaluation = evaluate_model_on_loader(model.to(device), test_loader or val_loader, class_names, device)
    return model.cpu(), history, evaluation, class_names


def emotion_fusion(face_result: dict[str, Any], voice_result: dict[str, Any], text_result: dict[str, Any], emoji_mood: str):
    scores = {mood: 0.0 for mood in MOOD_CHOICES}
    weighted_inputs = [
        (face_result.get("mood", "calm"), float(face_result.get("confidence") or 0.45), 0.35),
        (voice_result.get("mood", "calm"), float(voice_result.get("confidence") or 0.45), 0.25),
        (text_result.get("mood", "calm"), float(text_result.get("confidence") or 0.45), 0.25),
        (emoji_mood, 1.0, 0.15),
    ]
    for mood, confidence, weight in weighted_inputs:
        if mood in scores:
            scores[mood] += weight * max(confidence, 0.30)
    fused_mood = max(scores.items(), key=lambda item: item[1])[0]
    total = sum(scores.values()) or 1.0
    return fused_mood, scores, float(scores[fused_mood] / total)


def ethical_monitor_report(face_result: dict[str, Any], voice_result: dict[str, Any], text_result: dict[str, Any], fusion_scores: dict[str, float]):
    warnings: list[str] = []
    modalities_used = 0
    if face_result.get("method") != "no_image":
        modalities_used += 1
    if voice_result.get("transcript") or voice_result.get("transcription_method") != "no_audio":
        modalities_used += 1
    if text_result.get("text"):
        modalities_used += 1
    if face_result.get("method") == "opencv_heuristic":
        warnings.append("Facial analysis is using a safe heuristic until a .pth model is trained.")
    if modalities_used < 2:
        warnings.append("Fusion confidence improves when at least two modalities are present.")
    warnings.append("This is supportive guidance, not clinical diagnosis.")
    dominant_score = max(fusion_scores.values()) if fusion_scores else 0.0
    confidence_band = "High" if dominant_score >= 0.45 else "Medium" if dominant_score >= 0.28 else "Low"
    return {"modalities_used": modalities_used, "confidence_band": confidence_band, "warnings": warnings}


def get_chat_provider_name(use_local_transformers: bool) -> str:
    if HF_TOKEN:
        return f"Hugging Face Inference API ({HF_CHAT_MODEL})"
    if use_local_transformers:
        return f"Local transformers pipeline ({LOCAL_CHAT_MODEL})"
    return "Rule-based local coach"


def build_chat_system_prompt(username: str, mood: str, tone_mode: str, twin_summary: str, intent: str, analysis_snapshot: str) -> str:
    mood = mood if mood in MOOD_CHOICES else "calm"
    return textwrap.dedent(
        f"""
        You are an emotionally intelligent therapeutic mascot chatbot.
        User name: {username}
        Current fused mood: {mood}
        Conversation mode: {tone_mode}
        Tone instructions: {TONE_GUIDES.get(tone_mode, TONE_GUIDES["Therapist"])}
        Detected intent: {intent}
        Digital emotional twin summary: {twin_summary}
        Emotion snapshot: {analysis_snapshot}

        Requirements:
        1. Be warm, specific, and practical.
        2. Avoid generic FAQ answers.
        3. Give one short supportive reflection.
        4. Ask exactly three meaningful questions.
        5. Suggest one next action for the next 10-20 minutes.
        6. Do not diagnose disease or claim certainty.
        """
    ).strip()


def huggingface_free_chat(system_prompt: str, user_prompt: str) -> str | None:
    if not HF_TOKEN:
        return None
    payload = {
        "inputs": f"Instruction:\n{system_prompt}\n\nUser:\n{user_prompt}\n\nAssistant:\n",
        "parameters": {"max_new_tokens": 220, "temperature": 0.7, "top_p": 0.9, "return_full_text": False},
        "options": {"wait_for_model": True},
    }
    request = urllib.request.Request(
        url=f"https://api-inference.huggingface.co/models/{HF_CHAT_MODEL}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            body = response.read().decode("utf-8")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, Exception):
        return None
    try:
        data = json.loads(body)
    except Exception:
        return body.strip() or None
    if isinstance(data, dict) and isinstance(data.get("generated_text"), str):
        return data["generated_text"].strip()
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict) and isinstance(first.get("generated_text"), str):
            return first["generated_text"].strip()
        if isinstance(first, str):
            return first.strip()
    return None


def local_transformers_chat(system_prompt: str, user_prompt: str, use_local_transformers: bool) -> str | None:
    if not use_local_transformers:
        return None
    generator = get_local_chat_pipeline()
    if generator is None:
        return None
    prompt = f"{system_prompt}\n\nUser message: {user_prompt}\n\nWrite a concise response."
    try:
        result = generator(prompt, max_new_tokens=220, do_sample=True, temperature=0.7)[0]
        return str(result.get("generated_text", "")).strip() or None
    except Exception:
        return None


def local_rule_based_response(question: str, mood: str, tone_mode: str, twin_summary: str, intent: str) -> str:
    questions = MOOD_QUESTION_BANK.get(mood, MOOD_QUESTION_BANK["calm"])
    prefix = {
        "Therapist": "I hear you, and I want to respond carefully.",
        "Friendly": "I am with you, and we can keep this practical.",
        "Motivational": "You can turn this moment into one useful step.",
    }.get(tone_mode, "I am here with you.")
    return textwrap.dedent(
        f"""
        {prefix}

        Reflection:
        - Your message connects with a {mood} emotional state and the intent area of {intent}.

        Important questions:
        1. {questions[0]}
        2. {questions[1]}
        3. {questions[2]}

        One next action:
        - {FALLBACK_ACTIONS.get(mood, FALLBACK_ACTIONS["calm"])}

        Digital twin context:
        - {twin_summary}

        Your message:
        "{question}"
        """
    ).strip()


def generate_chat_response(username: str, question: str, current_mood: str, tone_mode: str, analysis: dict[str, Any] | None, use_local_transformers: bool):
    text_result = analyze_text_emotion(question, use_local_transformers=use_local_transformers)
    active_mood = current_mood if current_mood in MOOD_CHOICES else text_result["mood"]
    twin_summary = get_user_twin_summary(username)
    if analysis:
        analysis_snapshot = (
            f"Face={analysis['face_result']['mood']}, Voice={analysis['voice_result']['mood']}, "
            f"Text={analysis['text_result']['mood']}, Emoji={analysis['emoji_mood']}, "
            f"Fused={analysis['fused_mood']}, StateScore={analysis['state_score']:.2f}"
        )
    else:
        analysis_snapshot = "No multimodal analysis available yet."
    system_prompt = build_chat_system_prompt(username, active_mood, tone_mode, twin_summary, text_result["intent"], analysis_snapshot)
    user_prompt = (
        f"User question: {question}\n"
        f"Sentiment: {text_result['sentiment_label']} ({text_result['sentiment_score']:.2f})\n"
        f"Emotion: {text_result['emotion_label']} ({text_result['emotion_score']:.2f})\n"
        f"Intent: {text_result['intent']}\n"
        f"Synonym hints: {', '.join(text_result['synonyms_preview']) or 'none'}"
    )
    response = huggingface_free_chat(system_prompt, user_prompt)
    if response:
        return response, get_chat_provider_name(use_local_transformers)
    response = local_transformers_chat(system_prompt, user_prompt, use_local_transformers=use_local_transformers)
    if response:
        return response, get_chat_provider_name(use_local_transformers)
    return local_rule_based_response(question, active_mood, tone_mode, twin_summary, text_result["intent"]), get_chat_provider_name(False)


def speak_text_locally(text: str) -> str:
    if pyttsx3 is None:
        return "pyttsx3 is not installed, so local text-to-speech is unavailable."
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 170)
        engine.say(text)
        engine.runAndWait()
        return "Reply spoken on the local machine."
    except Exception:
        return "Local text-to-speech could not be started on this machine."


def render_mascot(mood: str, username: str, tone_mode: str, subtitle: str) -> None:
    mood = mood if mood in MOOD_CHOICES else "calm"
    palette = {
        "happy": {"bg": "#fff9db", "face": "#ffd866", "accent": "#ff922b", "mouth": "smile", "eye": "open"},
        "sad": {"bg": "#eef2ff", "face": "#a5b4fc", "accent": "#4c6ef5", "mouth": "sad", "eye": "soft"},
        "calm": {"bg": "#ecfeff", "face": "#99f6e4", "accent": "#0f766e", "mouth": "calm", "eye": "soft"},
        "energetic": {"bg": "#fff1f2", "face": "#fda4af", "accent": "#e11d48", "mouth": "grin", "eye": "sharp"},
    }[mood]
    speech = html.escape(subtitle[:180] if subtitle else f"{username}, I am tuned to your {mood} state.")
    mouth_html = {
        "smile": '<div class="cei-mouth cei-mouth-smile"></div>',
        "sad": '<div class="cei-mouth cei-mouth-sad"></div>',
        "calm": '<div class="cei-mouth cei-mouth-calm"></div>',
        "grin": '<div class="cei-mouth cei-mouth-grin"></div>',
    }[palette["mouth"]]
    eye_class = {"open": "cei-eye-open", "soft": "cei-eye-soft", "sharp": "cei-eye-sharp"}[palette["eye"]]
    st.markdown(
        f"""
        <style>
        .cei-card {{ background:{palette["bg"]}; border-radius:22px; padding:18px; display:flex; flex-direction:column; gap:14px; box-shadow:0 12px 24px rgba(15,23,42,.08); }}
        .cei-bubble {{ background:white; border-radius:18px; padding:12px 14px; color:#0f172a; font-size:.95rem; line-height:1.5; box-shadow:0 6px 14px rgba(15,23,42,.08); }}
        .cei-stage {{ position:relative; display:flex; align-items:center; justify-content:center; min-height:250px; overflow:hidden; }}
        .cei-glow {{ position:absolute; width:220px; height:220px; background:radial-gradient(circle,{palette["accent"]}33 0%,transparent 70%); animation:ceiPulse 2.6s ease-in-out infinite; }}
        .cei-avatar {{ position:relative; width:180px; height:180px; border-radius:999px; background:{palette["face"]}; animation:ceiFloat 2.4s ease-in-out infinite; box-shadow:0 18px 28px rgba(15,23,42,.15); }}
        .cei-eye {{ position:absolute; top:68px; width:18px; height:18px; border-radius:999px; background:#111827; animation:ceiBlink 4.2s infinite; }}
        .cei-eye-left {{ left:48px; }} .cei-eye-right {{ right:48px; }} .cei-eye-soft {{ height:12px; top:71px; }} .cei-eye-sharp {{ transform:skewX(-12deg); }}
        .cei-mouth {{ position:absolute; left:50%; transform:translateX(-50%); bottom:44px; }}
        .cei-mouth-smile {{ width:56px; height:28px; border-bottom:6px solid #7f1d1d; border-radius:0 0 70px 70px; }}
        .cei-mouth-sad {{ width:56px; height:28px; border-top:6px solid #1e293b; border-radius:70px 70px 0 0; }}
        .cei-mouth-calm {{ width:44px; height:0; border-top:5px solid #0f172a; border-radius:20px; }}
        .cei-mouth-grin {{ width:64px; height:16px; border-bottom:6px solid #7f1d1d; border-radius:0 0 80px 80px; }}
        .cei-badge {{ display:inline-flex; background:white; width:fit-content; padding:8px 12px; border-radius:999px; color:#0f172a; font-weight:600; box-shadow:0 6px 14px rgba(15,23,42,.08); }}
        @keyframes ceiFloat {{ 0%,100% {{ transform:translateY(0); }} 50% {{ transform:translateY(-8px); }} }}
        @keyframes ceiBlink {{ 0%,92%,100% {{ transform:scaleY(1); }} 94%,96% {{ transform:scaleY(.1); }} }}
        @keyframes ceiPulse {{ 0%,100% {{ transform:scale(.92); opacity:.7; }} 50% {{ transform:scale(1.05); opacity:1; }} }}
        </style>
        <div class="cei-card">
            <div class="cei-bubble">{speech}</div>
            <div class="cei-stage">
                <div class="cei-glow"></div>
                <div class="cei-avatar">
                    <div class="cei-eye cei-eye-left {eye_class}"></div>
                    <div class="cei-eye cei-eye-right {eye_class}"></div>
                    {mouth_html}
                </div>
            </div>
            <div class="cei-badge">Mascot mode: {html.escape(tone_mode)} | Emotional state: {html.escape(mood.title())}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_mood_scores(scores: dict[str, float]) -> None:
    total = sum(scores.values()) or 1.0
    for mood in MOOD_CHOICES:
        value = float(scores.get(mood, 0.0) / total)
        st.write(f"{mood.title()}: {value:.2%}")
        st.progress(min(max(value, 0.0), 1.0))


def render_recommendation_card(item: dict[str, Any]) -> None:
    st.subheader(item["title"])
    st.caption(f"{item['mood'].title()} | {item['source']} | {item['resource_type']}")
    if item.get("playable") and "youtube.com/watch" in item.get("url", ""):
        st.video(item["url"])
    else:
        st.markdown(f"[Open resource]({item['url']})")
    st.caption(item.get("offline_fallback", ""))


def live_analysis(user: str, emoji: str, image_bytes: bytes | None, audio_bytes: bytes | None, text_input: str, tone_mode: str, use_local_transformers: bool):
    face_result = predict_face_emotion(image_bytes)
    voice_result = analyze_voice_emotion(audio_bytes, username=user, use_local_transformers=use_local_transformers)
    text_result = analyze_text_emotion(text_input, use_local_transformers=use_local_transformers)
    emoji_mood = EMOJI_MAP.get(emoji, "calm")
    fused_mood, fusion_scores, state_score = emotion_fusion(face_result, voice_result, text_result, emoji_mood)
    recommendation = recommend_resource(user, fused_mood, last_n=5)
    ethical_report = ethical_monitor_report(face_result, voice_result, text_result, fusion_scores)
    result = {
        "face_result": face_result,
        "voice_result": voice_result,
        "text_result": text_result,
        "emoji_mood": emoji_mood,
        "fused_mood": fused_mood,
        "fusion_scores": fusion_scores,
        "state_score": state_score,
        "recommendation": recommendation,
        "ethical_report": ethical_report,
        "tone_mode": tone_mode,
    }
    log_interaction(
        user=user,
        face_label=str(face_result.get("label", "neutral")),
        face_mood=str(face_result.get("mood", "calm")),
        voice_mood=str(voice_result.get("mood", "calm")),
        text_mood=str(text_result.get("mood", "calm")),
        emoji_mood=emoji_mood,
        fused_mood=fused_mood,
        state_score=state_score,
        tone_mode=tone_mode,
        recommendation=recommendation,
    )
    return result


def render_setup_tab(use_local_transformers: bool) -> None:
    st.subheader("Step-by-step VS Code execution guide")
    st.markdown(
        textwrap.dedent(
            """
            1. Create one project folder in VS Code.
            2. Put only these required files inside it: `app.py` and `requirements.txt`.
            3. Optional generated folders/files are created automatically: `models/`, `dataset/`, `cei_twin_log.csv`, `recommender_stats.csv`, and `resource_catalog.csv`.
            4. Create and activate a virtual environment in PowerShell:
               ```powershell
               python -m venv .venv
               .\\.venv\\Scripts\\Activate.ps1
               ```
            5. Install dependencies:
               ```powershell
               python -m pip install --upgrade pip
               pip install -r requirements.txt
               ```
            6. Optional free Hugging Face API:
               - Create a free account at huggingface.co
               - Open Settings -> Access Tokens
               - Create a read token
               - Set it in PowerShell:
               ```powershell
               $env:HF_TOKEN="your_hugging_face_token"
               ```
            7. Run:
               ```powershell
               streamlit run app.py
               ```
            """
        )
    )
    st.markdown("### requirements.txt")
    st.code(
        "\n".join(
            [
                "streamlit",
                "numpy",
                "opencv-python",
                "torch",
                "torchvision",
                "transformers",
                "nltk",
                "SpeechRecognition",
                "pyttsx3",
            ]
        ),
        language="text",
    )
    st.markdown("### Dataset folder structure for EfficientNetV2 training")
    st.code(
        textwrap.dedent(
            """
            dataset/
            |-- train/
            |   |-- happy/
            |   |-- sad/
            |   |-- angry/
            |   `-- neutral/
            |-- val/
            |   |-- happy/
            |   |-- sad/
            |   |-- angry/
            |   `-- neutral/
            `-- test/
                |-- happy/
                |-- sad/
                |-- angry/
                `-- neutral/
            """
        ).strip(),
        language="text",
    )
    st.markdown("### Why Hugging Face is the free API option")
    st.markdown(
        f"""
        - Hugging Face supports free account creation and free read tokens.
        - The app can use Hugging Face Inference API through `HF_TOKEN`.
        - Local transformers are lazy-loaded only when enabled, preventing startup warning spam.
        - OpenAI and paid APIs are not required.
        - Active provider mode: **{get_chat_provider_name(use_local_transformers)}**
        """
    )
    st.markdown("### Free APK conversion trick")
    st.info(
        "Run or host the Streamlit app, open the URL in Android Chrome, then choose Add to Home Screen. "
        "For a real APK later, wrap the same URL in a free Android Studio WebView project."
    )
    st.markdown("### Viva-ready project highlights")
    st.markdown(
        """
        - PyTorch-only implementation with EfficientNetV2-S, not TensorFlow/Keras.
        - No `model.h5` dependency; a missing `.pth` file safely falls back to OpenCV heuristics.
        - Grad-CAM is available after training a `.pth` model.
        - Multimodal fusion combines face, voice, text, and emoji signals.
        - Digital Emotional Twin logging gives personalized history.
        - RL-style recommendation avoids repeating recent resources.
        - Animated mascot changes expression according to fused emotion.
        - Ethical monitor explains confidence and avoids diagnosis claims.
        """
    )
    st.markdown("### Dataset references for report")
    st.table(DATASET_GUIDE)


def main() -> None:
    ensure_runtime_files()
    export_resource_catalog_csv()
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)

    if "analysis_result" not in st.session_state:
        st.session_state["analysis_result"] = None
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "last_reply" not in st.session_state:
        st.session_state["last_reply"] = ""
    if "feedback_saved" not in st.session_state:
        st.session_state["feedback_saved"] = None

    with st.sidebar:
        st.header("User Inputs")
        username = st.text_input("Username", value="guest_user")
        tone_mode = st.selectbox("Adaptive reply mode", options=list(TONE_GUIDES.keys()), index=0)
        emoji = st.select_slider("Emoji mood input", options=list(EMOJI_MAP.keys()), value="😊")
        use_local_transformers = st.checkbox(
            "Enable local Hugging Face transformers downloads",
            value=False,
            help="Keep off for fastest startup. Turn on only after dependencies and internet are ready.",
        )
        st.caption(f"Free chat provider: {get_chat_provider_name(use_local_transformers)}")
        st.caption("Recommended for 8 GB RAM: batch size 4-8 and demo training epochs 1-3.")
        speak_reply = st.checkbox("Speak chatbot reply aloud on local machine", value=False)

    tab_live, tab_chat, tab_train, tab_twin, tab_setup = st.tabs(
        ["Live Emotion Studio", "Mascot Chatbot", "Training + Grad-CAM", "Digital Twin + RL", "Setup + Viva Guide"]
    )

    with tab_live:
        st.subheader("Multimodal emotion detection")
        left, right = st.columns([1.2, 1.0])
        with left:
            camera_file = st.camera_input("Capture a face image")
            image_bytes = camera_file.getvalue() if camera_file is not None else None
            audio_bytes = None
            if hasattr(st, "audio_input"):
                voice_file = st.audio_input("Record your voice")
                audio_bytes = voice_file.getvalue() if voice_file is not None else None
            else:
                st.info("Update Streamlit if browser microphone capture is not shown.")
            text_input = st.text_area(
                "Context message",
                height=140,
                placeholder="Type how you feel, what happened, or what you need help with.",
            )
            if st.button("Analyze emotional state", type="primary"):
                with st.spinner("Running PyTorch + NLP multimodal analysis..."):
                    st.session_state["analysis_result"] = live_analysis(
                        user=username,
                        emoji=emoji,
                        image_bytes=image_bytes,
                        audio_bytes=audio_bytes,
                        text_input=text_input,
                        tone_mode=tone_mode,
                        use_local_transformers=use_local_transformers,
                    )
                    st.session_state["feedback_saved"] = None
        with right:
            active_result = st.session_state.get("analysis_result")
            active_mood = active_result["fused_mood"] if active_result else "calm"
            subtitle = (
                f"{username}, I am ready to read your mood through face, voice, text, and emoji."
                if active_result is None
                else f"{username}, your fused emotional state currently looks {active_mood}."
            )
            render_mascot(active_mood, username, tone_mode, subtitle)

        result = st.session_state.get("analysis_result")
        if result:
            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Face mood", result["face_result"]["mood"].title())
            c2.metric("Voice mood", result["voice_result"]["mood"].title())
            c3.metric("Text mood", result["text_result"]["mood"].title())
            c4.metric("Fused mood", result["fused_mood"].title())
            st.metric("User Emotional State Score", f"{result['state_score']:.2%}")

            col_a, col_b = st.columns([1.2, 1.0])
            with col_a:
                st.markdown("#### Grad-CAM / face analysis view")
                if result["face_result"].get("overlay_rgb") is not None:
                    safe_image(result["face_result"]["overlay_rgb"], caption=f"Method: {result['face_result']['method']}")
                else:
                    st.info("No face image available yet.")
            with col_b:
                st.markdown("#### Fused mood scores")
                render_mood_scores(result["fusion_scores"])
                st.markdown("#### Ethical AI monitor")
                st.write(f"Confidence band: {result['ethical_report']['confidence_band']}")
                st.write(f"Modalities used: {result['ethical_report']['modalities_used']}")
                for warning in result["ethical_report"]["warnings"]:
                    st.caption(f"- {warning}")

            st.markdown("#### Voice transcript")
            voice = result["voice_result"]
            if voice["transcript"]:
                st.write(voice["transcript"])
                st.caption(f"Transcription: {voice['transcription_method']} | Energy score: {voice['energy']:.3f}")
                if voice["spoken_name_detected"]:
                    st.success("The voice transcript contains the username, so the login name was recognized in speech.")
            else:
                st.info("No voice transcript available.")

            st.markdown("#### Text intelligence")
            text_result = result["text_result"]
            st.write(
                f"Emotion: {text_result['emotion_label']} ({text_result['emotion_score']:.2f}) | "
                f"Sentiment: {text_result['sentiment_label']} ({text_result['sentiment_score']:.2f}) | "
                f"Intent: {text_result['intent']} | Method: {text_result['method']}"
            )
            if text_result["synonyms_preview"]:
                st.caption("NLTK synonym hints: " + ", ".join(text_result["synonyms_preview"]))

            st.markdown("#### Adaptive recommendation")
            render_recommendation_card(result["recommendation"])
            like_col, skip_col = st.columns(2)
            with like_col:
                if st.button("I liked this recommendation", disabled=st.session_state["feedback_saved"] is not None):
                    upsert_recommender_feedback(result["recommendation"]["id"], feedback="liked")
                    update_last_feedback(username, result["recommendation"]["id"], "liked")
                    st.session_state["feedback_saved"] = "liked"
                    st.success("Positive feedback saved.")
            with skip_col:
                if st.button("Skip this recommendation", disabled=st.session_state["feedback_saved"] is not None):
                    upsert_recommender_feedback(result["recommendation"]["id"], feedback="skipped")
                    update_last_feedback(username, result["recommendation"]["id"], "skipped")
                    st.session_state["feedback_saved"] = "skipped"
                    st.info("Skip feedback saved.")

    with tab_chat:
        st.subheader("Animated mascot chatbot")
        analysis = st.session_state.get("analysis_result")
        active_mood = analysis["fused_mood"] if analysis else "calm"
        render_mascot(active_mood, username, tone_mode, f"{username}, ask me anything. I will answer from your emotional context.")
        for role, message in st.session_state["chat_history"]:
            with st.chat_message(role):
                st.write(message)
        prompt = st.chat_input("Ask the mascot something meaningful...")
        if prompt:
            with st.chat_message("user"):
                st.write(prompt)
            reply, provider_name = generate_chat_response(
                username=username,
                question=prompt,
                current_mood=active_mood,
                tone_mode=tone_mode,
                analysis=analysis,
                use_local_transformers=use_local_transformers,
            )
            with st.chat_message("assistant"):
                st.write(reply)
                st.caption(f"Provider: {provider_name}")
            st.session_state["chat_history"].append(("user", prompt))
            st.session_state["chat_history"].append(("assistant", reply))
            st.session_state["last_reply"] = reply
            current_recommendation = analysis["recommendation"] if analysis else {"id": "", "title": "", "url": "", "source": ""}
            log_interaction(
                user=username,
                face_label=analysis["face_result"]["label"] if analysis else "neutral",
                face_mood=analysis["face_result"]["mood"] if analysis else "calm",
                voice_mood=analysis["voice_result"]["mood"] if analysis else "calm",
                text_mood=analysis["text_result"]["mood"] if analysis else simple_text_mood(prompt),
                emoji_mood=analysis["emoji_mood"] if analysis else "calm",
                fused_mood=active_mood,
                state_score=analysis["state_score"] if analysis else 0.0,
                tone_mode=tone_mode,
                recommendation=current_recommendation,
                chat_query=prompt,
                chat_reply=reply,
            )
            if speak_reply:
                st.info(speak_text_locally(reply))
        if st.button("Clear chat history"):
            st.session_state["chat_history"] = []
            st.session_state["last_reply"] = ""
            st.success("Chat history cleared.")

    with tab_train:
        st.subheader("PyTorch EfficientNetV2-S training pipeline")
        st.caption("Expected folders: dataset/train, dataset/val, and dataset/test. The trained model is saved as .pth.")
        summary_rows = dataset_directory_summary(DATASET_DIR)
        if summary_rows:
            st.markdown("#### Current dataset summary")
            st.table(summary_rows)
        else:
            st.info("No local dataset found yet. Create the dataset folders shown in the setup tab.")

        batch_size = st.select_slider("Batch size", options=[4, 8, 12, 16], value=8)
        epochs = st.slider("Training epochs", min_value=1, max_value=6, value=2)
        learning_rate = st.select_slider("Learning rate", options=[1e-4, 2e-4, 5e-4, 1e-3], value=2e-4)
        freeze_backbone = st.checkbox("Freeze EfficientNetV2 backbone for lightweight demo training", value=True)
        if st.button("Train EfficientNetV2-S model"):
            progress_bar = st.progress(0.0)
            history_placeholder = st.empty()

            def update_progress(fraction, payload):
                progress_bar.progress(float(fraction))
                history_placeholder.write(payload)

            try:
                with st.spinner("Training PyTorch EfficientNetV2-S on the local dataset..."):
                    _, history, evaluation, class_names = train_emotion_model(
                        dataset_root=DATASET_DIR,
                        batch_size=batch_size,
                        epochs=epochs,
                        learning_rate=float(learning_rate),
                        freeze_backbone=freeze_backbone,
                        progress_callback=update_progress,
                    )
                st.success(f"Training complete. Model saved to {MODEL_PATH.name}")
                st.write(f"Classes: {', '.join(class_names)}")
                st.table(history)
                st.markdown("#### Evaluation metrics")
                st.json(evaluation)
            except Exception as exc:
                st.error(f"Training failed: {exc}")

        if MODEL_META_PATH.exists():
            st.markdown("#### Saved model metadata")
            try:
                st.json(json.loads(MODEL_META_PATH.read_text(encoding="utf-8")))
            except Exception:
                st.caption("Model metadata could not be parsed.")

    with tab_twin:
        st.subheader("Digital Emotional Twin and recommender memory")
        twin_rows = read_csv_rows(TWIN_LOG_PATH)
        if twin_rows:
            st.markdown("#### Interaction log")
            safe_dataframe(twin_rows)
            st.download_button("Download twin log CSV", TWIN_LOG_PATH.read_bytes(), file_name=TWIN_LOG_PATH.name, mime="text/csv")
        else:
            st.info("No interaction log entries yet.")

        st.markdown("#### Recommender stats")
        recommender_rows = get_recommender_stats()
        if recommender_rows:
            safe_dataframe(recommender_rows)
        else:
            st.info("No recommender exposure data yet.")

        st.markdown("#### Structured resource catalog")
        catalog = catalog_rows()
        mood_filter = st.selectbox("Mood filter", options=["all"] + MOOD_CHOICES, index=0)
        source_filter = st.selectbox("Source filter", options=["all", "YouTube", "Spotify", "YouTube Music"], index=0)
        filtered = [
            item
            for item in catalog
            if (mood_filter == "all" or item["mood"] == mood_filter) and (source_filter == "all" or item["source"] == source_filter)
        ]
        st.write(f"Catalog items: {len(filtered)} / {len(catalog)}")
        safe_dataframe(filtered)
        st.download_button("Download catalog CSV", CATALOG_EXPORT_PATH.read_bytes(), file_name=CATALOG_EXPORT_PATH.name, mime="text/csv")

    with tab_setup:
        render_setup_tab(use_local_transformers=use_local_transformers)


if __name__ == "__main__":
    main()
