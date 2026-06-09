from __future__ import annotations
import argparse
import csv
import html
import io
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
import os
os.environ["HF_TOKEN"] = "hf_dHMVCSQtMfQNurUDVY~ixLoQKZCrILGkUpR"

# Reduce noisy third-party logs before imports.
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
# Streamlit's source watcher can introspect PyTorch internals on Windows/Python 3.11
# and leave a noisy "Event loop is closed" traceback during shutdown.
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")
warnings.filterwarnings("ignore", message=r".*Accessing `__path__`.*")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
for logger_name in (
    "transformers",
    "transformers.utils",
    "transformers.utils.import_utils",
):
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)

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

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s

    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    DataLoader = None
    EfficientNet_V2_S_Weights = None
    efficientnet_v2_s = None
    TORCH_AVAILABLE = False

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    TRANSFORMERS_AVAILABLE = True
except Exception:
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    TRANSFORMERS_AVAILABLE = False


APP_TITLE = "Emotionally Intelligent Animated Mascot Chatbot"
APP_SUBTITLE = (
    "Single-file PyTorch + Streamlit project with EfficientNetV2-S, Grad-CAM, "
    "multimodal fusion, digital emotional twin logging, and free Hugging Face API chat."
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
    "🙂": "calm",
    "😴": "calm",
    "😡": "energetic",
    "😤": "energetic",
    "🤩": "energetic",
}

EMOTION_TO_MOOD = {
    "happy": "happy",
    "joy": "happy",
    "love": "happy",
    "positive": "happy",
    "surprise": "energetic",
    "excitement": "energetic",
    "excited": "energetic",
    "angry": "energetic",
    "anger": "energetic",
    "frustration": "energetic",
    "fear": "sad",
    "sad": "sad",
    "sadness": "sad",
    "negative": "sad",
    "neutral": "calm",
    "calm": "calm",
    "relaxed": "calm",
}

DEFAULT_FACE_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

FALLBACK_ACTIONS = {
    "happy": "Write one win, one gratitude point, and one next meaningful goal.",
    "sad": "Do two minutes of slow breathing, drink water, and text one trusted person.",
    "calm": "Protect this state with one low-distraction focus sprint.",
    "energetic": "Channel your energy into one important task or a short movement break.",
}

MOOD_QUESTION_BANK = {
    "happy": [
        "What specific action created this positive shift?",
        "How can you repeat that action tomorrow in a small way?",
        "Who can you share this momentum with today?",
    ],
    "sad": [
        "What feels heaviest right now?",
        "What helped even a little the last time this happened?",
        "What tiny step would make the next hour easier?",
    ],
    "calm": [
        "What is currently protecting your balance?",
        "Which routine is helping your focus most?",
        "What boundary will help maintain this calm state?",
    ],
    "energetic": [
        "Is this energy helping progress or causing overload?",
        "What single priority deserves this energy first?",
        "What boundary keeps this intensity healthy?",
    ],
}

TONE_GUIDES = {
    "Therapist": "Warm, reflective, and structured. Ask deep but safe questions.",
    "Friendly": "Caring and conversational. Sound natural and supportive.",
    "Motivational": "Positive and action-oriented without sounding harsh.",
}

DIRECT_VIDEO_URLS = {
    "happy": [
        "https://www.youtube.com/watch?v=d-diB65scQU",
        "https://www.youtube.com/watch?v=OPf0YbXqDm0",
        "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
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

YOUTUBE_SEARCH_QUERIES = {
    "happy": ["feel good playlist", "confidence boost songs", "gratitude meditation"],
    "sad": ["comfort songs", "self compassion meditation", "gentle piano healing"],
    "calm": ["lofi focus session", "rain sounds for study", "mindful breathing guide"],
    "energetic": ["workout motivation mix", "focus sprint soundtrack", "productivity hype music"],
}

SPOTIFY_SEARCH_QUERIES = {
    "happy": ["happy playlist", "feel good hits", "good vibes only"],
    "sad": ["comfort songs", "calm down playlist", "healing ambient music"],
    "calm": ["lofi beats", "deep focus", "peaceful piano"],
    "energetic": ["workout hits", "motivation songs", "high energy mix"],
}

YTMUSIC_SEARCH_QUERIES = {
    "happy": ["cheerful playlist", "dance around room songs", "joyful clean mix"],
    "sad": ["rainy evening songs", "soft comfort music", "quiet reflection songs"],
    "calm": ["brown noise focus", "ambient reading music", "calm coding playlist"],
    "energetic": ["power walk songs", "confidence rap clean", "quick cardio mix"],
}

INTENT_SEEDS = {
    "stress_relief": {"stress", "overwhelmed", "pressure", "anxious", "burnout"},
    "motivation": {"motivation", "discipline", "goal", "progress", "improve"},
    "study_focus": {"study", "exam", "assignment", "project", "focus"},
    "confidence": {"confidence", "presentation", "interview", "fear", "nervous"},
    "loneliness": {"alone", "lonely", "isolated", "empty"},
    "self_reflection": {"reflect", "journal", "understand", "meaning"},
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
        raise RuntimeError("Streamlit is not installed. Install requirements and run `streamlit run app.py`.")


def utc_now() -> str:
    return datetime.utcnow().isoformat()


def slugify(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower())
    return text.strip("_") or "unknown"


def ensure_csv(path: Path, headers: list[str]) -> None:
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()


def ensure_runtime_files() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    ensure_csv(
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
    ensure_csv(
        RECOMMENDER_STATS_PATH,
        ["ItemId", "Exposures", "Likes", "Skips", "LastShown", "LastFeedback"],
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, headers: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({header: row.get(header, "") for header in headers})


def append_row(path: Path, headers: list[str], row: dict[str, Any]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        if not exists:
            writer.writeheader()
        writer.writerow({header: row.get(header, "") for header in headers})


@lru_cache(maxsize=1)
def build_resource_catalog() -> tuple[CatalogEntry, ...]:
    entries: list[CatalogEntry] = []
    for mood in MOOD_CHOICES:
        fallback = FALLBACK_ACTIONS[mood]
        for i, url in enumerate(DIRECT_VIDEO_URLS[mood], start=1):
            entries.append(
                CatalogEntry(
                    id=f"{mood}_direct_{i}",
                    mood=mood,
                    title=f"{mood.title()} direct video {i}",
                    url=url,
                    source="YouTube",
                    resource_type="youtube_video",
                    playable=True,
                    tags=f"{mood},video,direct",
                    offline_fallback=fallback,
                )
            )
        for i, query in enumerate(YOUTUBE_SEARCH_QUERIES[mood], start=1):
            entries.append(
                CatalogEntry(
                    id=f"{mood}_yt_search_{i}",
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
        for i, query in enumerate(SPOTIFY_SEARCH_QUERIES[mood], start=1):
            entries.append(
                CatalogEntry(
                    id=f"{mood}_spotify_search_{i}",
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
        for i, query in enumerate(YTMUSIC_SEARCH_QUERIES[mood], start=1):
            entries.append(
                CatalogEntry(
                    id=f"{mood}_ytmusic_search_{i}",
                    mood=mood,
                    title=f"{mood.title()} YouTube Music search: {query.title()}",
                    url=f"https://music.youtube.com/search?q={urllib.parse.quote_plus(query)}",
                    source="YouTube Music",
                    resource_type="ytmusic_search",
                    playable=False,
                    tags=f"{mood},ytmusic,{slugify(query)}",
                    offline_fallback=fallback,
                )
            )
    return tuple(entries)


def catalog_rows() -> list[dict[str, Any]]:
    return [asdict(item) for item in build_resource_catalog()]


def export_catalog(path: Path = CATALOG_EXPORT_PATH) -> Path:
    rows = catalog_rows()
    if not rows:
        return path
    write_rows(path, list(rows[0].keys()), rows)
    return path


def get_recommender_stats() -> list[dict[str, str]]:
    ensure_runtime_files()
    return read_rows(RECOMMENDER_STATS_PATH)


def upsert_feedback(item_id: str, feedback: str = "shown") -> None:
    rows = get_recommender_stats()
    headers = ["ItemId", "Exposures", "Likes", "Skips", "LastShown", "LastFeedback"]
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
    write_rows(RECOMMENDER_STATS_PATH, headers, rows)


def get_user_history(user: str) -> list[str]:
    rows = read_rows(TWIN_LOG_PATH)
    return [
        str(row.get("RecommendedId", "")).strip()
        for row in rows
        if str(row.get("User", "")) == str(user) and str(row.get("RecommendedId", "")).strip()
    ]


def recommend_item(user: str, mood: str, last_n: int = 5) -> dict[str, Any]:
    catalog = catalog_rows()
    candidates = [row for row in catalog if row["mood"] == mood] or catalog[:]
    recent_ids = set(get_user_history(user)[-last_n:])
    stats_by_id = {row["ItemId"]: row for row in get_recommender_stats()}

    scored: list[tuple[float, dict[str, Any]]] = []
    for item in candidates:
        stats = stats_by_id.get(item["id"], {})
        exposures = int(str(stats.get("Exposures", "0") or "0"))
        likes = int(str(stats.get("Likes", "0") or "0"))
        skips = int(str(stats.get("Skips", "0") or "0"))
        like_ratio = likes / max(exposures, 1)
        skip_ratio = skips / max(exposures, 1)
        novelty = 1.0 / (exposures + 1.0)
        playable_bonus = 0.12 if item.get("playable") else 0.0
        recent_penalty = 0.50 if item["id"] in recent_ids else 0.0
        score = (
            0.45 * like_ratio
            + 0.35 * novelty
            + playable_bonus
            - 0.20 * skip_ratio
            - recent_penalty
            + random.uniform(0.0, 0.05)
        )
        scored.append((score, item))

    pool = [(s, i) for s, i in scored if i["id"] not in recent_ids] or scored
    pool.sort(key=lambda pair: pair[0], reverse=True)
    top = pool[: min(8, len(pool))]
    weights = np.array([max(score, 0.01) for score, _ in top], dtype=np.float64)
    weights = weights / weights.sum()
    chosen_idx = int(np.random.choice(np.arange(len(top)), p=weights))
    choice = top[chosen_idx][1]
    upsert_feedback(choice["id"], "shown")
    return choice


def update_last_feedback(user: str, item_id: str, feedback: str) -> None:
    rows = read_rows(TWIN_LOG_PATH)
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
    write_rows(TWIN_LOG_PATH, list(rows[0].keys()), rows)


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
    append_row(
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


def get_twin_summary(user: str, limit: int = 12) -> str:
    rows = [row for row in read_rows(TWIN_LOG_PATH) if str(row.get("User", "")) == str(user)]
    if not rows:
        return "No digital emotional twin history is available yet."
    recent = rows[-limit:]
    moods = [str(row.get("FusedMood", "calm") or "calm") for row in recent]
    dominant = Counter(moods).most_common(1)[0][0] if moods else "calm"
    seq = ", ".join(moods[-5:]) if moods else "none"
    last_title = next((r.get("RecommendedTitle", "") for r in reversed(recent) if r.get("RecommendedTitle")), "")
    last_feedback = next((r.get("Feedback", "") for r in reversed(recent) if r.get("Feedback")), "")
    return (
        f"Dominant mood: {dominant}. "
        f"Recent sequence: {seq}. "
        f"Last recommendation: {last_title or 'not available'}. "
        f"Latest feedback: {last_feedback or 'not recorded'}."
    )


def simple_text_mood(text: str) -> str:
    lowered = text.lower()
    if any(t in lowered for t in ("sad", "lonely", "cry", "grief", "hurt")):
        return "sad"
    if any(t in lowered for t in ("happy", "joy", "love", "great", "awesome")):
        return "happy"
    if any(t in lowered for t in ("angry", "mad", "stress", "frustrat", "rage")):
        return "energetic"
    if any(t in lowered for t in ("calm", "focus", "peace", "relaxed")):
        return "calm"
    return "calm"


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
    for corpus in ("wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"corpora/{corpus}")
        except LookupError:
            try:
                nltk.download(corpus, quiet=True)
            except Exception:
                return


@lru_cache(maxsize=256)
def synonyms(word: str) -> tuple[str, ...]:
    if not NLTK_AVAILABLE:
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


def infer_intent(text: str) -> dict[str, Any]:
    token_set = set(tokenize_words(text))
    scores: dict[str, int] = {}
    matched: dict[str, list[str]] = {}
    for intent, seeds in INTENT_SEEDS.items():
        expanded = set(seeds)
        for seed in list(seeds):
            expanded.update(tokenize_words(" ".join(synonyms(seed))))
        hits = sorted(token for token in token_set if token in expanded)
        scores[intent] = len(hits)
        matched[intent] = hits
    best = max(scores.items(), key=lambda x: x[1])[0] if scores else "general_support"
    if scores.get(best, 0) == 0:
        best = "general_support"
    sample_synonyms = sorted({s for t in list(token_set)[:6] for s in synonyms(t)[:2]})[:8]
    return {"intent": best, "matched_terms": matched.get(best, []), "synonyms_preview": sample_synonyms}


@lru_cache(maxsize=1)
def load_text_emotion_assets():
    if not TRANSFORMERS_AVAILABLE:
        return None, None
    try:
        tokenizer = AutoTokenizer.from_pretrained(TEXT_EMOTION_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(TEXT_EMOTION_MODEL)
        model.eval()
        return tokenizer, model
    except Exception:
        return None, None


@lru_cache(maxsize=1)
def load_text_sentiment_assets():
    if not TRANSFORMERS_AVAILABLE:
        return None, None
    try:
        tokenizer = AutoTokenizer.from_pretrained(TEXT_SENTIMENT_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(TEXT_SENTIMENT_MODEL)
        model.eval()
        return tokenizer, model
    except Exception:
        return None, None


def run_classifier(tokenizer: Any, model: Any, text: str) -> tuple[str, float, dict[str, float]]:
    if tokenizer is None or model is None or not TORCH_AVAILABLE:
        return "", 0.0, {}
    with torch.no_grad():
        encoded = tokenizer(text[:512], return_tensors="pt", truncation=True)
        logits = model(**encoded).logits
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
    label_map = model.config.id2label if hasattr(model, "config") else {}
    prob_map: dict[str, float] = {}
    for idx, prob in enumerate(probs):
        label = str(label_map.get(idx, f"class_{idx}"))
        prob_map[label] = float(prob)
    top_label = max(prob_map.items(), key=lambda x: x[1])[0] if prob_map else "neutral"
    top_prob = float(prob_map.get(top_label, 0.0))
    return top_label, top_prob, prob_map


def analyze_text(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {
            "text": "",
            "mood": "calm",
            "confidence": 0.5,
            "emotion_label": "neutral",
            "emotion_score": 0.5,
            "sentiment_label": "NEUTRAL",
            "sentiment_score": 0.5,
            "intent": "general_support",
            "matched_terms": [],
            "synonyms_preview": [],
            "mood_scores": {m: 0.0 for m in MOOD_CHOICES},
            "method": "empty_text",
        }

    intent = infer_intent(text)
    mood_scores = {m: 0.0 for m in MOOD_CHOICES}
    emotion_label = simple_text_mood(text)
    emotion_score = 0.55
    sentiment_label = "NEUTRAL"
    sentiment_score = 0.5
    method = "heuristic"

    emo_tok, emo_model = load_text_emotion_assets()
    emo_label, emo_prob, emo_all = run_classifier(emo_tok, emo_model, text)
    if emo_all:
        emotion_label = emo_label
        emotion_score = emo_prob
        for label, score in emo_all.items():
            mood_scores[normalize_label_to_mood(label)] += float(score)
        method = "transformers"

    sent_tok, sent_model = load_text_sentiment_assets()
    sent_label, sent_prob, _ = run_classifier(sent_tok, sent_model, text)
    if sent_label:
        sentiment_label = sent_label.upper()
        sentiment_score = sent_prob
        method = "transformers"

    if max(mood_scores.values()) <= 0.0:
        mood_scores[simple_text_mood(text)] = 1.0
    if sentiment_label.startswith("NEG"):
        mood_scores["sad"] += 0.12
    if sentiment_label.startswith("POS"):
        mood_scores["happy"] += 0.12

    mood = max(mood_scores.items(), key=lambda x: x[1])[0]
    confidence = float(max(max(mood_scores.values()), emotion_score, sentiment_score))
    return {
        "text": text,
        "mood": mood,
        "confidence": confidence,
        "emotion_label": emotion_label,
        "emotion_score": float(emotion_score),
        "sentiment_label": sentiment_label,
        "sentiment_score": float(sentiment_score),
        "intent": intent["intent"],
        "matched_terms": intent["matched_terms"],
        "synonyms_preview": intent["synonyms_preview"],
        "mood_scores": mood_scores,
        "method": method,
    }


def transcribe_audio(audio_bytes: bytes | None) -> tuple[str, str]:
    if not audio_bytes:
        return "", "no_audio"
    if sr is None:
        return "", "speechrecognition_missing"
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
        return text.strip(), "google_web_speech"
    except Exception:
        return "", "transcription_failed"


def wav_energy(audio_bytes: bytes | None) -> float:
    if not audio_bytes:
        return 0.0
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            frame_count = wf.getnframes()
            sample_width = wf.getsampwidth()
            channels = wf.getnchannels()
            frames = wf.readframes(frame_count)
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


def analyze_voice(audio_bytes: bytes | None, username: str) -> dict[str, Any]:
    transcript, method = transcribe_audio(audio_bytes)
    text_result = analyze_text(transcript) if transcript else analyze_text("")
    energy = wav_energy(audio_bytes)
    scores = dict(text_result["mood_scores"])
    if energy > 0.14:
        scores["energetic"] += 0.15
    if energy < 0.04:
        scores["calm"] += 0.08
    if text_result["sentiment_label"].startswith("NEG") and energy < 0.05:
        scores["sad"] += 0.10
    mood = max(scores.items(), key=lambda x: x[1])[0]
    name_detected = bool(username and transcript and username.lower() in transcript.lower())
    return {
        "transcript": transcript,
        "transcription_method": method,
        "energy": energy,
        "mood": mood,
        "confidence": float(max(scores.values()) if scores else 0.0),
        "spoken_name_detected": name_detected,
        "text_result": text_result,
    }


def decode_image_bytes(image_bytes: bytes | None) -> np.ndarray | None:
    if not image_bytes or cv2 is None:
        return None
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return image


@lru_cache(maxsize=1)
def face_cascade():
    if cv2 is None:
        return None
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return cascade if not cascade.empty() else None


@lru_cache(maxsize=1)
def smile_cascade():
    if cv2 is None:
        return None
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
    return cascade if not cascade.empty() else None


def largest_face_box(image_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    cascade = face_cascade()
    if cascade is None or cv2 is None:
        return None
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    return max(faces, key=lambda box: int(box[2] * box[3]))


def heuristic_face_label(face_bgr: np.ndarray) -> tuple[str, float]:
    if cv2 is None:
        return "neutral", 0.4
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    texture = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    has_smile = False
    sc = smile_cascade()
    if sc is not None:
        try:
            smiles = sc.detectMultiScale(gray, scaleFactor=1.7, minNeighbors=20, minSize=(25, 25))
            has_smile = len(smiles) > 0
        except Exception:
            has_smile = False
    if has_smile or brightness > 150:
        return "happy", 0.58
    if texture > 450 and brightness < 130:
        return "angry", 0.47
    if brightness < 95:
        return "sad", 0.46
    return "neutral", 0.44


def imagenet_mean_std() -> tuple[list[float], list[float]]:
    # Fixed values for EfficientNet family.
    return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def image_to_tensor(image_rgb: np.ndarray, size: int = 224) -> Any:
    if not TORCH_AVAILABLE:
        return None
    resized = cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_AREA)
    arr = resized.astype(np.float32) / 255.0
    mean, std = imagenet_mean_std()
    arr = (arr - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr).unsqueeze(0).float()


def build_model(num_classes: int, pretrained: bool = True) -> Any:
    if not TORCH_AVAILABLE:
        return None
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
        "architecture": "efficientnet_v2_s",
        "class_names": class_names,
        "saved_at_utc": utc_now(),
        "model_path": str(MODEL_PATH),
    }
    if extra:
        payload.update(extra)
    MODEL_META_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_checkpoint(model: Any, class_names: list[str], extra: dict[str, Any] | None = None) -> None:
    if not TORCH_AVAILABLE:
        return
    payload = {
        "architecture": "efficientnet_v2_s",
        "class_names": class_names,
        "saved_at_utc": utc_now(),
        "model_state_dict": model.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, MODEL_PATH)
    save_model_metadata(class_names, extra=extra)


@lru_cache(maxsize=1)
def load_checkpoint() -> tuple[Any, dict[str, Any]]:
    if not TORCH_AVAILABLE or not MODEL_PATH.exists():
        return None, {}
    try:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
        classes = checkpoint.get("class_names") or DEFAULT_FACE_CLASSES
        model = build_model(len(classes), pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        model.eval()
        metadata = {}
        if MODEL_META_PATH.exists():
            try:
                metadata = json.loads(MODEL_META_PATH.read_text(encoding="utf-8"))
            except Exception:
                metadata = {}
        metadata.setdefault("class_names", classes)
        return model, metadata
    except Exception:
        return None, {}


def last_conv_module(model: Any) -> tuple[str | None, Any]:
    name, module = None, None
    if model is None:
        return None, None
    for n, m in model.named_modules():
        if TORCH_AVAILABLE and isinstance(m, nn.Conv2d):
            name, module = n, m
    return name, module


def gradcam_heatmap(model: Any, tensor: Any, target_index: int | None = None) -> np.ndarray | None:
    if not TORCH_AVAILABLE or model is None:
        return None
    layer_name, layer = last_conv_module(model)
    if layer_name is None or layer is None:
        return None
    activations: list[Any] = []
    gradients: list[Any] = []

    def f_hook(_module, _inputs, output):
        activations.append(output.detach())

    def b_hook(_module, _gin, gout):
        gradients.append(gout[0].detach())

    h1 = layer.register_forward_hook(f_hook)
    h2 = layer.register_full_backward_hook(b_hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(tensor)
        if target_index is None:
            target_index = int(torch.argmax(logits, dim=1).item())
        score = logits[:, target_index].sum()
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
    if cv2 is None or heatmap is None:
        return None
    heatmap = cv2.resize(heatmap, (face_bgr.shape[1], face_bgr.shape[0]))
    colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(colored, 0.35, face_bgr, 0.65, 0.0)


def predict_face(image_bytes: bytes | None) -> dict[str, Any]:
    if not image_bytes:
        return {
            "label": "neutral",
            "mood": "calm",
            "confidence": 0.0,
            "method": "no_image",
            "overlay_rgb": None,
            "face_found": False,
        }
    image = decode_image_bytes(image_bytes)
    if image is None or cv2 is None:
        return {
            "label": "neutral",
            "mood": "calm",
            "confidence": 0.0,
            "method": "opencv_missing",
            "overlay_rgb": None,
            "face_found": False,
        }

    box = largest_face_box(image)
    display = image.copy()
    if box is None:
        x, y, w, h = 0, 0, image.shape[1], image.shape[0]
        face = image
    else:
        x, y, w, h = map(int, box)
        face = image[y : y + h, x : x + w]
        cv2.rectangle(display, (x, y), (x + w, y + h), (70, 255, 140), 2)

    model, metadata = load_checkpoint()
    if model is not None and TORCH_AVAILABLE:
        try:
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            tensor = image_to_tensor(face_rgb, size=224)
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
            classes = metadata.get("class_names") or DEFAULT_FACE_CLASSES
            pred_idx = int(np.argmax(probs))
            label = classes[pred_idx] if pred_idx < len(classes) else f"class_{pred_idx}"
            conf = float(probs[pred_idx])
            heatmap = gradcam_heatmap(model, tensor, pred_idx)
            overlay = overlay_heatmap(face, heatmap)
            if overlay is not None and box is not None:
                display[y : y + h, x : x + w] = cv2.resize(overlay, (w, h))
                cv2.rectangle(display, (x, y), (x + w, y + h), (70, 255, 140), 2)
            elif overlay is not None:
                display = overlay
            return {
                "label": str(label),
                "mood": normalize_label_to_mood(str(label)),
                "confidence": conf,
                "method": "trained_efficientnet_v2_s",
                "overlay_rgb": cv2.cvtColor(display, cv2.COLOR_BGR2RGB),
                "face_found": box is not None,
            }
        except Exception:
            pass

    label, conf = heuristic_face_label(face)
    return {
        "label": label,
        "mood": normalize_label_to_mood(label),
        "confidence": conf,
        "method": "opencv_heuristic",
        "overlay_rgb": cv2.cvtColor(display, cv2.COLOR_BGR2RGB),
        "face_found": box is not None,
    }


def dataset_has_images(split_dir: Path) -> bool:
    if not split_dir.exists():
        return False
    for p in split_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            return True
    return False


def dataset_summary(root: Path = DATASET_DIR) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        split_dir = root / split
        if not split_dir.exists():
            continue
        for class_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
            count = sum(1 for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS)
            rows.append({"split": split, "class_name": class_dir.name, "count": count})
    return rows


def np_to_tensor_image(image_bgr: np.ndarray, train: bool = False) -> Any:
    if not TORCH_AVAILABLE or cv2 is None:
        return None
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
    if train and random.random() < 0.5:
        rgb = cv2.flip(rgb, 1)
    arr = rgb.astype(np.float32) / 255.0
    mean, std = imagenet_mean_std()
    arr = (arr - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr).float()


if TORCH_AVAILABLE:
    class FolderDataset(torch.utils.data.Dataset):
        def __init__(self, root: Path, class_to_idx: dict[str, int] | None = None, train: bool = False):
            self.root = root
            self.train = train
            classes = sorted([p.name for p in root.iterdir() if p.is_dir()]) if root.exists() else []
            if class_to_idx is None:
                self.class_to_idx = {name: idx for idx, name in enumerate(classes)}
            else:
                self.class_to_idx = dict(class_to_idx)
            self.samples: list[tuple[Path, int]] = []
            for cls_name, idx in self.class_to_idx.items():
                cls_dir = root / cls_name
                if not cls_dir.exists():
                    continue
                for fp in cls_dir.rglob("*"):
                    if fp.is_file() and fp.suffix.lower() in IMAGE_EXTENSIONS:
                        self.samples.append((fp, idx))

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, index: int):
            path, label = self.samples[index]
            image = cv2.imread(str(path))
            if image is None:
                image = np.zeros((224, 224, 3), dtype=np.uint8)
            tensor = np_to_tensor_image(image, train=self.train)
            return tensor, torch.tensor(label, dtype=torch.long)
else:
    class FolderDataset:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is not installed.")


def build_dataloaders(dataset_root: Path, batch_size: int):
    if not TORCH_AVAILABLE or cv2 is None:
        raise RuntimeError("PyTorch or OpenCV is not available.")
    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"
    test_dir = dataset_root / "test"
    if not dataset_has_images(train_dir):
        raise RuntimeError("No images found in dataset/train.")

    train_ds = FolderDataset(train_dir, train=True)
    class_to_idx = train_ds.class_to_idx
    class_names = list(class_to_idx.keys())
    if len(class_names) < 2:
        raise RuntimeError("At least two classes are required for training.")

    val_ds = FolderDataset(val_dir, class_to_idx=class_to_idx, train=False) if dataset_has_images(val_dir) else None
    test_ds = FolderDataset(test_dir, class_to_idx=class_to_idx, train=False) if dataset_has_images(test_dir) else None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0) if val_ds else None
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0) if test_ds else None
    return train_loader, val_loader, test_loader, class_names


def train_epoch(model: Any, loader: Any, optimizer: Any, criterion: Any, device: Any) -> tuple[float, float]:
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
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


def eval_epoch(model: Any, loader: Any, criterion: Any, device: Any) -> tuple[float, float]:
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
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


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            matrix[t, p] += 1
    return matrix


def classification_report_np(matrix: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    total = int(matrix.sum())
    precisions, recalls, f1s, supports = [], [], [], []
    for idx, name in enumerate(class_names):
        tp = int(matrix[idx, idx])
        fp = int(matrix[:, idx].sum() - tp)
        fn = int(matrix[idx, :].sum() - tp)
        support = int(matrix[idx, :].sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
        report[name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1-score": round(f1, 4),
            "support": support,
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)

    accuracy = float(np.trace(matrix) / max(total, 1))
    report["accuracy"] = round(accuracy, 4)
    report["macro avg"] = {
        "precision": round(float(np.mean(precisions)) if precisions else 0.0, 4),
        "recall": round(float(np.mean(recalls)) if recalls else 0.0, 4),
        "f1-score": round(float(np.mean(f1s)) if f1s else 0.0, 4),
        "support": total,
    }
    if sum(supports) > 0:
        report["weighted avg"] = {
            "precision": round(float(np.average(precisions, weights=supports)), 4),
            "recall": round(float(np.average(recalls, weights=supports)), 4),
            "f1-score": round(float(np.average(f1s, weights=supports)), 4),
            "support": total,
        }
    else:
        report["weighted avg"] = {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": total}
    return report


def avg_ranks(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(scores)
    ranks = np.zeros_like(scores, dtype=np.float64)
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start
        while end + 1 < len(scores) and sorted_scores[end + 1] == sorted_scores[start]:
            end += 1
        rank = (start + end + 2) / 2.0
        ranks[order[start : end + 1]] = rank
        start = end + 1
    return ranks


def binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    pos = int(y_true.sum())
    neg = int(len(y_true) - pos)
    if pos == 0 or neg == 0:
        return None
    ranks = avg_ranks(y_score.astype(np.float64))
    pos_sum = float(ranks[y_true == 1].sum())
    auc = (pos_sum - pos * (pos + 1) / 2.0) / (pos * neg)
    return float(auc)


def macro_roc_auc(y_true: np.ndarray, y_prob: np.ndarray, n_classes: int) -> float | None:
    aucs: list[float] = []
    for c in range(n_classes):
        auc = binary_auc((y_true == c).astype(np.int32), y_prob[:, c])
        if auc is not None:
            aucs.append(auc)
    if not aucs:
        return None
    return float(np.mean(aucs))


def evaluate_model(model: Any, loader: Any, class_names: list[str], device: Any) -> dict[str, Any]:
    if loader is None:
        return {"message": "No validation/test split available."}
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            y_prob.extend(probs.tolist())
            y_pred.extend(preds.tolist())
            y_true.extend(labels.numpy().tolist())
    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_pred_np = np.asarray(y_pred, dtype=np.int64)
    y_prob_np = np.asarray(y_prob, dtype=np.float64)
    matrix = confusion_matrix_np(y_true_np, y_pred_np, len(class_names))
    report = classification_report_np(matrix, class_names)
    auc = macro_roc_auc(y_true_np, y_prob_np, len(class_names))
    return {"report": report, "confusion_matrix": matrix.tolist(), "roc_auc_macro": round(auc, 4) if auc else None}


def train_model(
    dataset_root: Path,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    freeze_backbone: bool,
    progress_callback=None,
) -> tuple[list[dict[str, float]], dict[str, Any], list[str]]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed.")
    train_loader, val_loader, test_loader, class_names = build_dataloaders(dataset_root, batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(class_names), pretrained=True)
    if freeze_backbone:
        for p in model.features.parameters():
            p.requires_grad = False
    model.to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc = (0.0, 0.0)
        if val_loader is not None:
            va_loss, va_acc = eval_epoch(model, val_loader, criterion, device)
        row = {
            "epoch": float(epoch),
            "train_loss": round(tr_loss, 4),
            "train_accuracy": round(tr_acc, 4),
            "val_loss": round(va_loss, 4),
            "val_accuracy": round(va_acc, 4),
        }
        history.append(row)
        if progress_callback:
            progress_callback(epoch / epochs, row)

    model_cpu = model.cpu()
    save_checkpoint(
        model_cpu,
        class_names,
        extra={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "freeze_backbone": freeze_backbone,
            "dataset_root": str(dataset_root),
        },
    )
    load_checkpoint.cache_clear()
    evaluation = evaluate_model(model.to(device), test_loader or val_loader, class_names, device)
    return history, evaluation, class_names


def fuse_emotions(
    face_result: dict[str, Any],
    voice_result: dict[str, Any],
    text_result: dict[str, Any],
    emoji_mood: str,
) -> tuple[str, dict[str, float], float]:
    scores = {m: 0.0 for m in MOOD_CHOICES}
    face_mood = face_result.get("mood", "calm")
    voice_mood = voice_result.get("mood", "calm")
    text_mood = text_result.get("mood", "calm")
    face_conf = float(face_result.get("confidence") or 0.45)
    voice_conf = float(voice_result.get("confidence") or 0.45)
    text_conf = float(text_result.get("confidence") or 0.45)

    scores[face_mood] += 0.35 * max(face_conf, 0.30)
    scores[voice_mood] += 0.25 * max(voice_conf, 0.30)
    scores[text_mood] += 0.25 * max(text_conf, 0.30)
    scores[emoji_mood] += 0.15
    fused = max(scores.items(), key=lambda x: x[1])[0]
    total = sum(scores.values()) or 1.0
    state_score = scores[fused] / total
    return fused, scores, float(state_score)


def ethical_report(face_result: dict[str, Any], voice_result: dict[str, Any], text_result: dict[str, Any], scores: dict[str, float]) -> dict[str, Any]:
    warnings_out: list[str] = []
    modalities = 0
    if face_result.get("method") != "no_image":
        modalities += 1
    if voice_result.get("transcript") or voice_result.get("transcription_method") != "no_audio":
        modalities += 1
    if text_result.get("text"):
        modalities += 1
    if face_result.get("method") == "opencv_heuristic":
        warnings_out.append("Face analysis is running in heuristic fallback mode. Train the .pth model for stronger accuracy.")
    if modalities < 2:
        warnings_out.append("Fusion confidence improves when at least two modalities are available.")
    max_score = max(scores.values()) if scores else 0.0
    if max_score < 0.25:
        warnings_out.append("Signals are mixed, so confidence is moderate.")
    warnings_out.append("This assistant provides supportive guidance, not medical diagnosis.")
    band = "High" if max_score >= 0.45 else "Medium" if max_score >= 0.28 else "Low"
    return {"modalities_used": modalities, "confidence_band": band, "warnings": warnings_out}


def chat_provider_name() -> str:
    if HF_TOKEN:
        return f"Hugging Face Inference API ({HF_CHAT_MODEL})"
    return "Local rule-based emotional coach"


def build_system_prompt(username: str, mood: str, tone_mode: str, twin_summary: str, intent: str, analysis_snapshot: str) -> str:
    guide = TONE_GUIDES.get(tone_mode, TONE_GUIDES["Therapist"])
    return textwrap.dedent(
        f"""
        You are an emotionally intelligent therapeutic chatbot and animated mascot.
        User name: {username}
        Current fused mood: {mood}
        Tone mode: {tone_mode}
        Tone guide: {guide}
        Detected intent: {intent}
        Digital twin summary: {twin_summary}
        Emotion snapshot: {analysis_snapshot}

        Requirements:
        1) Give one short supportive reflection.
        2) Ask exactly three important reflective questions.
        3) Suggest one practical action for the next 10-20 minutes.
        4) Avoid diagnosis and overconfident claims.
        """
    ).strip()


def huggingface_chat(system_prompt: str, user_prompt: str) -> str | None:
    if not HF_TOKEN:
        return None
    payload = {
        "inputs": f"System:\n{system_prompt}\n\nUser:\n{user_prompt}\n\nAssistant:\n",
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
        with urllib.request.urlopen(request, timeout=90) as resp:
            body = resp.read().decode("utf-8")
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


def local_coach(question: str, mood: str, tone_mode: str, twin_summary: str, intent: str) -> str:
    questions = MOOD_QUESTION_BANK.get(mood, MOOD_QUESTION_BANK["calm"])
    opener = {
        "Therapist": "I hear you, and I want to respond thoughtfully.",
        "Friendly": "I am with you, and we can handle this together.",
        "Motivational": "You are capable, and we can turn this into action right now.",
    }.get(tone_mode, "I am here for you.")
    return textwrap.dedent(
        f"""
        {opener}

        Based on your current **{mood}** state and intent around **{intent}**, here is a focused response.
        Twin summary: {twin_summary}

        Reflection:
        - Your message points to a meaningful emotional need, and that deserves attention.

        Important questions:
        1. {questions[0]}
        2. {questions[1]}
        3. {questions[2]}

        One next action:
        - {FALLBACK_ACTIONS.get(mood, FALLBACK_ACTIONS["calm"])}
        """
    ).strip()


def generate_reply(username: str, question: str, current_mood: str, tone_mode: str, analysis: dict[str, Any] | None) -> tuple[str, str]:
    text_result = analyze_text(question)
    mood = current_mood if current_mood in MOOD_CHOICES else text_result["mood"]
    twin_summary = get_twin_summary(username)
    if analysis:
        snapshot = (
            f"Face={analysis['face_result']['mood']}, Voice={analysis['voice_result']['mood']}, "
            f"Text={analysis['text_result']['mood']}, Emoji={analysis['emoji_mood']}, "
            f"Fused={analysis['fused_mood']}, Score={analysis['state_score']:.2f}"
        )
    else:
        snapshot = "No multimodal analysis available yet."
    system_prompt = build_system_prompt(
        username=username,
        mood=mood,
        tone_mode=tone_mode,
        twin_summary=twin_summary,
        intent=text_result["intent"],
        analysis_snapshot=snapshot,
    )
    user_prompt = (
        f"User question: {question}\n"
        f"Sentiment: {text_result['sentiment_label']} ({text_result['sentiment_score']:.2f})\n"
        f"Emotion: {text_result['emotion_label']} ({text_result['emotion_score']:.2f})\n"
        f"Intent: {text_result['intent']}\n"
        f"Synonym hints: {', '.join(text_result['synonyms_preview']) or 'none'}"
    )
    reply = huggingface_chat(system_prompt, user_prompt)
    if reply:
        return reply, chat_provider_name()
    return local_coach(question, mood, tone_mode, twin_summary, text_result["intent"]), chat_provider_name()


def speak_text(text: str) -> str:
    if pyttsx3 is None:
        return "pyttsx3 is not installed; text-to-speech is unavailable."
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 170)
        engine.say(text)
        engine.runAndWait()
        return "Reply spoken on the local machine."
    except Exception:
        return "Text-to-speech failed on this machine."


def render_mascot(mood: str, username: str, tone_mode: str, subtitle: str) -> None:
    require_streamlit()
    mood = mood if mood in MOOD_CHOICES else "calm"
    style = {
        "happy": {"bg": "#fff9db", "face": "#ffd866", "accent": "#ff922b", "mouth": "smile", "eye": "open"},
        "sad": {"bg": "#eef2ff", "face": "#a5b4fc", "accent": "#4c6ef5", "mouth": "sad", "eye": "soft"},
        "calm": {"bg": "#ecfeff", "face": "#99f6e4", "accent": "#0f766e", "mouth": "calm", "eye": "soft"},
        "energetic": {"bg": "#fff1f2", "face": "#fda4af", "accent": "#e11d48", "mouth": "grin", "eye": "sharp"},
    }[mood]
    safe_text = html.escape(subtitle[:180] if subtitle else f"{username}, I am tuned to your {mood} state.")
    mouth_html = {
        "smile": '<div class="cei-mouth cei-mouth-smile"></div>',
        "sad": '<div class="cei-mouth cei-mouth-sad"></div>',
        "calm": '<div class="cei-mouth cei-mouth-calm"></div>',
        "grin": '<div class="cei-mouth cei-mouth-grin"></div>',
    }[style["mouth"]]
    eye_class = {"open": "cei-eye-open", "soft": "cei-eye-soft", "sharp": "cei-eye-sharp"}[style["eye"]]
    markup = f"""
    <style>
    .cei-card {{
        background: {style["bg"]};
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 22px;
        padding: 18px;
        display: flex;
        flex-direction: column;
        gap: 14px;
        box-shadow: 0 12px 24px rgba(15, 23, 42, 0.08);
    }}
    .cei-bubble {{
        background: white;
        border-radius: 18px;
        padding: 12px 14px;
        color: #0f172a;
        font-size: 0.95rem;
        line-height: 1.5;
        box-shadow: 0 6px 14px rgba(15, 23, 42, 0.08);
    }}
    .cei-stage {{
        position: relative;
        display: flex;
        align-items: center;
        justify-content: center;
        min-height: 240px;
        overflow: hidden;
    }}
    .cei-glow {{
        position: absolute;
        width: 220px;
        height: 220px;
        background: radial-gradient(circle, {style["accent"]}33 0%, transparent 70%);
        animation: ceiPulse 2.6s ease-in-out infinite;
    }}
    .cei-avatar {{
        position: relative;
        width: 170px;
        height: 170px;
        border-radius: 999px;
        background: {style["face"]};
        animation: ceiFloat 2.4s ease-in-out infinite;
        box-shadow: 0 18px 28px rgba(15, 23, 42, 0.15);
    }}
    .cei-eye {{
        position: absolute;
        top: 66px;
        width: 18px;
        height: 18px;
        border-radius: 999px;
        background: #111827;
        animation: ceiBlink 4.2s infinite;
    }}
    .cei-eye-left {{ left: 48px; }}
    .cei-eye-right {{ right: 48px; }}
    .cei-eye-soft {{ height: 12px; top: 70px; }}
    .cei-eye-sharp {{ transform: skewX(-12deg); }}
    .cei-mouth {{
        position: absolute;
        left: 50%;
        transform: translateX(-50%);
        bottom: 42px;
    }}
    .cei-mouth-smile {{
        width: 56px;
        height: 28px;
        border-bottom: 6px solid #7f1d1d;
        border-radius: 0 0 70px 70px;
    }}
    .cei-mouth-sad {{
        width: 56px;
        height: 28px;
        border-top: 6px solid #1e293b;
        border-radius: 70px 70px 0 0;
    }}
    .cei-mouth-calm {{
        width: 44px;
        height: 0;
        border-top: 5px solid #0f172a;
        border-radius: 20px;
    }}
    .cei-mouth-grin {{
        width: 64px;
        height: 16px;
        border-bottom: 6px solid #7f1d1d;
        border-radius: 0 0 80px 80px;
    }}
    .cei-badge {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: white;
        width: fit-content;
        padding: 8px 12px;
        border-radius: 999px;
        color: #0f172a;
        font-weight: 600;
        box-shadow: 0 6px 14px rgba(15, 23, 42, 0.08);
    }}
    @keyframes ceiFloat {{
        0%, 100% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(-8px); }}
    }}
    @keyframes ceiBlink {{
        0%, 92%, 100% {{ transform: scaleY(1); }}
        94%, 96% {{ transform: scaleY(0.1); }}
    }}
    @keyframes ceiPulse {{
        0%, 100% {{ transform: scale(0.92); opacity: 0.7; }}
        50% {{ transform: scale(1.05); opacity: 1; }}
    }}
    </style>
    <div class="cei-card">
        <div class="cei-bubble">{safe_text}</div>
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
    """
    st.markdown(markup, unsafe_allow_html=True)


def render_scores(scores: dict[str, float]) -> None:
    total = sum(scores.values()) or 1.0
    for mood in MOOD_CHOICES:
        value = float(scores.get(mood, 0.0) / total)
        st.write(f"{mood.title()}: {value:.2%}")
        st.progress(min(max(value, 0.0), 1.0))


def render_item(item: dict[str, Any]) -> None:
    st.subheader(item["title"])
    st.caption(f"{item['mood'].title()} | {item['source']} | {item['resource_type']}")
    if item.get("playable") and "youtube.com/watch" in item.get("url", ""):
        st.video(item["url"])
    else:
        st.markdown(f"[Open resource]({item['url']})")
    st.caption(item.get("offline_fallback", ""))


def safe_st_image(image: Any, caption: str = "") -> None:
    try:
        st.image(image, caption=caption, width="stretch")
    except TypeError:
        # Fallback for older Streamlit versions.
        st.image(image, caption=caption, use_container_width=True)


def safe_st_dataframe(data: Any) -> None:
    try:
        st.dataframe(data, width="stretch")
    except TypeError:
        # Fallback for older Streamlit versions.
        st.dataframe(data, use_container_width=True)


def live_analysis(user: str, emoji: str, image_bytes: bytes | None, audio_bytes: bytes | None, text_input: str, tone_mode: str) -> dict[str, Any]:
    face_result = predict_face(image_bytes)
    voice_result = analyze_voice(audio_bytes, username=user)
    text_result = analyze_text(text_input)
    emoji_mood = EMOJI_MAP.get(emoji, "calm")
    fused_mood, fusion_scores, state_score = fuse_emotions(face_result, voice_result, text_result, emoji_mood)
    recommendation = recommend_item(user, fused_mood, last_n=5)
    ethics = ethical_report(face_result, voice_result, text_result, fusion_scores)
    result = {
        "face_result": face_result,
        "voice_result": voice_result,
        "text_result": text_result,
        "emoji_mood": emoji_mood,
        "fused_mood": fused_mood,
        "fusion_scores": fusion_scores,
        "state_score": state_score,
        "recommendation": recommendation,
        "ethical_report": ethics,
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


def render_setup_tab() -> None:
    st.subheader("Step-by-step VS Code execution guide")
    st.markdown(
        textwrap.dedent(
            """
            1. Create a folder and keep these files:
               - `app.py`
               - `requirements.txt`
               - `.gitignore`

            2. Create virtual environment in VS Code terminal (PowerShell):
               ```powershell
               python -m venv .venv
               .\\.venv\\Scripts\\Activate.ps1
               ```

            3. Upgrade pip and install dependencies:
               ```powershell
               python -m pip install --upgrade pip
               pip install -r requirements.txt
               ```

            4. Optional free Hugging Face API token:
               - Create a free account on huggingface.co
               - Open Settings -> Access Tokens
               - Create a **read** token
               - Set token:
               ```powershell
               $env:HF_TOKEN="your_hugging_face_token"
               ```

            5. Run the app:
               ```powershell
               streamlit run app.py
               ```

            6. Open the local URL from terminal (usually `http://localhost:8501`).
            """
        )
    )

    st.markdown("### `requirements.txt` content")
    st.code(
        "\n".join(
            [
                "streamlit",
                "torch",
                "torchvision",
                "transformers",
                "opencv-python",
                "numpy",
                "nltk",
                "SpeechRecognition",
                "pyttsx3",
            ]
        ),
        language="text",
    )

    st.markdown("### Required folder structure for training")
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

    st.markdown("### Why Hugging Face is the best free API option")
    st.markdown(
        textwrap.dedent(
            f"""
            - Free account + free read token is enough for inference usage.
            - No paid GPT key is required.
            - This app automatically falls back to local rule-based coaching if API is unavailable.
            - Active provider mode: **{chat_provider_name()}**
            """
        )
    )

    st.markdown("### Free APK conversion trick")
    st.markdown(
        textwrap.dedent(
            """
            - Run Streamlit on a public URL.
            - Open it on Android Chrome and use **Add to Home Screen**.
            - This gives a PWA-like app experience for free.
            - Later, if needed, wrap the URL with a simple WebView Android project.
            """
        )
    )

    st.info(
        "Important compatibility note: Python 3.11 is currently the safest version for PyTorch + Streamlit. "
        "Python 3.14 may work if all wheels are available for your platform, but 3.11 is recommended for stable demos."
    )


def run_cli() -> int:
    parser = argparse.ArgumentParser(description="Single-file CEI utility CLI")
    parser.add_argument("--export-catalog", action="store_true", help="Export resource catalog CSV and exit.")
    parser.add_argument("--catalog-path", default=str(CATALOG_EXPORT_PATH), help="Catalog export path.")
    args, _ = parser.parse_known_args()
    ensure_runtime_files()
    if args.export_catalog:
        out = export_catalog(Path(args.catalog_path))
        print(f"Catalog exported to {out}")
        return 0
    return -1


def main() -> None:
    cli = run_cli()
    if cli == 0:
        return
    require_streamlit()
    ensure_runtime_files()
    export_catalog()

    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)

    if "analysis_result" not in st.session_state:
        st.session_state["analysis_result"] = None
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "feedback_saved" not in st.session_state:
        st.session_state["feedback_saved"] = None
    if "last_reply" not in st.session_state:
        st.session_state["last_reply"] = ""

    with st.sidebar:
        st.header("Inputs")
        username = st.text_input("Username", value="guest_user")
        tone_mode = st.selectbox("Adaptive tone mode", options=list(TONE_GUIDES.keys()))
        emoji = st.select_slider("Emoji mood input", options=list(EMOJI_MAP.keys()), value="😊")
        speak_reply = st.checkbox("Speak chatbot reply locally", value=False)
        st.caption(f"Chat provider: {chat_provider_name()}")
        st.caption("Recommended on 8 GB RAM: batch size 4-8 and epochs 1-3.")

    tab_live, tab_chat, tab_train, tab_twin, tab_setup = st.tabs(
        ["Live Emotion Studio", "Mascot Chat", "Training + Grad-CAM", "Twin + Recommender", "Setup Guide"]
    )

    with tab_live:
        st.subheader("Multimodal emotion detection")
        c_left, c_right = st.columns([1.2, 1.0])
        with c_left:
            cam_file = st.camera_input("Capture face image")
            image_bytes = cam_file.getvalue() if cam_file is not None else None
            audio_bytes = None
            if hasattr(st, "audio_input"):
                voice_file = st.audio_input("Record voice")
                audio_bytes = voice_file.getvalue() if voice_file is not None else None
            else:
                st.info("Upgrade Streamlit to use browser microphone recording.")
            text_input = st.text_area(
                "Context text",
                height=140,
                placeholder="Type your thoughts, mood, and what help you need.",
            )
            if st.button("Analyze emotional state", type="primary"):
                with st.spinner("Analyzing using PyTorch + NLP..."):
                    st.session_state["analysis_result"] = live_analysis(
                        user=username,
                        emoji=emoji,
                        image_bytes=image_bytes,
                        audio_bytes=audio_bytes,
                        text_input=text_input,
                        tone_mode=tone_mode,
                    )
                    st.session_state["feedback_saved"] = None

        with c_right:
            result = st.session_state.get("analysis_result")
            mood = result["fused_mood"] if result else "calm"
            subtitle = (
                f"{username}, I am ready to detect your emotional state."
                if not result
                else f"{username}, your current fused mood appears {mood}."
            )
            render_mascot(mood, username, tone_mode, subtitle)

        result = st.session_state.get("analysis_result")
        if result:
            st.markdown("---")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Face mood", result["face_result"]["mood"].title())
            m2.metric("Voice mood", result["voice_result"]["mood"].title())
            m3.metric("Text mood", result["text_result"]["mood"].title())
            m4.metric("Fused mood", result["fused_mood"].title())
            st.metric("User Emotional State Score", f"{result['state_score']:.2%}")

            a_col, b_col = st.columns([1.2, 1.0])
            with a_col:
                st.markdown("#### Face analysis and Grad-CAM")
                if result["face_result"].get("overlay_rgb") is not None:
                    st.image(
                        result["face_result"]["overlay_rgb"],
                        caption=f"Method: {result['face_result']['method']}",
                        width="stretch",
                    )
                else:
                    st.info("No face image available yet.")
            with b_col:
                st.markdown("#### Fusion scores")
                render_scores(result["fusion_scores"])
                st.markdown("#### Ethical AI monitor")
                st.write(f"Confidence band: {result['ethical_report']['confidence_band']}")
                st.write(f"Modalities used: {result['ethical_report']['modalities_used']}")
                for w in result["ethical_report"]["warnings"]:
                    st.caption(f"- {w}")

            st.markdown("#### Voice transcript")
            voice = result["voice_result"]
            if voice["transcript"]:
                st.write(voice["transcript"])
                st.caption(f"Method: {voice['transcription_method']} | Energy: {voice['energy']:.3f}")
                if voice["spoken_name_detected"]:
                    st.success("Username detected in voice transcript.")
            else:
                st.info("No voice transcript captured.")

            st.markdown("#### Text intelligence")
            text_result = result["text_result"]
            st.write(
                f"Emotion: {text_result['emotion_label']} ({text_result['emotion_score']:.2f}) | "
                f"Sentiment: {text_result['sentiment_label']} ({text_result['sentiment_score']:.2f}) | "
                f"Intent: {text_result['intent']}"
            )
            if text_result["matched_terms"]:
                st.caption("Matched intent terms: " + ", ".join(text_result["matched_terms"]))
            if text_result["synonyms_preview"]:
                st.caption("NLTK synonym hints: " + ", ".join(text_result["synonyms_preview"]))

            st.markdown("#### Adaptive recommendation")
            render_item(result["recommendation"])
            like_col, skip_col = st.columns(2)
            with like_col:
                if st.button("I liked this recommendation", disabled=st.session_state["feedback_saved"] is not None):
                    upsert_feedback(result["recommendation"]["id"], "liked")
                    update_last_feedback(username, result["recommendation"]["id"], "liked")
                    st.session_state["feedback_saved"] = "liked"
                    st.success("Feedback saved.")
            with skip_col:
                if st.button("Skip recommendation", disabled=st.session_state["feedback_saved"] is not None):
                    upsert_feedback(result["recommendation"]["id"], "skipped")
                    update_last_feedback(username, result["recommendation"]["id"], "skipped")
                    st.session_state["feedback_saved"] = "skipped"
                    st.info("Feedback saved.")

    with tab_chat:
        st.subheader("Emotion-aware mascot chatbot")
        analysis = st.session_state.get("analysis_result")
        mood = analysis["fused_mood"] if analysis else "calm"
        render_mascot(
            mood,
            username,
            tone_mode,
            f"{username}, ask anything. I will respond according to your emotional state and mindset.",
        )
        for role, message in st.session_state["chat_history"]:
            with st.chat_message(role):
                st.write(message)
        prompt = st.chat_input("Ask the mascot your question...")
        if prompt:
            with st.chat_message("user"):
                st.write(prompt)
            reply, provider = generate_reply(
                username=username,
                question=prompt,
                current_mood=mood,
                tone_mode=tone_mode,
                analysis=analysis,
            )
            with st.chat_message("assistant"):
                st.write(reply)
                st.caption(f"Provider: {provider}")
            st.session_state["chat_history"].append(("user", prompt))
            st.session_state["chat_history"].append(("assistant", reply))
            st.session_state["last_reply"] = reply

            rec = analysis["recommendation"] if analysis else {"id": "", "title": "", "url": "", "source": ""}
            log_interaction(
                user=username,
                face_label=analysis["face_result"]["label"] if analysis else "neutral",
                face_mood=analysis["face_result"]["mood"] if analysis else "calm",
                voice_mood=analysis["voice_result"]["mood"] if analysis else "calm",
                text_mood=analysis["text_result"]["mood"] if analysis else simple_text_mood(prompt),
                emoji_mood=analysis["emoji_mood"] if analysis else "calm",
                fused_mood=mood,
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
            st.session_state["last_reply"] = ""
            st.success("Chat history cleared.")

    with tab_train:
        st.subheader("PyTorch EfficientNetV2-S training")
        st.caption(
            "This block expects local folders: dataset/train, dataset/val, dataset/test with class subfolders."
        )
        summary = dataset_summary(DATASET_DIR)
        if summary:
            st.table(summary)
        else:
            st.info("No local dataset found yet. Use the folder structure shown in Setup Guide.")

        batch_size = st.select_slider("Batch size", options=[4, 8, 12, 16], value=8)
        epochs = st.slider("Epochs", min_value=1, max_value=6, value=2)
        learning_rate = st.select_slider("Learning rate", options=[1e-4, 2e-4, 5e-4, 1e-3], value=2e-4)
        freeze_backbone = st.checkbox("Freeze backbone for lightweight training", value=True)
        if st.button("Train EfficientNetV2-S"):
            progress = st.progress(0.0)
            history_box = st.empty()

            def callback(fraction: float, payload: dict[str, float]):
                progress.progress(float(fraction))
                history_box.write(payload)

            try:
                with st.spinner("Training model..."):
                    history, evaluation, classes = train_model(
                        dataset_root=DATASET_DIR,
                        batch_size=batch_size,
                        epochs=epochs,
                        learning_rate=float(learning_rate),
                        freeze_backbone=freeze_backbone,
                        progress_callback=callback,
                    )
                st.success(f"Training complete. Model saved: {MODEL_PATH.name}")
                st.write("Classes: " + ", ".join(classes))
                st.table(history)
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
        st.subheader("Digital Emotional Twin + RL recommender stats")
        twin_rows = read_rows(TWIN_LOG_PATH)
        if twin_rows:
            st.dataframe(twin_rows, width="stretch")
            st.download_button(
                "Download twin log CSV",
                TWIN_LOG_PATH.read_bytes(),
                file_name=TWIN_LOG_PATH.name,
                mime="text/csv",
            )
        else:
            st.info("No interaction logs yet.")

        stats = get_recommender_stats()
        st.markdown("#### Recommender memory")
        if stats:
            st.dataframe(stats, width="stretch")
        else:
            st.info("No recommender stats yet.")

        st.markdown("#### Resource catalog")
        all_items = catalog_rows()
        mood_filter = st.selectbox("Mood filter", options=["all"] + MOOD_CHOICES, index=0)
        source_filter = st.selectbox("Source filter", options=["all", "YouTube", "Spotify", "YouTube Music"], index=0)
        filtered = []
        for item in all_items:
            if mood_filter != "all" and item["mood"] != mood_filter:
                continue
            if source_filter != "all" and item["source"] != source_filter:
                continue
            filtered.append(item)
        st.write(f"Catalog items: {len(filtered)} / {len(all_items)}")
        st.dataframe(filtered, width="stretch")
        st.download_button(
            "Download resource catalog CSV",
            CATALOG_EXPORT_PATH.read_bytes(),
            file_name=CATALOG_EXPORT_PATH.name,
            mime="text/csv",
        )

    with tab_setup:
        render_setup_tab()


if __name__ == "__main__":
    main()