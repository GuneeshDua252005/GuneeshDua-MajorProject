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

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

try:
    import nltk
    from nltk.corpus import wordnet as wn
except Exception:
    nltk = None
    wn = None

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
    from torchvision import datasets as tv_datasets
    from torchvision import transforms
    from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s

    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    DataLoader = None
    tv_datasets = None
    transforms = None
    EfficientNet_V2_S_Weights = None
    efficientnet_v2_s = None
    TORCH_AVAILABLE = False


APP_TITLE = "Emotionally Intelligent Animated Mascot Chatbot"
APP_SUBTITLE = (
    "PyTorch EfficientNetV2-S, Grad-CAM, voice/text/face emotion fusion, "
    "Digital Emotional Twin logging, and free Hugging Face API support."
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
LOCAL_TRANSFORMERS_ENABLED = os.getenv("CEI_USE_LOCAL_TRANSFORMERS", "0").strip() == "1"
TEXT_EMOTION_MODEL = os.getenv("TEXT_EMOTION_MODEL", "j-hartmann/emotion-english-distilroberta-base").strip()
LOCAL_CHAT_MODEL = os.getenv("LOCAL_CHAT_MODEL", "google/flan-t5-small").strip()

MOOD_CHOICES = ["happy", "sad", "calm", "energetic"]
DEFAULT_FACE_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
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
    "joy": "happy",
    "happy": "happy",
    "love": "happy",
    "gratitude": "happy",
    "surprise": "energetic",
    "excitement": "energetic",
    "anger": "energetic",
    "angry": "energetic",
    "frustration": "energetic",
    "fear": "sad",
    "sad": "sad",
    "sadness": "sad",
    "lonely": "sad",
    "neutral": "calm",
    "calm": "calm",
    "peace": "calm",
}

FALLBACK_ACTIONS = {
    "happy": "Write one win, one gratitude, and one next goal while this positive state is fresh.",
    "sad": "Take two slow breaths, drink water, and choose one tiny step that makes the next hour easier.",
    "calm": "Protect this balanced state with one low-distraction task for the next 20 minutes.",
    "energetic": "Channel the intensity into movement or one focused sprint, then pause before reacting.",
}

MOOD_QUESTION_BANK = {
    "happy": [
        "What exactly created this positive shift today?",
        "Which small action can you repeat to protect this mood?",
        "Who can you share this progress with in a simple way?",
    ],
    "sad": [
        "What feels heaviest right now?",
        "What helped even slightly the last time you felt this way?",
        "What one small action would make the next hour easier?",
    ],
    "calm": [
        "What is helping you stay balanced right now?",
        "Which routine is protecting your focus today?",
        "What should you avoid so this calm does not get disturbed?",
    ],
    "energetic": [
        "Is this energy helping progress or creating overload?",
        "Where can you channel this intensity most productively?",
        "What boundary would keep this energy healthy?",
    ],
}

TONE_GUIDES = {
    "Therapist": "warm, grounded, reflective, structured, non-clinical",
    "Friendly": "natural, caring, conversational, practical",
    "Motivational": "focused, energetic, action-oriented, respectful",
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
    "happy": ["feel good playlist", "gratitude meditation", "confidence booster", "upbeat study break"],
    "sad": ["comfort songs", "self compassion meditation", "gentle piano", "mindfulness for low mood"],
    "calm": ["lofi focus", "rain sounds", "box breathing", "peaceful instrumental"],
    "energetic": ["workout hits", "motivation speech", "focus sprint music", "power walk music"],
}

SPOTIFY_QUERIES = {
    "happy": ["happy playlist", "feel good hits", "good vibes only", "sunshine acoustic"],
    "sad": ["comfort songs", "gentle piano", "self care songs", "healing ambient music"],
    "calm": ["lofi beats", "deep focus", "peaceful piano", "meditation music"],
    "energetic": ["workout hits", "high energy mix", "running playlist", "motivation songs"],
}


def require_streamlit() -> None:
    if st is None:
        raise RuntimeError("Streamlit is not installed. Run: pip install -r requirements.txt")


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
            entries.append(
                CatalogEntry(
                    id=f"{mood}_yt_search_{index}",
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
        for index, query in enumerate(SPOTIFY_QUERIES[mood], start=1):
            entries.append(
                CatalogEntry(
                    id=f"{mood}_spotify_{index}",
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
    return [
        row.get("RecommendedId", "")
        for row in read_csv_rows(TWIN_LOG_PATH)
        if row.get("User") == user and row.get("RecommendedId")
    ]


def recommend_resource(user: str, mood: str, last_n: int = 5) -> dict[str, Any]:
    catalog = catalog_rows()
    candidates = [row for row in catalog if row["mood"] == mood] or catalog
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
        recent_penalty = 0.50 if item["id"] in recent_ids else 0.0
        score = 0.45 * like_ratio + 0.35 * novelty_bonus - 0.20 * skip_ratio - recent_penalty + random.uniform(0, 0.05)
        scored.append((score, item))
    pool = [(score, item) for score, item in scored if item["id"] not in recent_ids] or scored
    pool.sort(key=lambda pair: pair[0], reverse=True)
    chosen = random.choice([item for _, item in pool[: min(5, len(pool))]])
    upsert_recommender_feedback(chosen["id"], feedback="shown")
    return chosen


def normalize_label_to_mood(label: str) -> str:
    token = slugify(label).replace("_", " ")
    for emotion, mood in EMOTION_TO_MOOD.items():
        if emotion in token:
            return mood
    return "calm"


def tokenize_words(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


@lru_cache(maxsize=128)
def synonyms_for_word(word: str) -> tuple[str, ...]:
    if nltk is None or wn is None:
        return tuple()
    for resource in ["wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                return tuple()
    results: list[str] = []
    try:
        for synset in wn.synsets(word):
            for lemma in synset.lemmas():
                candidate = lemma.name().replace("_", " ").lower().strip()
                if candidate and candidate != word.lower() and candidate not in results:
                    results.append(candidate)
                if len(results) >= 6:
                    return tuple(results)
    except Exception:
        return tuple()
    return tuple(results)


INTENT_SEEDS = {
    "stress_relief": {"stress", "overwhelmed", "pressure", "tense", "burnout", "anxious"},
    "motivation": {"motivation", "discipline", "goal", "progress", "improve"},
    "study_focus": {"study", "exam", "assignment", "focus", "college", "project"},
    "loneliness": {"alone", "lonely", "isolated", "empty"},
    "confidence": {"confidence", "nervous", "presentation", "interview", "fear"},
}


def infer_user_intent(text: str) -> dict[str, Any]:
    tokens = set(tokenize_words(text))
    best_intent = "general_support"
    best_matches: list[str] = []
    best_count = 0
    for intent, seeds in INTENT_SEEDS.items():
        expanded = set(seeds)
        for seed in seeds:
            expanded.update(tokenize_words(" ".join(synonyms_for_word(seed))))
        matches = sorted(token for token in tokens if token in expanded)
        if len(matches) > best_count:
            best_intent = intent
            best_matches = matches
            best_count = len(matches)
    preview = sorted({syn for token in list(tokens)[:6] for syn in synonyms_for_word(token)[:2]})[:8]
    return {"intent": best_intent, "matched_terms": best_matches, "synonyms_preview": preview}


def simple_text_mood(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ["sad", "cry", "lonely", "down", "grief", "hurt"]):
        return "sad"
    if any(token in lowered for token in ["happy", "joy", "great", "awesome", "love", "good"]):
        return "happy"
    if any(token in lowered for token in ["angry", "mad", "stressed", "frustrated", "furious", "rage"]):
        return "energetic"
    return "calm"


@lru_cache(maxsize=1)
def get_text_emotion_pipeline():
    if not LOCAL_TRANSFORMERS_ENABLED:
        return None
    try:
        from transformers import logging as hf_logging
        from transformers import pipeline

        hf_logging.set_verbosity_error()
        return pipeline("text-classification", model=TEXT_EMOTION_MODEL, device=-1, top_k=None)
    except Exception:
        return None


@lru_cache(maxsize=1)
def get_local_chat_pipeline():
    if not LOCAL_TRANSFORMERS_ENABLED:
        return None
    try:
        from transformers import logging as hf_logging
        from transformers import pipeline

        hf_logging.set_verbosity_error()
        return pipeline("text2text-generation", model=LOCAL_CHAT_MODEL, device=-1)
    except Exception:
        return None


def analyze_text_emotion(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    intent = infer_user_intent(text)
    if not text:
        return {
            "text": "",
            "emotion_label": "neutral",
            "emotion_score": 0.50,
            "mood": "calm",
            "confidence": 0.50,
            "intent": intent["intent"],
            "matched_terms": intent["matched_terms"],
            "synonyms_preview": intent["synonyms_preview"],
            "mood_scores": {mood: 0.0 for mood in MOOD_CHOICES},
            "method": "empty_text",
        }
    mood_scores = {mood: 0.0 for mood in MOOD_CHOICES}
    emotion_label = simple_text_mood(text)
    emotion_score = 0.60
    method = "heuristic"
    emotion_pipeline = get_text_emotion_pipeline()
    if emotion_pipeline is not None:
        try:
            result = emotion_pipeline(text[:512], truncation=True)
            items = result[0] if result and isinstance(result[0], list) else result
            if items:
                top = max(items, key=lambda item: float(item.get("score", 0)))
                emotion_label = str(top.get("label", emotion_label))
                emotion_score = float(top.get("score", emotion_score))
                for item in items:
                    mood_scores[normalize_label_to_mood(str(item.get("label", "neutral")))] += float(item.get("score", 0))
                method = "local_transformers"
        except Exception:
            pass
    if max(mood_scores.values()) <= 0:
        mood_scores[simple_text_mood(text)] = 1.0
    mood = max(mood_scores.items(), key=lambda item: item[1])[0]
    return {
        "text": text,
        "emotion_label": emotion_label,
        "emotion_score": emotion_score,
        "mood": mood,
        "confidence": max(float(max(mood_scores.values())), emotion_score),
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
        return "", "speechrecognition_missing"
    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio = recognizer.record(source)
        return recognizer.recognize_google(audio).strip(), "google_web_speech"
    except Exception:
        return "", "transcription_failed"


def estimate_wav_energy(audio_bytes: bytes | None) -> float:
    if not audio_bytes:
        return 0.0
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            width = wav_file.getsampwidth()
            channels = wav_file.getnchannels()
        dtype = np.uint8 if width == 1 else np.int16 if width == 2 else np.int32
        data = np.frombuffer(frames, dtype=dtype).astype(np.float32)
        if width == 1:
            data -= 128.0
            max_abs = 128.0
        elif width == 2:
            max_abs = 32768.0
        else:
            max_abs = float(2**31)
        if channels > 1 and len(data) % channels == 0:
            data = data.reshape(-1, channels).mean(axis=1)
        return float(np.mean(np.abs(data)) / max_abs) if data.size else 0.0
    except Exception:
        return 0.0


def analyze_voice_emotion(audio_bytes: bytes | None, username: str) -> dict[str, Any]:
    transcript, method = transcribe_audio_bytes(audio_bytes)
    text_result = analyze_text_emotion(transcript) if transcript else analyze_text_emotion("")
    energy = estimate_wav_energy(audio_bytes)
    scores = dict(text_result["mood_scores"])
    if energy > 0.14:
        scores["energetic"] += 0.15
    if energy < 0.04:
        scores["calm"] += 0.08
    mood = max(scores.items(), key=lambda item: item[1])[0]
    return {
        "transcript": transcript,
        "transcription_method": method,
        "energy": energy,
        "mood": mood,
        "confidence": float(max(scores.values()) if scores else 0.0),
        "spoken_name_detected": bool(username and transcript and username.lower() in transcript.lower()),
    }


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


def decode_image_bytes(image_bytes: bytes | None) -> np.ndarray | None:
    if not image_bytes or cv2 is None:
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
    return tuple(map(int, max(faces, key=lambda box: int(box[2] * box[3]))))


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


def imagenet_mean_std() -> tuple[list[float], list[float]]:
    if not TORCH_AVAILABLE:
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    try:
        weights = EfficientNet_V2_S_Weights.DEFAULT
        return list(weights.transforms().mean), list(weights.transforms().std)
    except Exception:
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def build_efficientnet_model(num_classes: int, pretrained: bool = True):
    if not TORCH_AVAILABLE:
        return None
    weights = None
    if pretrained:
        try:
            weights = EfficientNet_V2_S_Weights.DEFAULT
        except Exception:
            weights = None
    model = efficientnet_v2_s(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(p=0.30), nn.Linear(in_features, num_classes))
    return model


@lru_cache(maxsize=1)
def load_saved_model():
    if not (TORCH_AVAILABLE and MODEL_PATH.exists()):
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
    if not TORCH_AVAILABLE:
        return None
    mean, std = imagenet_mean_std()
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def get_last_conv_module(model):
    last_name = None
    last_module = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_name = name
            last_module = module
    return last_name, last_module


def compute_gradcam(model, image_tensor, target_index: int | None = None) -> np.ndarray | None:
    if not TORCH_AVAILABLE:
        return None
    _, target_module = get_last_conv_module(model)
    if target_module is None:
        return None
    activations: list[Any] = []
    gradients: list[Any] = []

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
        return {"label": "neutral", "mood": "calm", "confidence": 0.0, "method": "no_image", "overlay_rgb": None, "face_found": False}
    image_bgr = decode_image_bytes(image_bytes)
    if image_bgr is None or cv2 is None:
        return {"label": "neutral", "mood": "calm", "confidence": 0.0, "method": "opencv_missing", "overlay_rgb": None, "face_found": False}
    face_box = detect_largest_face(image_bgr)
    display_bgr = image_bgr.copy()
    if face_box is None:
        x, y, w, h = 0, 0, image_bgr.shape[1], image_bgr.shape[0]
    else:
        x, y, w, h = face_box
        cv2.rectangle(display_bgr, (x, y), (x + w, y + h), (70, 255, 140), 2)
    face_crop = image_bgr[y : y + h, x : x + w]
    model, metadata = load_saved_model()
    if model is not None:
        try:
            transform = get_inference_transform()
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            tensor = transform(face_rgb).unsqueeze(0)
            logits = model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
            index = int(np.argmax(probabilities))
            class_names = metadata.get("class_names") or DEFAULT_FACE_CLASSES
            label = class_names[index] if index < len(class_names) else f"class_{index}"
            heatmap = compute_gradcam(model, tensor, index)
            overlay = overlay_heatmap(face_crop, heatmap)
            if overlay is not None:
                display_bgr[y : y + h, x : x + w] = cv2.resize(overlay, (w, h))
            return {
                "label": str(label),
                "mood": normalize_label_to_mood(str(label)),
                "confidence": float(probabilities[index]),
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
    return split_dir.exists() and any(path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS for path in split_dir.rglob("*"))


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
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch and torchvision are required for training.")
    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"
    test_dir = dataset_root / "test"
    if not dataset_has_images(train_dir):
        raise RuntimeError("No training images found under dataset/train.")
    mean, std = imagenet_mean_std()
    train_transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(), transforms.RandomRotation(8), transforms.ToTensor(), transforms.Normalize(mean, std)]
    )
    eval_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_ds = tv_datasets.ImageFolder(train_dir, transform=train_transform)
    class_names = list(train_ds.classes)
    val_ds = tv_datasets.ImageFolder(val_dir, transform=eval_transform) if dataset_has_images(val_dir) else None
    test_ds = tv_datasets.ImageFolder(test_dir, transform=eval_transform) if dataset_has_images(test_dir) else None
    kwargs = {"batch_size": batch_size, "num_workers": 0, "pin_memory": torch.cuda.is_available()}
    return (
        DataLoader(train_ds, shuffle=True, **kwargs),
        DataLoader(val_ds, shuffle=False, **kwargs) if val_ds is not None else None,
        DataLoader(test_ds, shuffle=False, **kwargs) if test_ds is not None else None,
        class_names,
    )


def save_model_checkpoint(model, class_names: list[str], extra: dict[str, Any] | None = None) -> None:
    payload = {"architecture": "efficientnet_v2_s", "class_names": class_names, "model_state_dict": model.state_dict(), "saved_at_utc": utc_now()}
    if extra:
        payload.update(extra)
    torch.save(payload, MODEL_PATH)
    metadata = {key: value for key, value in payload.items() if key != "model_state_dict"}
    MODEL_META_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


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
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * labels.size(0)
            total_correct += int((torch.argmax(logits, dim=1) == labels).sum().item())
            total_samples += int(labels.size(0))
        row = {
            "epoch": float(epoch),
            "train_loss": round(total_loss / max(total_samples, 1), 4),
            "train_accuracy": round(total_correct / max(total_samples, 1), 4),
            "val_accuracy": 0.0,
        }
        if val_loader is not None:
            model.eval()
            correct = 0
            samples = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    logits = model(images)
                    correct += int((torch.argmax(logits, dim=1) == labels).sum().item())
                    samples += int(labels.size(0))
            row["val_accuracy"] = round(correct / max(samples, 1), 4)
        history.append(row)
        if progress_callback:
            progress_callback(epoch / epochs, row)
    model.cpu()
    save_model_checkpoint(model, class_names, {"epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate, "freeze_backbone": freeze_backbone})
    load_saved_model.cache_clear()
    return history, class_names


def emotion_fusion(face_result: dict[str, Any], voice_result: dict[str, Any], text_result: dict[str, Any], emoji_mood: str) -> tuple[str, dict[str, float], float]:
    scores = {mood: 0.0 for mood in MOOD_CHOICES}
    weights = {"face": 0.35, "voice": 0.25, "text": 0.25, "emoji": 0.15}
    scores[face_result.get("mood", "calm")] += weights["face"] * max(float(face_result.get("confidence") or 0.3), 0.3)
    scores[voice_result.get("mood", "calm")] += weights["voice"] * max(float(voice_result.get("confidence") or 0.3), 0.3)
    scores[text_result.get("mood", "calm")] += weights["text"] * max(float(text_result.get("confidence") or 0.3), 0.3)
    scores[emoji_mood] += weights["emoji"]
    fused = max(scores.items(), key=lambda item: item[1])[0]
    return fused, scores, scores[fused] / max(sum(scores.values()), 1e-9)


def ethical_monitor(face_result: dict[str, Any], voice_result: dict[str, Any], text_result: dict[str, Any], scores: dict[str, float]) -> dict[str, Any]:
    used = int(face_result.get("method") != "no_image") + int(bool(voice_result.get("transcript"))) + int(bool(text_result.get("text")))
    warnings_list = ["This app gives supportive guidance, not a medical diagnosis."]
    if face_result.get("method") == "opencv_heuristic":
        warnings_list.append("Facial emotion is using the safe OpenCV fallback because no trained .pth model is loaded.")
    if used < 2:
        warnings_list.append("Use at least two inputs for stronger multimodal confidence.")
    dominant = max(scores.values()) if scores else 0.0
    band = "High" if dominant >= 0.45 else "Medium" if dominant >= 0.28 else "Low"
    return {"modalities_used": used, "confidence_band": band, "warnings": warnings_list}


def get_user_twin_summary(user: str, limit: int = 12) -> str:
    rows = [row for row in read_csv_rows(TWIN_LOG_PATH) if row.get("User") == user]
    if not rows:
        return "No Digital Emotional Twin history is available yet."
    recent = rows[-limit:]
    moods = [row.get("FusedMood", "calm") for row in recent]
    dominant = Counter(moods).most_common(1)[0][0] if moods else "calm"
    last_title = next((row.get("RecommendedTitle", "") for row in reversed(recent) if row.get("RecommendedTitle")), "not available")
    return f"Dominant recent mood: {dominant}. Recent moods: {', '.join(moods[-5:])}. Last recommendation: {last_title}."


def huggingface_free_chat(system_prompt: str, user_prompt: str) -> str | None:
    if not HF_TOKEN:
        return None
    payload = {
        "inputs": f"Instruction:\n{system_prompt}\n\nUser:\n{user_prompt}\n\nAssistant:",
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
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict) and first.get("generated_text"):
            return str(first["generated_text"]).strip()
    if isinstance(data, dict) and data.get("generated_text"):
        return str(data["generated_text"]).strip()
    return None


def local_transformers_chat(system_prompt: str, user_prompt: str) -> str | None:
    generator = get_local_chat_pipeline()
    if generator is None:
        return None
    try:
        result = generator(f"{system_prompt}\n\n{user_prompt}", max_new_tokens=220, do_sample=True, temperature=0.7)[0]
        return str(result.get("generated_text", "")).strip() or None
    except Exception:
        return None


def local_rule_based_response(question: str, mood: str, tone_mode: str, twin_summary: str, intent: str) -> str:
    questions = MOOD_QUESTION_BANK.get(mood, MOOD_QUESTION_BANK["calm"])
    intro = {
        "Therapist": "I hear you, and I will keep this grounded and practical.",
        "Friendly": "I am with you. Let us make this feel a little clearer.",
        "Motivational": "You can work with this state instead of fighting it.",
    }.get(tone_mode, "I am here with you.")
    return textwrap.dedent(
        f"""
        {intro}

        Reflection:
        Your message connects with a {mood} emotional state and the main intent looks like {intent}.

        Important questions:
        1. {questions[0]}
        2. {questions[1]}
        3. {questions[2]}

        One next action:
        {FALLBACK_ACTIONS.get(mood, FALLBACK_ACTIONS["calm"])}

        Digital twin context:
        {twin_summary}
        """
    ).strip()


def get_chat_provider_name() -> str:
    if HF_TOKEN:
        return f"Hugging Face Inference API ({HF_CHAT_MODEL})"
    if LOCAL_TRANSFORMERS_ENABLED:
        return f"Local Transformers ({LOCAL_CHAT_MODEL})"
    return "Local rule-based emotional coach"


def generate_chat_response(username: str, question: str, current_mood: str, tone_mode: str, analysis: dict[str, Any] | None) -> tuple[str, str]:
    text_result = analyze_text_emotion(question)
    mood = current_mood if current_mood in MOOD_CHOICES else text_result["mood"]
    twin_summary = get_user_twin_summary(username)
    snapshot = "No multimodal snapshot yet."
    if analysis:
        snapshot = (
            f"Face={analysis['face_result']['mood']}, Voice={analysis['voice_result']['mood']}, "
            f"Text={analysis['text_result']['mood']}, Emoji={analysis['emoji_mood']}, Fused={analysis['fused_mood']}"
        )
    system_prompt = (
        f"You are a therapeutic animated mascot chatbot. User={username}. Mood={mood}. "
        f"Tone={tone_mode} ({TONE_GUIDES.get(tone_mode)}). Intent={text_result['intent']}. "
        f"Twin={twin_summary}. Snapshot={snapshot}. Ask exactly three important questions and suggest one action. "
        "Do not diagnose, do not mention paid APIs, and do not sound like a repeated FAQ bot."
    )
    user_prompt = f"User question: {question}"
    response = huggingface_free_chat(system_prompt, user_prompt) or local_transformers_chat(system_prompt, user_prompt)
    if response:
        return response, get_chat_provider_name()
    return local_rule_based_response(question, mood, tone_mode, twin_summary, text_result["intent"]), get_chat_provider_name()


def speak_text_locally(text: str) -> str:
    if pyttsx3 is None:
        return "pyttsx3 is not installed, so text-to-speech is unavailable."
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 170)
        engine.say(text)
        engine.runAndWait()
        return "Reply spoken locally."
    except Exception:
        return "Local text-to-speech could not start on this device."


def render_mascot(mood: str, username: str, tone_mode: str, subtitle: str) -> None:
    mood = mood if mood in MOOD_CHOICES else "calm"
    palette = {
        "happy": {"bg": "#fff9db", "face": "#ffd866", "accent": "#ff922b", "mouth": "smile"},
        "sad": {"bg": "#eef2ff", "face": "#a5b4fc", "accent": "#4c6ef5", "mouth": "sad"},
        "calm": {"bg": "#ecfeff", "face": "#99f6e4", "accent": "#0f766e", "mouth": "calm"},
        "energetic": {"bg": "#fff1f2", "face": "#fda4af", "accent": "#e11d48", "mouth": "grin"},
    }[mood]
    mouth_css = {
        "smile": "border-bottom: 6px solid #7f1d1d; border-radius: 0 0 70px 70px;",
        "sad": "border-top: 6px solid #1e293b; border-radius: 70px 70px 0 0;",
        "calm": "height: 0; border-top: 5px solid #0f172a; border-radius: 20px;",
        "grin": "border-bottom: 7px solid #7f1d1d; border-radius: 0 0 80px 80px;",
    }[palette["mouth"]]
    st.markdown(
        f"""
        <style>
        .cei-card {{background:{palette['bg']}; border-radius:22px; padding:18px; box-shadow:0 12px 24px rgba(15,23,42,.08);}}
        .cei-bubble {{background:white; border-radius:18px; padding:12px 14px; color:#0f172a; line-height:1.45;}}
        .cei-stage {{position:relative; display:flex; justify-content:center; align-items:center; min-height:230px; overflow:hidden;}}
        .cei-glow {{position:absolute; width:220px; height:220px; background:radial-gradient(circle,{palette['accent']}44 0%,transparent 70%); animation:pulse 2.4s ease-in-out infinite;}}
        .cei-avatar {{position:relative; width:170px; height:170px; border-radius:999px; background:{palette['face']}; animation:float 2.2s ease-in-out infinite; box-shadow:0 18px 28px rgba(15,23,42,.15);}}
        .cei-eye {{position:absolute; top:64px; width:18px; height:18px; border-radius:999px; background:#111827; animation:blink 4s infinite;}}
        .cei-left {{left:46px;}} .cei-right {{right:46px;}}
        .cei-mouth {{position:absolute; left:50%; transform:translateX(-50%); bottom:42px; width:58px; height:26px; {mouth_css}}}
        .cei-badge {{display:inline-block; background:white; border-radius:999px; padding:8px 12px; color:#0f172a; font-weight:600;}}
        @keyframes float {{0%,100%{{transform:translateY(0)}}50%{{transform:translateY(-8px)}}}}
        @keyframes blink {{0%,92%,100%{{transform:scaleY(1)}}94%,96%{{transform:scaleY(.1)}}}}
        @keyframes pulse {{0%,100%{{transform:scale(.92);opacity:.7}}50%{{transform:scale(1.05);opacity:1}}}}
        </style>
        <div class="cei-card">
          <div class="cei-bubble">{html.escape(subtitle[:220])}</div>
          <div class="cei-stage"><div class="cei-glow"></div><div class="cei-avatar">
            <div class="cei-eye cei-left"></div><div class="cei-eye cei-right"></div><div class="cei-mouth"></div>
          </div></div>
          <div class="cei-badge">Mascot: {html.escape(tone_mode)} | {html.escape(username)} | {html.escape(mood.title())}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def safe_image(image: Any, caption: str = "") -> None:
    st.image(image, caption=caption, width="stretch")


def render_mood_scores(scores: dict[str, float]) -> None:
    total = sum(scores.values()) or 1.0
    for mood in MOOD_CHOICES:
        value = float(scores.get(mood, 0.0) / total)
        st.write(f"{mood.title()}: {value:.1%}")
        st.progress(min(max(value, 0.0), 1.0))


def render_recommendation_card(item: dict[str, Any]) -> None:
    st.subheader(item["title"])
    st.caption(f"{item['mood'].title()} | {item['source']} | {item['resource_type']}")
    if item.get("playable") and "youtube.com/watch" in item.get("url", ""):
        st.video(item["url"])
    else:
        st.markdown(f"[Open resource]({item['url']})")
    st.caption(item.get("offline_fallback", ""))


def log_interaction(user: str, face_label: str, face_mood: str, voice_mood: str, text_mood: str, emoji_mood: str, fused_mood: str, state_score: float, tone_mode: str, recommendation: dict[str, Any], chat_query: str = "", chat_reply: str = "", feedback: str = "") -> None:
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


def live_analysis(user: str, emoji: str, image_bytes: bytes | None, audio_bytes: bytes | None, text_input: str, tone_mode: str) -> dict[str, Any]:
    face_result = predict_face_emotion(image_bytes)
    voice_result = analyze_voice_emotion(audio_bytes, username=user)
    text_result = analyze_text_emotion(text_input)
    emoji_mood = EMOJI_MAP.get(emoji, "calm")
    fused_mood, fusion_scores, state_score = emotion_fusion(face_result, voice_result, text_result, emoji_mood)
    recommendation = recommend_resource(user, fused_mood, last_n=5)
    result = {
        "face_result": face_result,
        "voice_result": voice_result,
        "text_result": text_result,
        "emoji_mood": emoji_mood,
        "fused_mood": fused_mood,
        "fusion_scores": fusion_scores,
        "state_score": state_score,
        "recommendation": recommendation,
        "ethical_report": ethical_monitor(face_result, voice_result, text_result, fusion_scores),
        "tone_mode": tone_mode,
    }
    log_interaction(
        user=user,
        face_label=str(face_result["label"]),
        face_mood=str(face_result["mood"]),
        voice_mood=str(voice_result["mood"]),
        text_mood=str(text_result["mood"]),
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
            1. Create a project folder and put these files inside it:
               - `app.py`
               - `requirements.txt`
               - `.gitignore`

            2. Create and activate a virtual environment in VS Code PowerShell:
               ```powershell
               py -3.11 -m venv .venv
               .\\.venv\\Scripts\\Activate.ps1
               ```

            3. Install only the essential packages:
               ```powershell
               python -m pip install --upgrade pip
               pip install -r requirements.txt
               ```

            4. Optional free Hugging Face token:
               - Create a free account at `huggingface.co`
               - Go to Settings -> Access Tokens
               - Create a read token
               - Set it before running:
               ```powershell
               $env:HF_TOKEN="your_hugging_face_token"
               ```

            5. Run the app:
               ```powershell
               streamlit run app.py
               ```

            Note: Python 3.11 is recommended for reliable PyTorch wheels on Windows. If Python 3.14 does not have a PyTorch wheel on your machine, use Python 3.11.9 for the virtual environment.
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
    st.markdown("### Training dataset folder structure")
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
    st.markdown("### Why Hugging Face is used")
    st.write(
        "Hugging Face is the best free API option here because it provides free read tokens, many open models, and a safe fallback path. "
        "This app uses the Hugging Face HTTP API when `HF_TOKEN` is set, and otherwise uses a local rule-based emotional coach."
    )
    st.markdown("### Free APK conversion trick")
    st.write(
        "For a free Android demo, host or run Streamlit, open the URL in Chrome on Android, and choose Add to Home Screen. "
        "For a real APK later, wrap the Streamlit URL in an Android WebView project."
    )


def run_cli() -> int:
    parser = argparse.ArgumentParser(description="CEI mascot utility")
    parser.add_argument("--export-catalog", action="store_true")
    parser.add_argument("--catalog-path", default=str(CATALOG_EXPORT_PATH))
    args, _ = parser.parse_known_args()
    ensure_runtime_files()
    if args.export_catalog:
        print(f"Catalog exported to: {export_resource_catalog_csv(Path(args.catalog_path))}")
        return 0
    return -1


def main() -> None:
    if run_cli() == 0:
        return
    require_streamlit()
    ensure_runtime_files()
    export_resource_catalog_csv()

    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)
    st.info("This app is PyTorch-only and does not require any legacy `.h5` model file.")

    if "analysis_result" not in st.session_state:
        st.session_state["analysis_result"] = None
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "feedback_saved" not in st.session_state:
        st.session_state["feedback_saved"] = None

    with st.sidebar:
        st.header("Controls")
        username = st.text_input("Username", value="guest_user")
        tone_mode = st.selectbox("Adaptive tone", list(TONE_GUIDES.keys()), index=0)
        emoji = st.select_slider("Emoji mood input", options=list(EMOJI_MAP.keys()), value="😊")
        speak_reply = st.checkbox("Speak chatbot reply locally", value=False)
        st.caption(f"Chat provider: {get_chat_provider_name()}")
        st.caption(f"PyTorch available: {'yes' if TORCH_AVAILABLE else 'no'}")

    tab_live, tab_chat, tab_train, tab_twin, tab_setup = st.tabs(
        ["Live Emotion Studio", "Mascot Chatbot", "Training + Grad-CAM", "Digital Twin + RL", "Setup + Viva Guide"]
    )

    with tab_live:
        st.subheader("Multimodal emotion detection")
        left, right = st.columns([1.2, 1.0])
        with left:
            camera_file = st.camera_input("Capture face image")
            image_bytes = camera_file.getvalue() if camera_file is not None else None
            audio_bytes = None
            if hasattr(st, "audio_input"):
                voice_file = st.audio_input("Record voice")
                audio_bytes = voice_file.getvalue() if voice_file is not None else None
            else:
                st.warning("Update Streamlit to use browser microphone input.")
            text_input = st.text_area("Context text", height=140, placeholder="Type how you feel or what happened.")
            if st.button("Analyze emotional state", type="primary"):
                with st.spinner("Analyzing face, voice, text, and emoji signals..."):
                    st.session_state["analysis_result"] = live_analysis(username, emoji, image_bytes, audio_bytes, text_input, tone_mode)
                    st.session_state["feedback_saved"] = None
        with right:
            result = st.session_state["analysis_result"]
            active_mood = result["fused_mood"] if result else "calm"
            subtitle = f"{username}, your mascot is ready." if not result else f"{username}, your fused state looks {active_mood}."
            render_mascot(active_mood, username, tone_mode, subtitle)

        result = st.session_state["analysis_result"]
        if result:
            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Face", result["face_result"]["mood"].title())
            c2.metric("Voice", result["voice_result"]["mood"].title())
            c3.metric("Text", result["text_result"]["mood"].title())
            c4.metric("Fused", result["fused_mood"].title())
            st.metric("User Emotional State Score", f"{result['state_score']:.1%}")
            col_a, col_b = st.columns([1.2, 1.0])
            with col_a:
                st.markdown("#### Face analysis / Grad-CAM")
                if result["face_result"].get("overlay_rgb") is not None:
                    safe_image(result["face_result"]["overlay_rgb"], caption=f"Method: {result['face_result']['method']}")
                else:
                    st.info("No face image was supplied.")
            with col_b:
                st.markdown("#### Mood fusion")
                render_mood_scores(result["fusion_scores"])
                st.markdown("#### Ethical AI monitor")
                st.write(f"Confidence band: {result['ethical_report']['confidence_band']}")
                st.write(f"Modalities used: {result['ethical_report']['modalities_used']}")
                for warning in result["ethical_report"]["warnings"]:
                    st.caption(f"- {warning}")
            if result["voice_result"]["transcript"]:
                st.markdown("#### Voice transcript")
                st.write(result["voice_result"]["transcript"])
            st.markdown("#### Text intelligence")
            text_result = result["text_result"]
            st.write(f"Emotion: {text_result['emotion_label']} | Intent: {text_result['intent']} | Method: {text_result['method']}")
            if text_result["synonyms_preview"]:
                st.caption("NLTK synonym hints: " + ", ".join(text_result["synonyms_preview"]))
            st.markdown("#### Recommendation")
            render_recommendation_card(result["recommendation"])

    with tab_chat:
        st.subheader("Animated mascot chatbot")
        analysis = st.session_state["analysis_result"]
        active_mood = analysis["fused_mood"] if analysis else "calm"
        render_mascot(active_mood, username, tone_mode, "Ask anything. The answer adapts to your emotional state and digital twin context.")
        for role, message in st.session_state["chat_history"]:
            with st.chat_message(role):
                st.write(message)
        prompt = st.chat_input("Ask the mascot a question...")
        if prompt:
            with st.chat_message("user"):
                st.write(prompt)
            reply, provider = generate_chat_response(username, prompt, active_mood, tone_mode, analysis)
            with st.chat_message("assistant"):
                st.write(reply)
                st.caption(f"Provider: {provider}")
            st.session_state["chat_history"].append(("user", prompt))
            st.session_state["chat_history"].append(("assistant", reply))
            recommendation = analysis["recommendation"] if analysis else {}
            log_interaction(
                username,
                analysis["face_result"]["label"] if analysis else "neutral",
                analysis["face_result"]["mood"] if analysis else "calm",
                analysis["voice_result"]["mood"] if analysis else "calm",
                simple_text_mood(prompt),
                analysis["emoji_mood"] if analysis else "calm",
                active_mood,
                analysis["state_score"] if analysis else 0.0,
                tone_mode,
                recommendation,
                chat_query=prompt,
                chat_reply=reply,
            )
            if speak_reply:
                st.info(speak_text_locally(reply))

    with tab_train:
        st.subheader("PyTorch EfficientNetV2-S training")
        if not TORCH_AVAILABLE:
            st.error("Install torch and torchvision to enable training.")
        summary = dataset_directory_summary(DATASET_DIR)
        if summary:
            st.table(summary)
        else:
            st.info("Create dataset/train, dataset/val, and dataset/test folders before training.")
        batch_size = st.select_slider("Batch size", [4, 8, 12, 16], value=8)
        epochs = st.slider("Epochs", 1, 6, 2)
        learning_rate = st.select_slider("Learning rate", [1e-4, 2e-4, 5e-4, 1e-3], value=2e-4)
        freeze_backbone = st.checkbox("Freeze backbone for fast demo training", value=True)
        if st.button("Train .pth model", disabled=not TORCH_AVAILABLE):
            progress = st.progress(0.0)
            output = st.empty()

            def update_progress(fraction, row):
                progress.progress(float(fraction))
                output.write(row)

            try:
                with st.spinner("Training EfficientNetV2-S..."):
                    history, class_names = train_emotion_model(DATASET_DIR, batch_size, epochs, float(learning_rate), freeze_backbone, update_progress)
                st.success(f"Saved {MODEL_PATH.name}")
                st.write("Classes: " + ", ".join(class_names))
                st.table(history)
            except Exception as exc:
                st.error(f"Training failed: {exc}")
        if MODEL_META_PATH.exists():
            st.markdown("#### Saved model metadata")
            try:
                st.json(json.loads(MODEL_META_PATH.read_text(encoding="utf-8")))
            except Exception:
                st.caption("Metadata exists but could not be parsed.")

    with tab_twin:
        st.subheader("Digital Emotional Twin")
        rows = read_csv_rows(TWIN_LOG_PATH)
        if rows:
            st.dataframe(rows, width="stretch")
            st.download_button("Download twin log", TWIN_LOG_PATH.read_bytes(), file_name=TWIN_LOG_PATH.name, mime="text/csv")
        else:
            st.info("No interactions logged yet.")
        st.subheader("Recommender memory")
        st.dataframe(get_recommender_stats(), width="stretch")
        st.subheader("Resource catalog")
        st.dataframe(catalog_rows(), width="stretch")
        st.download_button("Download catalog", CATALOG_EXPORT_PATH.read_bytes(), file_name=CATALOG_EXPORT_PATH.name, mime="text/csv")

    with tab_setup:
        render_setup_tab()


if __name__ == "__main__":
    main()
