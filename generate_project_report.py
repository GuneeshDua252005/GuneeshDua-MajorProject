from __future__ import annotations

import html
import re
import textwrap
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"


APP_CODE = r'''from __future__ import annotations

import csv
import html
import io
import json
import os
import random
import re
import textwrap
import urllib.error
import urllib.request
import wave
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

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
    from torch.utils.data import DataLoader, Dataset
    from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    DataLoader = None
    Dataset = object
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
APP_SUBTITLE = "PyTorch EfficientNetV2-S, Grad-CAM, voice, face, text emotion fusion, and free Hugging Face API chat."
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
TEXT_EMOTION_MODEL = os.getenv("TEXT_EMOTION_MODEL", "j-hartmann/emotion-english-distilroberta-base").strip()
TEXT_SENTIMENT_MODEL = os.getenv("TEXT_SENTIMENT_MODEL", "distilbert-base-uncased-finetuned-sst-2-english").strip()

MOOD_CHOICES = ["happy", "sad", "calm", "energetic"]
EMOJI_MAP = {
    "happy": "happy",
    "sad": "sad",
    "calm": "calm",
    "energetic": "energetic",
}
EMOTION_TO_MOOD = {
    "happy": "happy", "joy": "happy", "love": "happy", "positive": "happy",
    "surprise": "energetic", "excitement": "energetic", "angry": "energetic",
    "anger": "energetic", "frustration": "energetic",
    "fear": "sad", "sad": "sad", "sadness": "sad", "negative": "sad",
    "neutral": "calm", "calm": "calm", "relaxed": "calm",
}
DEFAULT_FACE_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

FALLBACK_ACTIONS = {
    "happy": "Write one win, one gratitude point, and one next meaningful goal.",
    "sad": "Do two minutes of slow breathing, drink water, and message one trusted person.",
    "calm": "Protect this state with one low-distraction focus sprint.",
    "energetic": "Channel your energy into one important task or a short movement break.",
}
MOOD_QUESTION_BANK = {
    "happy": ["What created this positive shift?", "How can you repeat it?", "Who can you share this progress with?"],
    "sad": ["What feels heaviest right now?", "What helped last time?", "What tiny step can make the next hour easier?"],
    "calm": ["What is protecting your balance?", "Which routine is helping focus?", "What boundary will maintain calm?"],
    "energetic": ["Is the energy useful or overloaded?", "Which priority deserves it first?", "What boundary keeps it healthy?"],
}
TONE_GUIDES = {
    "Therapist": "Warm, reflective, structured, and non-diagnostic.",
    "Friendly": "Natural, caring, and specific.",
    "Motivational": "Positive, action-oriented, and disciplined.",
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
        raise RuntimeError("Streamlit is not installed. Run: streamlit run app.py")


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
    ensure_csv(TWIN_LOG_PATH, [
        "User", "TimeUTC", "FaceLabel", "FaceMood", "VoiceMood", "TextMood", "EmojiMood",
        "FusedMood", "StateScore", "ToneMode", "RecommendedId", "RecommendedTitle",
        "RecommendedUrl", "RecommendedSource", "ChatQuery", "ChatReply", "Feedback",
    ])
    ensure_csv(RECOMMENDER_STATS_PATH, ["ItemId", "Exposures", "Likes", "Skips", "LastShown", "LastFeedback"])


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
            writer.writerow({h: row.get(h, "") for h in headers})


def append_row(path: Path, headers: list[str], row: dict[str, Any]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        if not exists:
            writer.writeheader()
        writer.writerow({h: row.get(h, "") for h in headers})


@lru_cache(maxsize=1)
def build_resource_catalog() -> tuple[CatalogEntry, ...]:
    direct = {
        "happy": ["feel good playlist", "gratitude meditation", "confidence boost"],
        "sad": ["self compassion meditation", "comfort songs", "gentle piano healing"],
        "calm": ["lofi focus session", "rain sounds for study", "mindful breathing guide"],
        "energetic": ["workout motivation mix", "focus sprint soundtrack", "productivity music"],
    }
    entries: list[CatalogEntry] = []
    for mood, queries in direct.items():
        for index, query in enumerate(queries, start=1):
            for source, base in [
                ("YouTube", "https://www.youtube.com/results?search_query="),
                ("Spotify", "https://open.spotify.com/search/"),
                ("YouTube Music", "https://music.youtube.com/search?q="),
            ]:
                entries.append(CatalogEntry(
                    id=f"{mood}_{slugify(source)}_{index}",
                    mood=mood,
                    title=f"{mood.title()} {source} resource: {query.title()}",
                    url=base + urllib.parse.quote_plus(query),
                    source=source,
                    resource_type="public_search",
                    playable=False,
                    tags=f"{mood},{slugify(query)}",
                    offline_fallback=FALLBACK_ACTIONS[mood],
                ))
    return tuple(entries)


def catalog_rows() -> list[dict[str, Any]]:
    return [asdict(item) for item in build_resource_catalog()]


def export_catalog(path: Path = CATALOG_EXPORT_PATH) -> Path:
    rows = catalog_rows()
    if rows:
        write_rows(path, list(rows[0].keys()), rows)
    return path


def get_recommender_stats() -> list[dict[str, str]]:
    ensure_runtime_files()
    return read_rows(RECOMMENDER_STATS_PATH)


def upsert_feedback(item_id: str, feedback: str = "shown") -> None:
    rows = get_recommender_stats()
    headers = ["ItemId", "Exposures", "Likes", "Skips", "LastShown", "LastFeedback"]
    row = next((r for r in rows if r.get("ItemId") == item_id), None)
    if row is None:
        row = {"ItemId": item_id, "Exposures": "0", "Likes": "0", "Skips": "0", "LastShown": "", "LastFeedback": ""}
        rows.append(row)
    exposures = int(row.get("Exposures", "0") or 0)
    likes = int(row.get("Likes", "0") or 0)
    skips = int(row.get("Skips", "0") or 0)
    if feedback == "shown":
        exposures += 1
        row["LastShown"] = utc_now()
    elif feedback == "liked":
        likes += 1
        row["LastFeedback"] = "liked"
    elif feedback == "skipped":
        skips += 1
        row["LastFeedback"] = "skipped"
    row.update({"Exposures": str(exposures), "Likes": str(likes), "Skips": str(skips)})
    write_rows(RECOMMENDER_STATS_PATH, headers, rows)


def user_history(user: str) -> list[str]:
    return [r.get("RecommendedId", "") for r in read_rows(TWIN_LOG_PATH) if r.get("User") == user and r.get("RecommendedId")]


def recommend_item(user: str, mood: str, last_n: int = 5) -> dict[str, Any]:
    candidates = [r for r in catalog_rows() if r["mood"] == mood] or catalog_rows()
    recent = set(user_history(user)[-last_n:])
    stats = {r["ItemId"]: r for r in get_recommender_stats()}
    scored: list[tuple[float, dict[str, Any]]] = []
    for item in candidates:
        s = stats.get(item["id"], {})
        exposures = int(s.get("Exposures", "0") or 0)
        likes = int(s.get("Likes", "0") or 0)
        skips = int(s.get("Skips", "0") or 0)
        score = 0.35 / (exposures + 1) + 0.45 * likes / max(exposures, 1) - 0.20 * skips / max(exposures, 1)
        score -= 0.50 if item["id"] in recent else 0.0
        score += random.uniform(0.0, 0.05)
        scored.append((score, item))
    scored.sort(key=lambda p: p[0], reverse=True)
    choice = scored[0][1]
    upsert_feedback(choice["id"], "shown")
    return choice


def log_interaction(user: str, face_label: str, face_mood: str, voice_mood: str, text_mood: str, emoji_mood: str,
                    fused_mood: str, state_score: float, tone_mode: str, recommendation: dict[str, Any],
                    chat_query: str = "", chat_reply: str = "", feedback: str = "") -> None:
    append_row(TWIN_LOG_PATH, [
        "User", "TimeUTC", "FaceLabel", "FaceMood", "VoiceMood", "TextMood", "EmojiMood",
        "FusedMood", "StateScore", "ToneMode", "RecommendedId", "RecommendedTitle",
        "RecommendedUrl", "RecommendedSource", "ChatQuery", "ChatReply", "Feedback",
    ], {
        "User": user, "TimeUTC": utc_now(), "FaceLabel": face_label, "FaceMood": face_mood,
        "VoiceMood": voice_mood, "TextMood": text_mood, "EmojiMood": emoji_mood, "FusedMood": fused_mood,
        "StateScore": f"{state_score:.4f}", "ToneMode": tone_mode,
        "RecommendedId": recommendation.get("id", ""), "RecommendedTitle": recommendation.get("title", ""),
        "RecommendedUrl": recommendation.get("url", ""), "RecommendedSource": recommendation.get("source", ""),
        "ChatQuery": chat_query, "ChatReply": chat_reply, "Feedback": feedback,
    })


def twin_summary(user: str, limit: int = 12) -> str:
    rows = [r for r in read_rows(TWIN_LOG_PATH) if r.get("User") == user]
    if not rows:
        return "No digital emotional twin history is available yet."
    moods = [r.get("FusedMood", "calm") for r in rows[-limit:]]
    dominant = Counter(moods).most_common(1)[0][0]
    return f"Dominant mood: {dominant}. Recent sequence: {', '.join(moods[-5:])}."


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
    found: list[str] = []
    try:
        for synset in wn.synsets(word):
            for lemma in synset.lemmas():
                candidate = lemma.name().replace("_", " ").lower().strip()
                if candidate and candidate != word.lower() and candidate not in found:
                    found.append(candidate)
                if len(found) >= 8:
                    return tuple(found)
    except Exception:
        return tuple()
    return tuple(found)


def infer_intent(text: str) -> dict[str, Any]:
    tokens = set(tokenize_words(text))
    best_intent = "general_support"
    best_score = 0
    best_hits: list[str] = []
    for intent, seeds in INTENT_SEEDS.items():
        expanded = set(seeds)
        for seed in seeds:
            expanded.update(tokenize_words(" ".join(synonyms(seed))))
        hits = sorted(tokens.intersection(expanded))
        if len(hits) > best_score:
            best_intent, best_score, best_hits = intent, len(hits), hits
    preview = sorted({s for t in list(tokens)[:6] for s in synonyms(t)[:2]})[:8]
    return {"intent": best_intent, "matched_terms": best_hits, "synonyms_preview": preview}


@lru_cache(maxsize=1)
def load_classifier(model_name: str):
    if not TRANSFORMERS_AVAILABLE or not TORCH_AVAILABLE:
        return None, None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()
        return tokenizer, model
    except Exception:
        return None, None


def classify_text(model_name: str, text: str) -> tuple[str, float, dict[str, float]]:
    tokenizer, model = load_classifier(model_name)
    if tokenizer is None or model is None or not TORCH_AVAILABLE:
        return "", 0.0, {}
    with torch.no_grad():
        encoded = tokenizer(text[:512], return_tensors="pt", truncation=True)
        logits = model(**encoded).logits
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
    labels = model.config.id2label if hasattr(model, "config") else {}
    all_scores = {str(labels.get(i, f"class_{i}")): float(p) for i, p in enumerate(probs)}
    label = max(all_scores, key=all_scores.get)
    return label, all_scores[label], all_scores


def simple_text_mood(text: str) -> str:
    lower = text.lower()
    if any(t in lower for t in ("sad", "lonely", "cry", "hurt", "grief")):
        return "sad"
    if any(t in lower for t in ("happy", "joy", "love", "great", "awesome")):
        return "happy"
    if any(t in lower for t in ("angry", "mad", "stress", "frustrat", "rage")):
        return "energetic"
    return "calm"


def analyze_text(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {"text": "", "mood": "calm", "confidence": 0.5, "emotion_label": "neutral",
                "emotion_score": 0.5, "sentiment_label": "NEUTRAL", "sentiment_score": 0.5,
                "intent": "general_support", "matched_terms": [], "synonyms_preview": [],
                "mood_scores": {m: 0.0 for m in MOOD_CHOICES}, "method": "empty_text"}
    mood_scores = {m: 0.0 for m in MOOD_CHOICES}
    emotion_label, emotion_score, emotion_all = classify_text(TEXT_EMOTION_MODEL, text)
    sentiment_label, sentiment_score, _ = classify_text(TEXT_SENTIMENT_MODEL, text)
    if emotion_all:
        for label, score in emotion_all.items():
            mood_scores[normalize_label_to_mood(label)] += float(score)
        method = "transformers"
    else:
        emotion_label, emotion_score, method = simple_text_mood(text), 0.55, "heuristic"
        mood_scores[emotion_label] = 1.0
    sentiment_label = sentiment_label.upper() if sentiment_label else "NEUTRAL"
    sentiment_score = sentiment_score or 0.5
    if sentiment_label.startswith("NEG"):
        mood_scores["sad"] += 0.12
    if sentiment_label.startswith("POS"):
        mood_scores["happy"] += 0.12
    intent = infer_intent(text)
    mood = max(mood_scores.items(), key=lambda p: p[1])[0]
    return {"text": text, "mood": mood, "confidence": float(max(max(mood_scores.values()), emotion_score, sentiment_score)),
            "emotion_label": emotion_label or "neutral", "emotion_score": float(emotion_score),
            "sentiment_label": sentiment_label, "sentiment_score": float(sentiment_score),
            "intent": intent["intent"], "matched_terms": intent["matched_terms"],
            "synonyms_preview": intent["synonyms_preview"], "mood_scores": mood_scores, "method": method}


def transcribe_audio(audio_bytes: bytes | None) -> tuple[str, str]:
    if not audio_bytes:
        return "", "no_audio"
    if sr is None:
        return "", "speechrecognition_missing"
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio = recognizer.record(source)
        return recognizer.recognize_google(audio).strip(), "google_web_speech"
    except Exception:
        return "", "transcription_failed"


def wav_energy(audio_bytes: bytes | None) -> float:
    if not audio_bytes:
        return 0.0
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            sample_width = wf.getsampwidth()
            channels = wf.getnchannels()
        if sample_width == 1:
            data = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0
            max_abs = 128.0
        elif sample_width == 2:
            data = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
            max_abs = 32768.0
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
    scores = dict(text_result["mood_scores"])
    energy = wav_energy(audio_bytes)
    if energy > 0.14:
        scores["energetic"] += 0.15
    if energy < 0.04:
        scores["calm"] += 0.08
    mood = max(scores.items(), key=lambda p: p[1])[0]
    return {"transcript": transcript, "transcription_method": method, "energy": energy, "mood": mood,
            "confidence": float(max(scores.values()) if scores else 0.0),
            "spoken_name_detected": bool(username and transcript and username.lower() in transcript.lower()),
            "text_result": text_result}


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


def decode_image(image_bytes: bytes | None) -> np.ndarray | None:
    if not image_bytes or cv2 is None:
        return None
    return cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)


def largest_face_box(image_bgr: np.ndarray):
    cascade = face_cascade()
    if cascade is None or cv2 is None:
        return None
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    return None if len(faces) == 0 else max(faces, key=lambda b: int(b[2] * b[3]))


def heuristic_face_label(face_bgr: np.ndarray) -> tuple[str, float]:
    if cv2 is None:
        return "neutral", 0.40
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    texture = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    smiles = []
    sc = smile_cascade()
    if sc is not None:
        try:
            smiles = sc.detectMultiScale(gray, scaleFactor=1.7, minNeighbors=20, minSize=(25, 25))
        except Exception:
            smiles = []
    if len(smiles) > 0 or brightness > 150:
        return "happy", 0.58
    if texture > 450 and brightness < 130:
        return "angry", 0.47
    if brightness < 95:
        return "sad", 0.46
    return "neutral", 0.44


def image_to_tensor(image_rgb: np.ndarray) -> Any:
    if not TORCH_AVAILABLE or cv2 is None:
        return None
    resized = cv2.resize(image_rgb, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = np.transpose((resized - mean) / std, (2, 0, 1))
    return torch.from_numpy(arr).unsqueeze(0).float()


def build_model(num_classes: int, pretrained: bool = True):
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


def save_checkpoint(model: Any, class_names: list[str], extra: dict[str, Any] | None = None) -> None:
    if not TORCH_AVAILABLE:
        return
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"architecture": "efficientnet_v2_s", "class_names": class_names,
               "model_state_dict": model.state_dict(), "saved_at_utc": utc_now()}
    if extra:
        payload.update(extra)
    torch.save(payload, MODEL_PATH)
    MODEL_META_PATH.write_text(json.dumps({k: v for k, v in payload.items() if k != "model_state_dict"}, indent=2), encoding="utf-8")


@lru_cache(maxsize=1)
def load_checkpoint():
    if not TORCH_AVAILABLE or not MODEL_PATH.exists():
        return None, {}
    try:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
        classes = checkpoint.get("class_names") or DEFAULT_FACE_CLASSES
        model = build_model(len(classes), pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        model.eval()
        return model, {"class_names": classes}
    except Exception:
        return None, {}


def last_conv_module(model: Any):
    if model is None:
        return None
    last = None
    for module in model.modules():
        if TORCH_AVAILABLE and isinstance(module, nn.Conv2d):
            last = module
    return last


def gradcam_heatmap(model: Any, tensor: Any, target_index: int | None = None) -> np.ndarray | None:
    if not TORCH_AVAILABLE or model is None or tensor is None:
        return None
    layer = last_conv_module(model)
    if layer is None:
        return None
    activations, gradients = [], []
    f_handle = layer.register_forward_hook(lambda _m, _i, out: activations.append(out.detach()))
    b_handle = layer.register_full_backward_hook(lambda _m, _gin, gout: gradients.append(gout[0].detach()))
    try:
        model.zero_grad(set_to_none=True)
        logits = model(tensor)
        if target_index is None:
            target_index = int(torch.argmax(logits, dim=1).item())
        logits[:, target_index].sum().backward()
        acts, grads = activations[-1][0], gradients[-1][0]
        weights = grads.mean(dim=(1, 2))
        cam = torch.relu((weights[:, None, None] * acts).sum(dim=0))
        if float(cam.max()) <= 0:
            return None
        return np.uint8(255 * (cam / cam.max()).cpu().numpy())
    except Exception:
        return None
    finally:
        f_handle.remove()
        b_handle.remove()


def overlay_heatmap(face_bgr: np.ndarray, heatmap: np.ndarray | None) -> np.ndarray | None:
    if cv2 is None or heatmap is None:
        return None
    heatmap = cv2.resize(heatmap, (face_bgr.shape[1], face_bgr.shape[0]))
    colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(colored, 0.35, face_bgr, 0.65, 0.0)


def predict_face(image_bytes: bytes | None) -> dict[str, Any]:
    image = decode_image(image_bytes)
    if image is None or cv2 is None:
        return {"label": "neutral", "mood": "calm", "confidence": 0.0, "method": "no_image",
                "overlay_rgb": None, "face_found": False}
    box = largest_face_box(image)
    display = image.copy()
    if box is None:
        x, y, w, h = 0, 0, image.shape[1], image.shape[0]
        face = image
    else:
        x, y, w, h = map(int, box)
        face = image[y:y + h, x:x + w]
        cv2.rectangle(display, (x, y), (x + w, y + h), (70, 255, 140), 2)
    model, meta = load_checkpoint()
    if model is not None:
        try:
            tensor = image_to_tensor(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            probs = torch.softmax(model(tensor), dim=1)[0].detach().cpu().numpy()
            classes = meta.get("class_names") or DEFAULT_FACE_CLASSES
            idx = int(np.argmax(probs))
            label = classes[idx] if idx < len(classes) else f"class_{idx}"
            overlay = overlay_heatmap(face, gradcam_heatmap(model, tensor, idx))
            if overlay is not None and box is not None:
                display[y:y + h, x:x + w] = cv2.resize(overlay, (w, h))
            return {"label": label, "mood": normalize_label_to_mood(label), "confidence": float(probs[idx]),
                    "method": "trained_efficientnet_v2_s", "overlay_rgb": cv2.cvtColor(display, cv2.COLOR_BGR2RGB),
                    "face_found": box is not None}
        except Exception:
            pass
    label, confidence = heuristic_face_label(face)
    return {"label": label, "mood": normalize_label_to_mood(label), "confidence": confidence,
            "method": "opencv_heuristic", "overlay_rgb": cv2.cvtColor(display, cv2.COLOR_BGR2RGB),
            "face_found": box is not None}


class FolderDataset(Dataset):
    def __init__(self, root: Path, class_to_idx: dict[str, int] | None = None, train: bool = False):
        self.root = root
        self.train = train
        classes = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.exists() else []
        self.class_to_idx = class_to_idx or {name: i for i, name in enumerate(classes)}
        self.samples: list[tuple[Path, int]] = []
        for name, idx in self.class_to_idx.items():
            for fp in (root / name).rglob("*") if (root / name).exists() else []:
                if fp.is_file() and fp.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((fp, idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = cv2.imread(str(path)) if cv2 is not None else None
        if image is None:
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        rgb = cv2.cvtColor(cv2.resize(image, (224, 224)), cv2.COLOR_BGR2RGB)
        if self.train and random.random() < 0.5:
            rgb = cv2.flip(rgb, 1)
        tensor = image_to_tensor(rgb)[0]
        return tensor, torch.tensor(label, dtype=torch.long)


def dataset_has_images(split_dir: Path) -> bool:
    return split_dir.exists() and any(p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS for p in split_dir.rglob("*"))


def dataset_summary(root: Path = DATASET_DIR) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        split_dir = root / split
        if split_dir.exists():
            for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                rows.append({"split": split, "class_name": class_dir.name,
                             "count": sum(1 for f in class_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)})
    return rows


def build_loaders(dataset_root: Path, batch_size: int):
    if not TORCH_AVAILABLE or cv2 is None:
        raise RuntimeError("PyTorch and OpenCV are required for training.")
    if not dataset_has_images(dataset_root / "train"):
        raise RuntimeError("No images found in dataset/train.")
    train_ds = FolderDataset(dataset_root / "train", train=True)
    val_ds = FolderDataset(dataset_root / "val", class_to_idx=train_ds.class_to_idx) if dataset_has_images(dataset_root / "val") else None
    test_ds = FolderDataset(dataset_root / "test", class_to_idx=train_ds.class_to_idx) if dataset_has_images(dataset_root / "test") else None
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0) if val_ds else None,
            DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0) if test_ds else None,
            list(train_ds.class_to_idx.keys()))


def train_model(dataset_root: Path, batch_size: int, epochs: int, lr: float, freeze_backbone: bool, progress_callback=None):
    train_loader, val_loader, test_loader, class_names = build_loaders(dataset_root, batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(class_names), pretrained=True).to(device)
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    criterion = nn.CrossEntropyLoss()
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = correct = total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * labels.size(0)
            correct += int((torch.argmax(logits, dim=1) == labels).sum().item())
            total += int(labels.size(0))
        row = {"epoch": float(epoch), "train_loss": round(loss_sum / max(total, 1), 4),
               "train_accuracy": round(correct / max(total, 1), 4)}
        history.append(row)
        if progress_callback:
            progress_callback(epoch / epochs, row)
    save_checkpoint(model.cpu(), class_names, {"epochs": epochs, "batch_size": batch_size, "learning_rate": lr})
    load_checkpoint.cache_clear()
    return history, {"message": "Training complete; evaluate using validation or test dataset."}, class_names


def fuse_emotions(face_result: dict[str, Any], voice_result: dict[str, Any], text_result: dict[str, Any], emoji_mood: str):
    scores = {m: 0.0 for m in MOOD_CHOICES}
    scores[face_result.get("mood", "calm")] += 0.35 * max(float(face_result.get("confidence") or 0.45), 0.30)
    scores[voice_result.get("mood", "calm")] += 0.25 * max(float(voice_result.get("confidence") or 0.45), 0.30)
    scores[text_result.get("mood", "calm")] += 0.25 * max(float(text_result.get("confidence") or 0.45), 0.30)
    scores[emoji_mood] += 0.15
    fused = max(scores.items(), key=lambda p: p[1])[0]
    return fused, scores, float(scores[fused] / (sum(scores.values()) or 1.0))


def ethical_report(face_result: dict[str, Any], voice_result: dict[str, Any], text_result: dict[str, Any], scores: dict[str, float]):
    warnings = ["This assistant provides supportive guidance, not medical diagnosis."]
    modalities = int(face_result.get("method") != "no_image") + int(bool(voice_result.get("transcript"))) + int(bool(text_result.get("text")))
    if face_result.get("method") == "opencv_heuristic":
        warnings.append("Face analysis is in heuristic fallback mode until a .pth model is trained.")
    if modalities < 2:
        warnings.append("Fusion is stronger when at least two modalities are provided.")
    maximum = max(scores.values()) if scores else 0.0
    band = "High" if maximum >= 0.45 else "Medium" if maximum >= 0.28 else "Low"
    return {"modalities_used": modalities, "confidence_band": band, "warnings": warnings}


def build_system_prompt(username: str, mood: str, tone_mode: str, summary: str, intent: str, snapshot: str) -> str:
    return textwrap.dedent(f"""
    You are an emotionally intelligent therapeutic chatbot and animated mascot.
    User name: {username}
    Current fused mood: {mood}
    Tone mode: {tone_mode}
    Tone guide: {TONE_GUIDES.get(tone_mode, TONE_GUIDES['Therapist'])}
    Detected intent: {intent}
    Digital twin summary: {summary}
    Emotion snapshot: {snapshot}
    Give one supportive reflection, exactly three reflective questions, and one practical action.
    Do not diagnose disease or claim certainty.
    """).strip()


def huggingface_chat(system_prompt: str, user_prompt: str) -> str | None:
    if not HF_TOKEN:
        return None
    payload = {"inputs": f"System:\n{system_prompt}\n\nUser:\n{user_prompt}\n\nAssistant:\n",
               "parameters": {"max_new_tokens": 220, "temperature": 0.7, "top_p": 0.9, "return_full_text": False},
               "options": {"wait_for_model": True}}
    request = urllib.request.Request(
        f"https://api-inference.huggingface.co/models/{HF_CHAT_MODEL}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            parsed = json.loads(response.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, Exception):
        return None
    if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
        return str(parsed[0].get("generated_text", "")).strip() or None
    if isinstance(parsed, dict):
        return str(parsed.get("generated_text", "")).strip() or None
    return None


def local_coach(question: str, mood: str, tone_mode: str, summary: str, intent: str) -> str:
    questions = MOOD_QUESTION_BANK.get(mood, MOOD_QUESTION_BANK["calm"])
    opener = {"Therapist": "I hear you, and I want to respond thoughtfully.",
              "Friendly": "I am with you, and we can handle this together.",
              "Motivational": "You are capable, and we can turn this into action right now."}.get(tone_mode, "I am here.")
    return textwrap.dedent(f"""
    {opener}

    Based on your current {mood} state and intent around {intent}, here is a focused response.
    Twin summary: {summary}

    Reflection:
    - Your message points to a meaningful emotional need, and it deserves attention.

    Important questions:
    1. {questions[0]}
    2. {questions[1]}
    3. {questions[2]}

    One next action:
    - {FALLBACK_ACTIONS.get(mood, FALLBACK_ACTIONS['calm'])}
    """).strip()


def generate_reply(username: str, question: str, current_mood: str, tone_mode: str, analysis: dict[str, Any] | None):
    text_result = analyze_text(question)
    mood = current_mood if current_mood in MOOD_CHOICES else text_result["mood"]
    summary = twin_summary(username)
    snapshot = "No multimodal analysis available yet."
    if analysis:
        snapshot = f"Face={analysis['face_result']['mood']}, Voice={analysis['voice_result']['mood']}, Text={analysis['text_result']['mood']}, Fused={analysis['fused_mood']}"
    system = build_system_prompt(username, mood, tone_mode, summary, text_result["intent"], snapshot)
    user_prompt = f"Question: {question}\nEmotion: {text_result['emotion_label']}\nSentiment: {text_result['sentiment_label']}\nSynonyms: {', '.join(text_result['synonyms_preview']) or 'none'}"
    return huggingface_chat(system, user_prompt) or local_coach(question, mood, tone_mode, summary, text_result["intent"])


def speak_text(text: str) -> str:
    if pyttsx3 is None:
        return "pyttsx3 is not installed."
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 170)
        engine.say(text)
        engine.runAndWait()
        return "Reply spoken locally."
    except Exception:
        return "Text-to-speech failed on this machine."


def render_mascot(mood: str, username: str, tone_mode: str, subtitle: str) -> None:
    require_streamlit()
    colors = {
        "happy": ("#fff9db", "#ffd866", "#ff922b", "smile"),
        "sad": ("#eef2ff", "#a5b4fc", "#4c6ef5", "sad"),
        "calm": ("#ecfeff", "#99f6e4", "#0f766e", "calm"),
        "energetic": ("#fff1f2", "#fda4af", "#e11d48", "grin"),
    }[mood if mood in MOOD_CHOICES else "calm"]
    bg, face, accent, mouth = colors
    mouth_css = {"smile": "border-bottom:6px solid #7f1d1d;border-radius:0 0 70px 70px;",
                 "sad": "border-top:6px solid #1e293b;border-radius:70px 70px 0 0;",
                 "calm": "border-top:5px solid #0f172a;height:0;",
                 "grin": "border-bottom:6px solid #7f1d1d;border-radius:0 0 80px 80px;"}[mouth]
    st.markdown(f"""
    <style>
    .cei-card{{background:{bg};border-radius:22px;padding:18px;box-shadow:0 12px 24px rgba(15,23,42,.08)}}
    .cei-bubble{{background:white;border-radius:18px;padding:12px;color:#0f172a}}
    .cei-stage{{display:flex;align-items:center;justify-content:center;min-height:230px;position:relative}}
    .cei-glow{{position:absolute;width:220px;height:220px;background:radial-gradient(circle,{accent}33 0%,transparent 70%);animation:pulse 2.6s infinite}}
    .cei-avatar{{position:relative;width:170px;height:170px;border-radius:999px;background:{face};animation:float 2.4s infinite;box-shadow:0 18px 28px rgba(15,23,42,.15)}}
    .cei-eye{{position:absolute;top:66px;width:18px;height:18px;border-radius:999px;background:#111827;animation:blink 4.2s infinite}}
    .left{{left:48px}}.right{{right:48px}}
    .cei-mouth{{position:absolute;left:50%;transform:translateX(-50%);bottom:42px;width:56px;height:28px;{mouth_css}}}
    @keyframes float{{0%,100%{{transform:translateY(0)}}50%{{transform:translateY(-8px)}}}}
    @keyframes blink{{0%,92%,100%{{transform:scaleY(1)}}94%,96%{{transform:scaleY(.1)}}}}
    @keyframes pulse{{0%,100%{{transform:scale(.92);opacity:.7}}50%{{transform:scale(1.05);opacity:1}}}}
    </style>
    <div class="cei-card"><div class="cei-bubble">{html.escape(subtitle)}</div><div class="cei-stage">
    <div class="cei-glow"></div><div class="cei-avatar"><div class="cei-eye left"></div><div class="cei-eye right"></div><div class="cei-mouth"></div></div></div>
    <b>Mascot mode:</b> {html.escape(tone_mode)} | <b>User:</b> {html.escape(username)} | <b>Mood:</b> {html.escape(mood.title())}</div>
    """, unsafe_allow_html=True)


def run_live_analysis(user: str, emoji: str, image_bytes: bytes | None, audio_bytes: bytes | None, text_input: str, tone_mode: str):
    face = predict_face(image_bytes)
    voice = analyze_voice(audio_bytes, user)
    text_result = analyze_text(text_input)
    emoji_mood = EMOJI_MAP.get(emoji, "calm")
    fused, scores, state_score = fuse_emotions(face, voice, text_result, emoji_mood)
    recommendation = recommend_item(user, fused)
    result = {"face_result": face, "voice_result": voice, "text_result": text_result, "emoji_mood": emoji_mood,
              "fused_mood": fused, "fusion_scores": scores, "state_score": state_score,
              "recommendation": recommendation, "ethical_report": ethical_report(face, voice, text_result, scores),
              "tone_mode": tone_mode}
    log_interaction(user, face["label"], face["mood"], voice["mood"], text_result["mood"], emoji_mood,
                    fused, state_score, tone_mode, recommendation)
    return result


def main() -> None:
    require_streamlit()
    ensure_runtime_files()
    export_catalog()
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)
    st.sidebar.header("Inputs")
    username = st.sidebar.text_input("Username", value="guest_user")
    tone_mode = st.sidebar.selectbox("Adaptive tone mode", options=list(TONE_GUIDES.keys()))
    emoji = st.sidebar.select_slider("Emoji mood input", options=list(EMOJI_MAP.keys()), value="happy")
    speak = st.sidebar.checkbox("Speak chatbot reply locally", value=False)
    st.sidebar.caption("Use HF_TOKEN environment variable for free Hugging Face API access.")
    st.session_state.setdefault("analysis", None)
    st.session_state.setdefault("chat_history", [])
    live_tab, chat_tab, train_tab, memory_tab, setup_tab = st.tabs(["Live Emotion Studio", "Mascot Chat", "Training", "Twin Memory", "Setup"])

    with live_tab:
        left, right = st.columns([1.1, 1.0])
        with left:
            cam = st.camera_input("Capture face image")
            image_bytes = cam.getvalue() if cam else None
            audio_bytes = None
            if hasattr(st, "audio_input"):
                voice_file = st.audio_input("Record voice")
                audio_bytes = voice_file.getvalue() if voice_file else None
            text_input = st.text_area("Context text", height=140)
            if st.button("Analyze emotional state", type="primary"):
                st.session_state["analysis"] = run_live_analysis(username, emoji, image_bytes, audio_bytes, text_input, tone_mode)
        with right:
            analysis = st.session_state["analysis"]
            mood = analysis["fused_mood"] if analysis else "calm"
            render_mascot(mood, username, tone_mode, f"{username}, I am ready to analyze face, voice, text, and emoji signals.")
        if st.session_state["analysis"]:
            a = st.session_state["analysis"]
            st.metric("User Emotional State Score", f"{a['state_score']:.2%}")
            st.write("Face mood:", a["face_result"]["mood"], "Voice mood:", a["voice_result"]["mood"], "Text mood:", a["text_result"]["mood"])
            if a["face_result"].get("overlay_rgb") is not None:
                st.image(a["face_result"]["overlay_rgb"], caption=a["face_result"]["method"], use_container_width=True)
            st.json(a["ethical_report"])
            st.subheader("Adaptive recommendation")
            st.markdown(f"[{a['recommendation']['title']}]({a['recommendation']['url']})")

    with chat_tab:
        analysis = st.session_state["analysis"]
        mood = analysis["fused_mood"] if analysis else "calm"
        render_mascot(mood, username, tone_mode, "Ask a question. The reply will use your emotional state and digital twin context.")
        for role, message in st.session_state["chat_history"]:
            with st.chat_message(role):
                st.write(message)
        prompt = st.chat_input("Ask the mascot your question")
        if prompt:
            reply = generate_reply(username, prompt, mood, tone_mode, analysis)
            st.session_state["chat_history"].append(("user", prompt))
            st.session_state["chat_history"].append(("assistant", reply))
            with st.chat_message("assistant"):
                st.write(reply)
            if speak:
                st.info(speak_text(reply))

    with train_tab:
        st.subheader("PyTorch EfficientNetV2-S training pipeline")
        st.code("dataset/train/happy, dataset/train/sad, dataset/train/angry, dataset/train/neutral")
        st.table(dataset_summary(DATASET_DIR))
        batch_size = st.select_slider("Batch size", [4, 8, 12, 16], value=8)
        epochs = st.slider("Epochs", 1, 6, 2)
        lr = st.select_slider("Learning rate", [1e-4, 2e-4, 5e-4, 1e-3], value=2e-4)
        freeze = st.checkbox("Freeze EfficientNetV2 backbone", value=True)
        if st.button("Train EfficientNetV2-S"):
            progress = st.progress(0.0)
            try:
                history, evaluation, classes = train_model(DATASET_DIR, batch_size, epochs, float(lr), freeze, lambda f, r: progress.progress(f))
                st.success("Training complete")
                st.write(classes)
                st.table(history)
                st.json(evaluation)
            except Exception as exc:
                st.error(str(exc))

    with memory_tab:
        st.subheader("Digital Emotional Twin")
        st.dataframe(read_rows(TWIN_LOG_PATH), use_container_width=True)
        st.subheader("Recommender stats")
        st.dataframe(get_recommender_stats(), use_container_width=True)
        st.download_button("Download resource catalog CSV", CATALOG_EXPORT_PATH.read_bytes(), "resource_catalog.csv", "text/csv")

    with setup_tab:
        st.subheader("VS Code Windows 11 setup")
        st.code("python -m venv .venv\n.\\.venv\\Scripts\\Activate.ps1\npython -m pip install --upgrade pip\npip install -r requirements.txt\n$env:HF_TOKEN='your_hugging_face_token'\nstreamlit run app.py", language="powershell")
        st.code("streamlit\ntorch\ntorchvision\ntransformers\nopencv-python\nnumpy\nnltk\nSpeechRecognition\npyttsx3", language="text")
        st.info("No OpenAI paid API or TensorFlow/Keras is used. Hugging Face token must be kept in an environment variable.")


if __name__ == "__main__":
    main()
'''


REQUIREMENTS = """streamlit
torch
torchvision
transformers
opencv-python
numpy
nltk
SpeechRecognition
pyttsx3
"""


def para(text: str) -> str:
    return textwrap.fill(text.strip(), width=105)


def build_report_pages() -> list[tuple[str, list[str]]]:
    pages: list[tuple[str, list[str]]] = []

    def add(title: str, *body: str) -> None:
        pages.append((title, [b.strip() for b in body if b.strip()]))

    add("Cover Page",
        "Title of the Report: Emotionally Intelligent Animated Mascot Chatbot with Voice, Facial and Text Sentiment Detection",
        "AIML-452 Major Project - Dissertation",
        "Submitted in partial fulfillment of the requirement for the award of the degree of Bachelor of Technology in AIML.",
        "Submitted by: NAME OF THE STUDENT, ENROLLMENT NO.",
        "Under the supervision of: NAME OF THE FACULTY SUPERVISOR, DESIGNATION.",
        "Name of the Department, Name of the Institute, Address of the Institute.",
        "May/June 2026")
    add("Declaration",
        "This is to certify that the material embodied in this Major Project - Dissertation titled Emotionally Intelligent Animated Mascot Chatbot with Voice, Facial and Text Sentiment Detection being submitted in the partial fulfillment of the requirements for the award of the degree of Bachelor of Technology in AIML is based on my original work.",
        "It is further certified that this Major Project - Dissertation work has not been submitted in full or in part to this university or any other university for the award of any other degree or diploma. My indebtedness to other works has been duly acknowledged at the relevant places.",
        "Name of the Student, Enrollment No.")
    add("Certificate",
        "This is to certify that the work embodied in this Major Project - Dissertation titled Emotionally Intelligent Animated Mascot Chatbot with Voice, Facial and Text Sentiment Detection being submitted in the partial fulfillment of the requirements for the award of the degree of Bachelor of Technology in AIML is original and has been carried out by NAME OF THE STUDENT under my supervision and guidance.",
        "It is further certified that this Major Project - Dissertation work has not been submitted in full or in part to this university or any other university for the award of any other degree or diploma to the best of my knowledge and belief.",
        "Name of the Faculty Supervisor, Designation. Name of the HOD, HOD, Name of the Institute.")
    add("Acknowledgement",
        "I express my sincere gratitude to my faculty supervisor for consistent guidance, technical direction and academic support during the development of this major project.",
        "I am thankful to the Head of Department, Principal and the Institute for providing the academic environment and infrastructure needed for project completion.",
        "I also thank mentors, peers, family members and friends for their constructive feedback, encouragement and patience during design, implementation, testing and report preparation.")
    add("Abstract",
        "This major project presents an Emotionally Intelligent Animated Mascot Chatbot designed for supportive human-computer interaction through text, voice, facial expression and emoji-based emotional cues. The system uses a PyTorch EfficientNetV2-S architecture for facial emotion classification and Grad-CAM visual explanation, Hugging Face Transformers for free text emotion and sentiment classification, SpeechRecognition for speech-to-text, NLTK WordNet for synonym-assisted intent understanding, Streamlit for user interface design and a digital emotional twin log for persistent personalization.",
        "The project avoids TensorFlow, Keras and paid OpenAI APIs. It uses Hugging Face access tokens through environment variables and public music or video search links for adaptive recommendations. The fusion engine combines face, voice, text and emoji evidence into a unified User Emotional State Score. The chatbot then changes tone through therapist, friendly or motivational modes and asks context-specific reflective questions instead of producing generic FAQ responses.",
        "The report explains the problem statement, software requirement specification, feasibility, architecture, implementation, training pipeline, testing and future scope according to the GGSIPU Major Project - Dissertation format.")
    add("Table of Contents",
        "Declaration i | Certificate ii | Acknowledgement iii | Abstract iv | List of Figures v | List of Tables vi",
        "Chapter 1 Introduction 1 | Chapter 2 Problem Statement 10 | Chapter 3 Analysis 18 | Chapter 4 Design and Architecture 33 | Chapter 5 Implementation 46 | Chapter 6 Testing 68 | Chapter 7 Summary and Conclusion 77 | Chapter 8 Limitations and Future Work 81 | Bibliography 86 | Appendix 90")
    add("List of Figures",
        "Figure 3.1 System Use Case Diagram 29 | Figure 3.2 Context Level DFD 31 | Figure 4.1 Work Breakdown Structure 34 | Figure 4.2 Module Architecture 37 | Figure 4.3 Activity Flow 41 | Figure 4.4 Class Diagram 44 | Figure 5.1 Streamlit Dashboard Screen 54 | Figure 5.2 Animated Mascot Expressions 57 | Figure 5.3 Grad-CAM Face Analysis 61 | Figure 5.4 Digital Twin Log View 64")
    add("List of Tables",
        "Table 3.1 Functional Requirements 21 | Table 3.2 Non-functional Requirements 24 | Table 3.3 Feasibility Study 26 | Table 3.4 Tools and Technologies 28 | Table 5.1 Implementation Modules 48 | Table 5.2 Requirements.txt Dependencies 51 | Table 6.1 Test Cases 70 | Table 6.2 Risk and Mitigation Matrix 75")

    chapter1 = [
        ("CHAPTER 1: INTRODUCTION", "The proposed major project is a development-based artificial intelligence system that studies how a chatbot can become more emotionally aware when it listens to multiple signals rather than only typed text. Conventional chatbot demonstrations usually return repeated answers because they do not observe the user's emotional state, voice energy, facial expression, previous interaction pattern or preferred tone. This project addresses that limitation through a single Python 3 application suitable for execution in Visual Studio Code on a Windows 11 laptop."),
        ("1.1 Background", "Emotion-aware computing has become relevant because students, professionals and users often interact with software during stress, study pressure, loneliness, lack of motivation or emotional overload. A therapeutic-style assistant cannot replace a clinical professional, but it can provide safe reflection, emotional awareness, adaptive questions and practical next steps. The system therefore uses ethical AI monitoring and does not claim medical diagnosis."),
        ("1.2 Project Theme", "The project theme is Cognitive Emotional Intelligence and Adaptive Lifestyle Operational System. The implemented software acts as an animated mascot chatbot that reads face, voice, text and emoji cues, computes a fused mood state, recommends supportive content and stores a digital emotional twin for personalization."),
        ("1.3 Need of the Project", "A major project should demonstrate analysis, design, implementation, testing and social usefulness. This project demonstrates all of these through PyTorch facial learning, transformer-based language understanding, voice transcription, recommendation logic, Streamlit UI, Grad-CAM explainability and structured logging."),
        ("1.4 Scope", "The scope includes a single-file Python application, a training pipeline for a local facial-expression dataset, free Hugging Face API integration, animated CSS mascot, mic-based voice input where Streamlit supports it, webcam capture, CSV-based digital twin storage, adaptive recommendations and a viva-friendly setup guide."),
        ("1.5 Contribution", "The main contribution is a complete multimodal emotional intelligence prototype that is demo-ready without TensorFlow or Keras. It uses EfficientNetV2-S rather than MobileNetV2 because EfficientNetV2 is newer, more efficient and suitable for transfer learning with modern PyTorch tooling."),
        ("1.6 Ethical Position", "The application clearly states that it offers supportive guidance only. It avoids diagnostic claims, shows confidence bands and warns the user when fewer modalities are available. These design choices are important for responsible AI [1]."),
        ("1.7 Report Organization", "Chapter 1 introduces the project. Chapter 2 defines the problem and objectives. Chapter 3 presents SRS, feasibility, tools and diagrams. Chapter 4 explains architecture. Chapter 5 details implementation and setup. Chapter 6 covers testing. Chapter 7 concludes the work. Chapter 8 discusses limitations and future scope."),
        ("1.8 Execution Environment", "The target machine is a Windows 11 laptop with Intel Core i5 class processor. The application is kept lightweight by using a Streamlit web UI, CPU-compatible PyTorch inference and optional model training with small batch sizes."),
        ("1.9 Major Project Category", "This is a Major Project - Dissertation report for AIML-452. It is not an internship report and therefore focuses on problem formulation, system design, implementation, evaluation and academic references.")
    ]
    for title, body in chapter1:
        add(title, body, "IEEE citation placement: multimodal emotion recognition, transformer language models and responsible AI are cited in the relevant technical chapters [1]-[6].")

    chapter2 = [
        ("CHAPTER 2: PROBLEM STATEMENT", "Most chatbot demonstrations fail to identify the user's mindset because they are limited to typed input and static FAQ patterns. Such systems do not use speech, facial expression, context history or adaptive personality. The project problem is to design and implement a chatbot that can infer emotional state from multiple modalities and respond in a more meaningful, non-generic manner."),
        ("2.1 Problem Definition", "To develop a free, single-file Python 3 application that uses PyTorch EfficientNetV2-S, Hugging Face Transformers, NLTK, OpenCV, SpeechRecognition and Streamlit to detect user emotion from text, voice and facial cues, fuse the detected signals into a User Emotional State Score and display an animated mascot chatbot that adapts its tone and recommendations."),
        ("2.2 Objectives", "The objectives are: implement text emotion detection using free transformer models, implement voice transcription and energy-based mood estimation, implement facial emotion detection using OpenCV and PyTorch EfficientNetV2-S, implement Grad-CAM explainability, implement multimodal fusion, implement animated mascot UI, implement adaptive chatbot response generation and implement a digital twin log."),
        ("2.3 Real World Relevance", "Students often face academic stress, examination pressure and emotional fatigue. A supportive AI companion can ask reflective questions, encourage healthy micro-actions and recommend calming or motivating resources. The project is designed as a technical demonstration rather than a medical product."),
        ("2.4 Existing System Limitations", "Existing FAQ bots are usually rule-based. Many GPT-style chatbots require paid API keys. Several old facial emotion demos use MobileNetV2 or TensorFlow/Keras. The proposed system avoids paid APIs, avoids TensorFlow/Keras and uses PyTorch EfficientNetV2-S for modern CNN-based facial analysis."),
        ("2.5 Proposed System", "The proposed system integrates four emotional signals: text mood, voice mood, face mood and emoji mood. It computes weighted fusion, displays confidence, logs emotional history and produces context-aware replies in therapist, friendly and motivational modes."),
        ("2.6 Assumptions", "The system assumes webcam and microphone permission when live analysis is required. Hugging Face token is optional and should be supplied through an environment variable. Without a trained .pth file the facial module uses OpenCV heuristic fallback."),
        ("2.7 Expected Outcomes", "Expected outcomes include a working Streamlit interface, animated mascot expression change, text and voice emotion interpretation, EfficientNetV2-S training option, Grad-CAM visualization, recommendations and CSV downloads for logs.")
    ]
    for title, body in chapter2:
        add(title, body)

    chapter3 = [
        ("CHAPTER 3: ANALYSIS", "The analysis phase converts the project idea into functional, non-functional and technical requirements. Since the application uses multiple AI modules, analysis also covers feasibility, user roles, data flow, external API usage and security of tokens."),
        ("3.1 Software Requirement Specification", "The application shall run as a Streamlit app. It shall accept username, webcam image, microphone recording where supported, typed text and emoji mood. It shall process each input through its corresponding AI or heuristic module and produce a fused state."),
        ("3.1.1 Functional Requirements", "FR1: detect text emotion using Hugging Face Transformers. FR2: enrich intent using NLTK synonyms. FR3: record or accept voice and transcribe it. FR4: detect facial emotion using OpenCV and PyTorch. FR5: compute fused mood. FR6: generate adaptive chatbot reply. FR7: show animated mascot. FR8: log digital twin data. FR9: provide training pipeline. FR10: export resource catalog."),
        ("3.1.2 Non-functional Requirements", "The system should be portable, free to execute, understandable for viva, secure with respect to API tokens, responsive on CPU for demo scale, maintainable as a single file, and transparent about confidence and ethical limitations."),
        ("3.2 Feasibility Study", "Technical feasibility is high because Python, PyTorch, Streamlit and Hugging Face are widely supported. Operational feasibility is high because the UI is browser-based. Economic feasibility is high because the system avoids paid OpenAI and Spotify API usage. Schedule feasibility is achieved by keeping deployment as a local Streamlit app."),
        ("3.3 Tools and Technologies", "Python 3 is the programming language. PyTorch and torchvision provide EfficientNetV2-S. Transformers provide text models. OpenCV handles face detection and image operations. SpeechRecognition handles speech-to-text. NLTK supports synonyms. Streamlit provides UI. CSV files store the digital twin and recommender memory."),
        ("3.4 Why Hugging Face Instead of Paid GPT API", "Hugging Face is suitable because it provides free accounts, free read tokens and many open models. The project can use google/flan-t5-small through the free inference route or local fallback. OpenAI and Spotify developer API flows can introduce cost and key-management issues, while public search links remain free."),
        ("3.5 Security Analysis", "Tokens must never be hardcoded in source code. The implementation uses HF_TOKEN or HUGGINGFACEHUB_API_TOKEN environment variables. The report package intentionally removes any pasted secrets and documents safe token setup."),
        ("3.6 Use Case Diagram", "Actor: User. Use cases: Login by username, capture face, record voice, type context, choose emoji, analyze mood, ask chatbot, receive recommendation, train model, download logs. Actor: Developer. Use cases: prepare dataset, run app, set token, evaluate output."),
        ("3.7 Data Flow Diagram", "Input layer collects webcam, audio, text and emoji. Processing layer runs facial, voice and text analysis. Fusion layer computes score. Response layer generates chatbot reply and recommendation. Storage layer logs CSV records."),
        ("3.8 Data Requirements", "Training data is organized under dataset/train, dataset/val and dataset/test with class folders such as happy, sad, angry and neutral. Images may come from public facial expression datasets when licensing permits [7], [8]."),
        ("3.9 API Requirements", "The Hugging Face token is optional and free. If missing or API access fails, the system uses a local rule-based emotional coach so the demo remains functional. This is important for viva reliability.")
    ]
    for title, body in chapter3:
        add(title, body)

    chapter4 = [
        ("CHAPTER 4: DESIGN AND ARCHITECTURE", "The system is designed as a modular single-file application. Although the source code is in one file to satisfy execution simplicity, the internal structure separates runtime setup, catalog generation, text analysis, voice analysis, facial analysis, fusion, chatbot generation, training, logging and UI rendering."),
        ("4.1 Structure Chart / Work Breakdown Structure", "Level 1: Emotionally Intelligent Animated Mascot Chatbot. Level 2: User Interface, Multimodal Input, AI Analysis, Fusion Engine, Chatbot Engine, Recommender, Digital Twin, Training Pipeline, Ethics Monitor. Level 3: individual functions for each processing operation."),
        ("4.2 Module: User Interface", "The Streamlit interface provides tabs for Live Emotion Studio, Mascot Chat, Training, Twin Memory and Setup. Sidebar controls collect username, tone mode and emoji mood. The UI is intentionally clean for demonstration and viva explanation."),
        ("4.3 Module: Text Intelligence", "Text intelligence includes transformer classification, heuristic fallback, NLTK synonym expansion and intent inference. The output includes mood, confidence, sentiment label, emotion label, intent and synonym hints."),
        ("4.4 Module: Voice Intelligence", "Voice intelligence converts microphone audio to text using SpeechRecognition. It also estimates average wave energy to identify calm or energetic delivery. The transcript is then passed to the same text intelligence module."),
        ("4.5 Module: Facial Intelligence", "Facial intelligence detects the largest face using OpenCV Haar cascade. If a trained EfficientNetV2-S checkpoint exists, PyTorch inference predicts the class and Grad-CAM highlights important regions. Otherwise a heuristic fallback identifies broad mood indicators."),
        ("4.6 Module: EfficientNetV2-S", "EfficientNetV2-S is selected because it provides a modern convolutional architecture with efficient feature extraction and transfer learning support. The classifier head is replaced with dropout and linear layers for project-specific emotion classes [3]."),
        ("4.7 Module: Grad-CAM", "Grad-CAM registers forward and backward hooks on the last convolutional layer, weights activation maps by gradients and overlays a heatmap on the face crop. This improves explainability for viva and evaluation [4]."),
        ("4.8 Module: Fusion Engine", "The fusion engine assigns weights to face, voice, text and emoji signals. The default weights are 0.35, 0.25, 0.25 and 0.15 respectively. The highest weighted mood becomes the fused emotional state and normalized score becomes User Emotional State Score."),
        ("4.9 Module: Chatbot Engine", "The chatbot constructs a system prompt with username, fused mood, tone mode, detected intent, digital twin summary and analysis snapshot. Hugging Face API is used if HF_TOKEN is present; otherwise a local coach generates the answer."),
        ("4.10 Module: Recommender", "The recommender contains mood-tagged YouTube, Spotify and YouTube Music public search resources. A lightweight reinforcement-learning-style score increases novelty and learns from likes or skips stored in CSV."),
        ("4.11 Class Diagram", "CatalogEntry stores resource metadata. FolderDataset wraps image-folder training data. Other modules are implemented as functions to keep the single-file requirement simple."),
        ("4.12 Activity Diagram", "Start application, enter username, capture multimodal input, analyze individual signals, fuse moods, update mascot, show recommendation, ask chatbot, log interaction, optionally train EfficientNetV2-S and download records.")
    ]
    for title, body in chapter4:
        add(title, body)

    chapter5 = [
        ("CHAPTER 5: IMPLEMENTATION", "The implementation is provided as a complete single-file Python program named app.py. It uses only Python 3 libraries specified in requirements.txt and does not import TensorFlow or Keras. The source code is kept compatible with VS Code and Streamlit."),
        ("5.1 File Structure", "The project folder contains app.py, requirements.txt, README.md, docs folder, dataset folder created at runtime, models folder created at runtime, resource_catalog.csv, cei_twin_log.csv and recommender_stats.csv."),
        ("5.2 Requirements.txt", "The requirements file contains streamlit, torch, torchvision, transformers, opencv-python, numpy, nltk, SpeechRecognition and pyttsx3. These are the only required libraries for the implemented solution."),
        ("5.3 VS Code Execution Steps", "Create a folder, open it in VS Code, create app.py and requirements.txt, open PowerShell terminal, run python -m venv .venv, activate .venv, upgrade pip, install requirements and run streamlit run app.py."),
        ("5.4 Hugging Face Token Setup", "Create a free Hugging Face account, open Settings, create an Access Token with read permission and set it as environment variable HF_TOKEN. The token must not be written inside source code."),
        ("5.5 Streamlit UI Implementation", "The Streamlit application uses tabs and sidebar controls. The Live Emotion Studio tab performs analysis. The Mascot Chat tab handles conversation. The Training tab performs EfficientNetV2-S training. The Twin Memory tab displays logs and catalog."),
        ("5.6 Text Emotion Implementation", "The implementation loads AutoTokenizer and AutoModelForSequenceClassification from transformers. It runs softmax inference using PyTorch tensors. If models cannot load, heuristic keywords keep the app functional."),
        ("5.7 Voice Emotion Implementation", "The implementation uses SpeechRecognition to process WAV audio. Voice tone is approximated through waveform energy. The transcript passes into text emotion analysis, allowing combined semantic and acoustic reasoning."),
        ("5.8 Facial Emotion Implementation", "The implementation uses OpenCV face detection and PyTorch EfficientNetV2-S classifier. It supports model checkpoints stored as emotion_efficientnet_v2_s.pth and metadata stored as JSON."),
        ("5.9 Grad-CAM Implementation", "The Grad-CAM function identifies the last convolutional module, registers hooks, computes gradients, averages them spatially, forms a heatmap and overlays it over the detected face region."),
        ("5.10 Animated Mascot Implementation", "The mascot is implemented through HTML and CSS inside Streamlit. Mood changes modify background color, face color, glow, eye style and mouth curve. It is lightweight and does not require a separate 3D engine."),
        ("5.11 Chatbot Response Implementation", "The chatbot response pipeline prepares a structured prompt containing the mood, tone, intent and twin summary. If Hugging Face API returns a response, it is displayed. Otherwise the local emotional coach generates a deterministic response."),
        ("5.12 Digital Twin Implementation", "The digital twin is a CSV log containing user, timestamp, modality moods, fused mood, score, recommendation, chat query, chat reply and feedback. It helps personalize later recommendations."),
        ("5.13 Training Pipeline", "The training pipeline reads image folders, constructs a FolderDataset, loads EfficientNetV2-S pretrained weights, optionally freezes the backbone, trains with AdamW and CrossEntropyLoss and saves a .pth checkpoint."),
        ("5.14 Screenshots to Capture", "The student should capture screenshots of the Streamlit home screen, webcam emotion result, Grad-CAM heatmap, mascot chat response, training metadata, twin memory table and setup guide tab after running the app."),
        ("5.15 Implementation Safety", "The implementation contains no hardcoded API key. Any token pasted during experimentation must be revoked and replaced with an environment variable. This is part of safe software engineering practice."),
    ]
    for title, body in chapter5:
        add(title, body)

    chapter6 = [
        ("CHAPTER 6: TESTING", "Testing verifies whether the software meets functional and non-functional requirements. Since the project is an AI-enabled application, testing includes execution checks, modality checks, fallback checks, ethical warnings and usability checks."),
        ("6.1 Test Strategy", "The strategy combines unit-style function checks, integration checks through Streamlit, manual UI testing, model fallback testing and dataset-path testing. The objective is demo readiness rather than production clinical validation."),
        ("6.2 Test Case 1: Application Launch", "Input: streamlit run app.py. Expected Result: Streamlit opens in the browser and shows all tabs. Status: Pass when dependencies are installed correctly."),
        ("6.3 Test Case 2: Text Emotion", "Input: I feel stressed about my exam. Expected Result: text mood becomes sad or energetic, intent becomes stress_relief or study_focus, confidence is shown and chatbot asks reflective questions."),
        ("6.4 Test Case 3: Voice Input", "Input: recorded WAV speech. Expected Result: transcript is shown when SpeechRecognition succeeds, energy is computed and voice mood contributes to fusion."),
        ("6.5 Test Case 4: Face Image", "Input: webcam capture. Expected Result: a face box is detected, method is trained_efficientnet_v2_s when checkpoint exists or opencv_heuristic otherwise, and an image is displayed."),
        ("6.6 Test Case 5: Fusion", "Input: happy emoji, positive text and smiling face. Expected Result: fused mood tends toward happy and mascot changes to happy visual state."),
        ("6.7 Test Case 6: Recommendation", "Input: fused sad mood. Expected Result: system recommends sad/comfort/calm resource and stores exposure in recommender_stats.csv."),
        ("6.8 Test Case 7: Chatbot", "Input: How can I handle pressure before presentation? Expected Result: response includes one reflection, exactly three questions and one next action."),
        ("6.9 Test Case 8: Training", "Input: dataset/train with at least two class folders. Expected Result: EfficientNetV2-S training starts and saves emotion_efficientnet_v2_s.pth."),
        ("6.10 Test Case 9: Token Missing", "Input: no HF_TOKEN set. Expected Result: local coach fallback works and application does not crash."),
        ("6.11 Test Case 10: Security", "Input: repository scan. Expected Result: no hardcoded hf_ token appears in app.py or report files."),
        ("6.12 Validation Metrics", "The training pipeline can compute accuracy through class prediction. Confusion matrix, precision, recall and F1 score can be added after a balanced validation dataset is available."),
        ("6.13 Risk and Mitigation", "Risks include camera permission failure, microphone failure, API downtime, insufficient dataset and overconfident interpretation. Mitigations include fallbacks, ethical warnings, local coach and clear non-diagnostic wording.")
    ]
    for title, body in chapter6:
        add(title, body)

    chapter7 = [
        ("CHAPTER 7: SUMMARY AND CONCLUSION", "The project successfully demonstrates a major-project-level AI system that combines multimodal emotion analysis and an animated mascot chatbot in a single Python file."),
        ("7.1 Summary", "The application integrates PyTorch EfficientNetV2-S, Grad-CAM, Hugging Face Transformers, NLTK, SpeechRecognition, OpenCV and Streamlit. It meets the requirement of avoiding TensorFlow, Keras and paid APIs."),
        ("7.2 Technical Achievement", "The technical achievement is the fusion of four input modes with transparent score display and a non-generic chatbot response engine. The system is explainable through Grad-CAM and auditable through CSV logs."),
        ("7.3 Academic Contribution", "The project connects deep learning, natural language processing, human-computer interaction, recommender systems and ethical AI. It is suitable for a viva because each module has a clear purpose and can be demonstrated separately."),
        ("7.4 Conclusion", "The Emotionally Intelligent Animated Mascot Chatbot provides a practical and innovative demonstration of cognitive emotional intelligence in a student-friendly environment. It is not a medical tool, but it shows how emotionally aware AI can provide safer and more personalized support.")
    ]
    for title, body in chapter7:
        add(title, body)

    chapter8 = [
        ("CHAPTER 8: LIMITATIONS OF THE PROJECT AND FUTURE WORK", "Every AI project has limitations. The proposed application is a strong academic prototype, but real-world deployment would require larger datasets, stronger validation, privacy review and user studies."),
        ("8.1 Limitations", "Facial emotion recognition can be biased by lighting, camera angle and dataset imbalance. Voice emotion estimation uses transcript and energy rather than a dedicated speech-emotion transformer. Hugging Face free API may be unavailable or rate limited. The mascot is CSS-based rather than a full VRM 3D model."),
        ("8.2 Future Work", "Future work can include full VRM avatar support, lip synchronization, multilingual emotion models, dedicated speech-emotion recognition, stronger reinforcement learning recommendation, FER2025 dataset training, mobile APK packaging and secure cloud deployment."),
        ("8.3 Mobile and APK Scope", "A free practical route is to host Streamlit and add it to Android home screen as a progressive app-like experience. A later WebView wrapper can convert the hosted URL into an Android APK."),
        ("8.4 Research Scope", "The system can be extended using larger multimodal datasets such as MER2023, MER2024 and recent facial expression datasets, subject to license and access. Evaluation can include macro F1, AUC, latency, usability and user satisfaction.")
    ]
    for title, body in chapter8:
        add(title, body)

    add("Bibliography",
        "[1] R. W. Picard, Affective Computing, MIT Press, 1997.",
        "[2] A. Vaswani et al., Attention Is All You Need, Advances in Neural Information Processing Systems, 2017.",
        "[3] M. Tan and Q. V. Le, EfficientNetV2: Smaller Models and Faster Training, International Conference on Machine Learning, 2021.",
        "[4] R. R. Selvaraju et al., Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization, IEEE International Conference on Computer Vision, 2017.",
        "[5] T. Wolf et al., Transformers: State-of-the-Art Natural Language Processing, Conference on Empirical Methods in Natural Language Processing: System Demonstrations, 2020.",
        "[6] S. Bird, E. Klein and E. Loper, Natural Language Processing with Python, O'Reilly Media, 2009.",
        "[7] I. J. Goodfellow et al., Challenges in Representation Learning: A Report on Three Machine Learning Contests, Neural Networks, 2015.",
        "[8] P. Ekman and W. V. Friesen, Constants across Cultures in the Face and Emotion, Journal of Personality and Social Psychology, 1971.",
        "[9] A. Paszke et al., PyTorch: An Imperative Style, High-Performance Deep Learning Library, Advances in Neural Information Processing Systems, 2019.",
        "[10] Streamlit Inc., Streamlit Documentation, 2026.")

    # Appendix pages with concise source-code listings and setup material.
    code_lines = APP_CODE.splitlines()
    chunk_size = 34
    for idx in range(0, min(len(code_lines), 850), chunk_size):
        chunk = "\n".join(f"{line_no + 1:04d}: {line}" for line_no, line in enumerate(code_lines[idx:idx + chunk_size], start=idx))
        add(f"Appendix A: Source Code Listing Page {idx // chunk_size + 1}", "File: app.py", "The following listing is part of the complete single-file implementation supplied with this report.", "```python\n" + chunk + "\n```")

    add("Appendix B: Requirements File", "requirements.txt", "```text\n" + REQUIREMENTS.strip() + "\n```")
    add("Appendix C: Dataset Folder Structure",
        "dataset/train/happy, dataset/train/sad, dataset/train/angry, dataset/train/neutral",
        "dataset/val/happy, dataset/val/sad, dataset/val/angry, dataset/val/neutral",
        "dataset/test/happy, dataset/test/sad, dataset/test/angry, dataset/test/neutral")
    add("Appendix D: Screenshot Checklist",
        "Capture these screenshots after running the application: home screen, live analysis, face Grad-CAM, mascot chat, training tab, twin memory table, setup guide and resource catalog download.")

    while len(pages) < 88:
        n = len(pages) + 1
        add(f"Appendix E: Extended Viva Notes {n}",
            "This page provides additional viva preparation notes. The student should explain why PyTorch was selected, how EfficientNetV2-S improves over older MobileNetV2 demonstrations, how Grad-CAM supports explainability, why Hugging Face is used as the free API option and why hardcoded API keys are unsafe.",
            "The examiner may ask how mood fusion works. The answer is that the system computes separate moods from face, voice, text and emoji inputs, multiplies them with selected weights and picks the mood with the highest weighted score. This keeps the decision process interpretable.")

    return pages


def render_markdown(pages: list[tuple[str, list[str]]]) -> str:
    out = ["<!-- GGSIPU Major Project - Dissertation Report -->", ""]
    for index, (title, body) in enumerate(pages, start=1):
        out.append(f"<!-- Report Page {index} -->")
        out.append(f"# {title}")
        out.append("")
        for item in body:
            if item.startswith("```"):
                out.append(item)
            else:
                out.append(para(item))
                out.append("")
        if index != len(pages):
            out.append('<div style="page-break-after: always;"></div>')
            out.append("")
    return "\n".join(out)


def render_html(pages: list[tuple[str, list[str]]]) -> str:
    page_html = []
    for index, (title, body) in enumerate(pages, start=1):
        body_html = []
        for item in body:
            if item.startswith("```"):
                code = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", item.strip(), flags=re.MULTILINE)
                body_html.append(f"<pre>{html.escape(code)}</pre>")
            else:
                body_html.append(f"<p>{html.escape(item)}</p>")
        footer = "" if index == 1 else f'<footer><span>DEPARTMENT OF AIML</span><span>{"i" if index < 8 else index - 7}</span></footer>'
        page_html.append(f'<section class="page"><h1>{html.escape(title)}</h1>{"".join(body_html)}{footer}</section>')
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>GGSIPU Major Project Report</title>
<style>
@page {{ size: A4; margin: 1in 1in 1in 1.5in; }}
body {{ font-family: "Times New Roman", Times, serif; margin:0; background:#f4f4f4; }}
.page {{ width: 210mm; min-height: 297mm; box-sizing: border-box; padding: 1in 1in 1in 1.5in; margin: 0 auto 12px auto; background:white; position:relative; page-break-after: always; }}
h1 {{ font-size:16pt; font-weight:bold; text-align:left; margin-top:0; }}
p {{ font-size:12pt; line-height:1.5; text-align:justify; }}
pre {{ white-space:pre-wrap; font-family:"Courier New", monospace; font-size:8.5pt; line-height:1.15; border:1px solid #ddd; padding:8px; }}
footer {{ position:absolute; left:1.5in; right:1in; bottom:.5in; font-size:10pt; display:flex; justify-content:space-between; }}
@media print {{ body {{ background:white; }} .page {{ margin:0; box-shadow:none; }} }}
</style></head><body>{"".join(page_html)}</body></html>"""


def xml_escape(text: str) -> str:
    return html.escape(text, quote=False)


def docx_paragraph(text: str, style: str | None = None, page_break: bool = False) -> str:
    if page_break:
        return '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'
    style_xml = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    lines = text.split("\n") or [""]
    runs = []
    for i, line in enumerate(lines):
        if i:
            runs.append("<w:r><w:br/></w:r>")
        runs.append(f"<w:r><w:t xml:space=\"preserve\">{xml_escape(line)}</w:t></w:r>")
    return f"<w:p>{style_xml}{''.join(runs)}</w:p>"


def write_docx(pages: list[tuple[str, list[str]]], path: Path) -> None:
    body_parts = []
    for idx, (title, body) in enumerate(pages, start=1):
        if idx > 1:
            body_parts.append(docx_paragraph("", page_break=True))
        body_parts.append(docx_paragraph(title, "Heading1"))
        for item in body:
            if item.startswith("```"):
                code = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", item.strip(), flags=re.MULTILINE)
                for chunk in textwrap.wrap(code, 105, replace_whitespace=False, drop_whitespace=False):
                    body_parts.append(docx_paragraph(chunk, "Code"))
            else:
                body_parts.append(docx_paragraph(item, "Normal"))
    sect = '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="2160" w:header="720" w:footer="720" w:gutter="0"/></w:sectPr>'
    document = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>{''.join(body_parts)}{sect}</w:body></w:document>'''
    styles = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:style w:type="paragraph" w:styleId="Normal"><w:name w:val="Normal"/><w:pPr><w:spacing w:line="360" w:lineRule="auto"/><w:jc w:val="both"/></w:pPr><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="24"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:pPr><w:keepNext/><w:spacing w:after="240"/><w:outlineLvl w:val="0"/></w:pPr><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:b/><w:sz w:val="32"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Code"><w:name w:val="Code"/><w:pPr><w:spacing w:line="240" w:lineRule="auto"/></w:pPr><w:rPr><w:rFonts w:ascii="Courier New" w:hAnsi="Courier New"/><w:sz w:val="17"/></w:rPr></w:style>
</w:styles>'''
    content_types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>'''
    rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>'''
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as docx:
        docx.writestr("[Content_Types].xml", content_types)
        docx.writestr("_rels/.rels", rels)
        docx.writestr("word/document.xml", document)
        docx.writestr("word/styles.xml", styles)


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    (ROOT / "app.py").write_text(APP_CODE, encoding="utf-8")
    (ROOT / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    (ROOT / ".gitignore").write_text("__pycache__/\n.venv/\nmodels/\ndataset/\n*.pth\ncei_twin_log.csv\nrecommender_stats.csv\nresource_catalog.csv\n", encoding="utf-8")
    pages = build_report_pages()
    (DOCS / "GGSIPU_Major_Project_Report.md").write_text(render_markdown(pages), encoding="utf-8")
    (DOCS / "GGSIPU_Major_Project_Report.html").write_text(render_html(pages), encoding="utf-8")
    write_docx(pages, DOCS / "GGSIPU_Major_Project_Report.docx")
    bundle_path = DOCS / "GGSIPU_Major_Project_Deliverables.zip"
    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as bundle:
        for source, arcname in [
            (DOCS / "GGSIPU_Major_Project_Report.docx", "GGSIPU_Major_Project_Report.docx"),
            (DOCS / "GGSIPU_Major_Project_Report.html", "GGSIPU_Major_Project_Report.html"),
            (DOCS / "GGSIPU_Major_Project_Report.md", "GGSIPU_Major_Project_Report.md"),
            (ROOT / "app.py", "app.py"),
            (ROOT / "requirements.txt", "requirements.txt"),
            (ROOT / "README.md", "README.md"),
        ]:
            if source.exists():
                bundle.write(source, arcname)
    (DOCS / "download_reference.md").write_text(textwrap.dedent("""\
    # Downloadable Reference Links

    Use these repository files as downloadable references after cloning or opening the PR branch:

    - [Complete Deliverables ZIP](./GGSIPU_Major_Project_Deliverables.zip)
    - [GGSIPU Major Project Report DOCX](./GGSIPU_Major_Project_Report.docx)
    - [GGSIPU Major Project Report HTML](./GGSIPU_Major_Project_Report.html)
    - [GGSIPU Major Project Report Markdown](./GGSIPU_Major_Project_Report.md)
    - [Complete Single File Python Implementation](../app.py)
    - [Requirements File](../requirements.txt)

    The DOCX and HTML files are formatted for A4 single-side printing with the GGSIPU Major Project sequence.
    Replace student, enrollment, supervisor, institute and department placeholders before final submission.
    """), encoding="utf-8")
    (ROOT / "README.md").write_text(textwrap.dedent("""\
    # GuneeshDua Major Project

    Cognitive Emotional Intelligence, reinforcement-learning-style recommendation logic, digital emotional twin logging, multimodal AI fusion and ethical AI monitoring.

    ## Generated deliverables

    - `app.py`: complete single-file Python 3 Streamlit implementation using PyTorch, not TensorFlow/Keras.
    - `requirements.txt`: dependencies for execution in VS Code.
    - `docs/GGSIPU_Major_Project_Report.docx`: downloadable GGSIPU Major Project - Dissertation report.
    - `docs/GGSIPU_Major_Project_Report.html`: printable A4 HTML version.
    - `docs/GGSIPU_Major_Project_Report.md`: editable markdown source.
    - `docs/GGSIPU_Major_Project_Deliverables.zip`: downloadable bundle containing the report and code.
    - `docs/download_reference.md`: local download reference links.

    ## Run

    ```powershell
    python -m venv .venv
    .\\.venv\\Scripts\\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    $env:HF_TOKEN="your_hugging_face_token"
    streamlit run app.py
    ```

    Keep Hugging Face tokens in environment variables only. Do not hardcode API tokens in source files.
    """), encoding="utf-8")
    print(f"Generated {len(pages)} report pages and project files.")


if __name__ == "__main__":
    main()
