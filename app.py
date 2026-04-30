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
