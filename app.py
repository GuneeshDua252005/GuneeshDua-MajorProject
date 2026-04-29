from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote_plus

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, ImageOps
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
import streamlit as st
import streamlit.components.v1 as components

try:
    from huggingface_hub import InferenceClient
except Exception:
    InferenceClient = None

try:
    import tensorflow as tf
    from tensorflow import keras
except Exception:
    tf = None
    keras = None

try:
    from datasets import ClassLabel, load_dataset
except Exception:
    ClassLabel = None
    load_dataset = None


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
MODELS_DIR = ROOT / "models"
DOCS_DIR = ROOT / "docs"
MODEL_PATH = MODELS_DIR / "efficientnetv2_emotion.keras"
MODEL_META_PATH = MODELS_DIR / "efficientnetv2_emotion_metadata.json"
TWIN_LOG_PATH = ROOT / "cei_twin_log.csv"
RECOMMENDER_STATS_PATH = ROOT / "recommender_stats.csv"
RESOURCE_CATALOG_PATH = ROOT / "resource_catalog.csv"
DATASET_MANIFEST_PATH = DATASET_DIR / "dataset_manifest.json"

TWIN_LOG_FIELDNAMES = [
    "timestamp",
    "username",
    "user_hash",
    "face_emotion",
    "face_confidence",
    "text_emotion",
    "voice_emotion",
    "emoji_emotion",
    "fused_emotion",
    "final_emotion_after_override",
    "confidence",
    "valence",
    "arousal",
    "stress",
    "used_sources",
    "override_emotion",
    "context_excerpt",
    "voice_excerpt",
    "consent_to_log",
]
DEFAULT_TWIN_SNAPSHOT = {
    "dominant_emotion": "neutral",
    "avg_valence": 0.0,
    "avg_arousal": 0.0,
    "avg_stress": 0.0,
    "entries": 0,
    "volatility": 0.0,
}

APP_TITLE = "Cognitive Emotion Intelligence & Adaptive Lifestyle System"
APP_TAGLINE = (
    "A single-file Streamlit major-project prototype with multimodal emotion fusion, "
    "Grad-CAM explainability, digital emotional twin logging, ethical AI monitoring, "
    "history-aware recommendation, and free Hugging Face integrations."
)

DEFAULT_IMAGE_SIZE = 224
UNIFIED_EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

EMOTION_GEOMETRY = {
    "happy": (0.90, 0.65),
    "sad": (-0.82, 0.22),
    "angry": (-0.90, 0.85),
    "fear": (-0.85, 0.92),
    "surprise": (0.35, 0.95),
    "neutral": (0.00, 0.32),
    "disgust": (-0.65, 0.55),
}

EMOJI_PRIORS: Dict[str, Dict[str, float]] = {
    "None": {"neutral": 1.0},
    "😀": {"happy": 0.86, "surprise": 0.08, "neutral": 0.06},
    "😢": {"sad": 0.84, "fear": 0.08, "neutral": 0.08},
    "😡": {"angry": 0.88, "disgust": 0.06, "fear": 0.06},
    "😨": {"fear": 0.84, "surprise": 0.10, "sad": 0.06},
    "😐": {"neutral": 0.88, "sad": 0.06, "disgust": 0.06},
    "😲": {"surprise": 0.88, "fear": 0.08, "happy": 0.04},
    "😌": {"neutral": 0.72, "happy": 0.18, "sad": 0.10},
    "😫": {"sad": 0.42, "fear": 0.28, "angry": 0.18, "disgust": 0.12},
    "🤔": {"neutral": 0.58, "surprise": 0.20, "sad": 0.10, "fear": 0.12},
}

TEXT_KEYWORDS = {
    "happy": {
        "happy",
        "joy",
        "joyful",
        "grateful",
        "good",
        "great",
        "excited",
        "motivated",
        "energetic",
        "confident",
        "positive",
        "cheerful",
    },
    "sad": {
        "sad",
        "lonely",
        "down",
        "cry",
        "crying",
        "hurt",
        "upset",
        "broken",
        "tired",
        "exhausted",
        "hopeless",
        "empty",
    },
    "angry": {
        "angry",
        "mad",
        "frustrated",
        "irritated",
        "annoyed",
        "rage",
        "hate",
        "furious",
        "resentful",
    },
    "fear": {
        "fear",
        "afraid",
        "scared",
        "stress",
        "stressed",
        "anxious",
        "panic",
        "nervous",
        "worried",
        "pressure",
        "tense",
        "uncertain",
    },
    "surprise": {
        "surprised",
        "shock",
        "unexpected",
        "wow",
        "sudden",
        "amazed",
        "astonished",
    },
    "disgust": {
        "disgust",
        "gross",
        "sick",
        "nasty",
        "toxic",
        "repulsed",
    },
    "neutral": {
        "okay",
        "fine",
        "normal",
        "steady",
        "average",
        "balanced",
        "neutral",
    },
}

MOOD_BLUEPRINTS = {
    "happy": {
        "channels": [
            ("LoFi Girl", "https://www.youtube.com/@LofiGirl"),
            ("The Jazz Hop Cafe", "https://www.youtube.com/@TheJazzHopCafe"),
        ],
        "music": "upbeat positive instrumental music",
        "breathing": "gratitude journaling music",
        "podcast": "positive mindset podcast",
        "focus": "happy focus playlist",
        "offline": "Write down three wins from today and take a 10-minute gratitude walk.",
    },
    "sad": {
        "channels": [
            ("Mindful Peace", "https://www.youtube.com/@MindfulPeace"),
            ("PowerThoughts Meditation Club", "https://www.youtube.com/@PowerThoughtsMeditationClub"),
        ],
        "music": "gentle recovery piano music",
        "breathing": "self compassion breathing exercise",
        "podcast": "emotional resilience podcast",
        "focus": "calm healing playlist",
        "offline": "Use a low-pressure reset routine: water, slow breathing, sunlight, and one supportive call.",
    },
    "angry": {
        "channels": [
            ("Jason Lewis - Mind Amend", "https://www.youtube.com/@jasonstephensonmeditation"),
            ("Relax for a While", "https://www.youtube.com/@Relaxforawhile"),
        ],
        "music": "anger relief instrumental music",
        "breathing": "box breathing for anger control",
        "podcast": "stress management podcast",
        "focus": "cooldown ambient music",
        "offline": "Pause before reacting, walk for 5 minutes, and avoid making major decisions immediately.",
    },
    "fear": {
        "channels": [
            ("The Honest Guys", "https://www.youtube.com/@TheHonestGuys"),
            ("Yellow Brick Cinema", "https://www.youtube.com/@YellowBrickCinema"),
        ],
        "music": "anxiety relief music",
        "breathing": "grounding exercise for anxiety",
        "podcast": "confidence building podcast",
        "focus": "low stimulation calm playlist",
        "offline": "Anchor attention to one controllable task and use the 5-4-3-2-1 grounding method.",
    },
    "surprise": {
        "channels": [
            ("TEDx Talks", "https://www.youtube.com/@TEDx"),
            ("CrashCourse", "https://www.youtube.com/@crashcourse"),
        ],
        "music": "curious discovery playlist",
        "breathing": "reset breathing for overstimulation",
        "podcast": "innovation podcast",
        "focus": "curiosity music mix",
        "offline": "Capture the unexpected event in a notebook and decide whether it is a threat, opportunity, or neutral event.",
    },
    "neutral": {
        "channels": [
            ("Huberman Lab", "https://www.youtube.com/@hubermanlab"),
            ("Ali Abdaal", "https://www.youtube.com/@aliabdaal"),
        ],
        "music": "steady focus music",
        "breathing": "mindful breathing for clarity",
        "podcast": "focus productivity podcast",
        "focus": "deep work playlist",
        "offline": "Use neutral moments for planning, hydration, and setting one realistic target for the next hour.",
    },
    "calm": {
        "channels": [
            ("Meditative Mind", "https://www.youtube.com/@MeditativeMind"),
            ("Calm", "https://www.youtube.com/@calm"),
        ],
        "music": "relaxing ambient meditation music",
        "breathing": "slow breathing relaxation",
        "podcast": "mindfulness podcast",
        "focus": "peaceful ambient playlist",
        "offline": "Protect the calm state by avoiding multitasking and preserving the next quiet block.",
    },
    "focused": {
        "channels": [
            ("Thomas Frank", "https://www.youtube.com/@Thomasfrank"),
            ("Mariana's Study Corner", "https://www.youtube.com/@MarianasStudyCorner"),
        ],
        "music": "study concentration music",
        "breathing": "focus breathing routine",
        "podcast": "learning science podcast",
        "focus": "pomodoro instrumental playlist",
        "offline": "Enter a 25-minute deep-work block and silence nonessential notifications.",
    },
    "motivated": {
        "channels": [
            ("Improvement Pill", "https://www.youtube.com/@ImprovementPill"),
            ("TED", "https://www.youtube.com/@TED"),
        ],
        "music": "motivational cinematic instrumental",
        "breathing": "energizing breathing exercise",
        "podcast": "career growth podcast",
        "focus": "high energy work playlist",
        "offline": "Convert motivation into action by choosing one measurable task with a visible output.",
    },
    "lonely": {
        "channels": [
            ("Psych2Go", "https://www.youtube.com/@Psych2go"),
            ("HealthyGamerGG", "https://www.youtube.com/@HealthyGamerGG"),
        ],
        "music": "comfort acoustic playlist",
        "breathing": "self support meditation",
        "podcast": "mental wellbeing podcast",
        "focus": "comforting mellow playlist",
        "offline": "Reach out to one trusted person or visit a shared study or work space.",
    },
    "tired": {
        "channels": [
            ("Yoga With Adriene", "https://www.youtube.com/@yogawithadriene"),
            ("Walk at Home by Leslie Sansone", "https://www.youtube.com/@LeslieSansonesWalkatHome"),
        ],
        "music": "gentle recharge music",
        "breathing": "morning activation breathwork",
        "podcast": "sleep recovery podcast",
        "focus": "low intensity focus playlist",
        "offline": "Choose recovery before intensity: hydration, stretching, posture reset, and a lighter task list.",
    },
    "hopeful": {
        "channels": [
            ("Kurzgesagt", "https://www.youtube.com/@kurzgesagt"),
            ("Goalcast", "https://www.youtube.com/@Goalcast"),
        ],
        "music": "hopeful uplifting piano",
        "breathing": "visualization breathing exercise",
        "podcast": "personal growth podcast",
        "focus": "future planning playlist",
        "offline": "Record one thing improving in your life and one next step you can control.",
    },
    "confident": {
        "channels": [
            ("Matt D'Avella", "https://www.youtube.com/@mattdavella"),
            ("FightMediocrity", "https://www.youtube.com/@FightMediocrity"),
        ],
        "music": "confidence boost playlist",
        "breathing": "performance breathing routine",
        "podcast": "leadership podcast",
        "focus": "executive focus playlist",
        "offline": "Use the confident state to tackle the highest-value task, not the easiest one.",
    },
    "curious": {
        "channels": [
            ("Veritasium", "https://www.youtube.com/@veritasium"),
            ("Big Think", "https://www.youtube.com/@bigthink"),
        ],
        "music": "creative curiosity playlist",
        "breathing": "brain reset breathing",
        "podcast": "science ideas podcast",
        "focus": "creative thinking soundtrack",
        "offline": "Turn curiosity into learning by capturing one new question and researching it deeply.",
    },
    "overwhelmed": {
        "channels": [
            ("Headspace", "https://www.youtube.com/@Headspace"),
            ("Lavendaire", "https://www.youtube.com/@lavendaire"),
        ],
        "music": "stress relief nature music",
        "breathing": "2 minute nervous system reset",
        "podcast": "burnout recovery podcast",
        "focus": "gentle reset playlist",
        "offline": "Reduce cognitive load: remove one noncritical task, then finish the smallest useful task first.",
    },
    "grateful": {
        "channels": [
            ("The School of Life", "https://www.youtube.com/@theschooloflifetv"),
            ("Jay Shetty Podcast", "https://www.youtube.com/@JayShettyPodcast"),
        ],
        "music": "gratitude reflection playlist",
        "breathing": "gratitude meditation",
        "podcast": "meaningful life podcast",
        "focus": "warm reflective playlist",
        "offline": "Write one appreciation note or thank one person directly.",
    },
}

EMOTION_TO_CATALOG_MOODS = {
    "happy": ["happy", "grateful", "confident", "motivated"],
    "sad": ["sad", "hopeful", "calm", "lonely"],
    "angry": ["angry", "calm", "focused", "overwhelmed"],
    "fear": ["fear", "calm", "hopeful", "overwhelmed"],
    "surprise": ["surprise", "curious", "focused", "happy"],
    "neutral": ["neutral", "focused", "calm", "motivated"],
    "disgust": ["neutral", "calm", "focused", "overwhelmed"],
}

TOP_VIVA_QUESTIONS = [
    "Why did you choose EfficientNetV2 with Global Average Pooling instead of an older CNN like MobileNetV2?",
    "How does Grad-CAM make the emotion classifier explainable?",
    "What is a Digital Emotional Twin and how is it used in your project?",
    "How does the multi-modal fusion pipeline combine face, text, emoji, and voice?",
    "Where exactly is reinforcement learning logic used in the system?",
    "How does ethical AI monitoring protect the user?",
    "How does your system overcome research gaps from 2023-2025 emotion-aware lifestyle systems?",
    "What happens when the AI is uncertain or when the user disagrees with the prediction?",
    "Why did you use Hugging Face as the free AI integration layer?",
    "How can this Streamlit project be converted into an installable Android-style experience for free?",
]

DATASET_SOURCES = {
    "FER2013": {
        "repo_id": "AutumnQiu/fer2013",
        "preferred_splits": ["train", "validation", "test"],
        "streaming": False,
    },
    "EmoNet-Face-Binary": {
        "repo_id": "laion/emonet-face-binary",
        "preferred_splits": ["train", "validation", "test"],
        "streaming": True,
    },
    "EmoNet-Face-HQ": {
        "repo_id": "laion/emonet-face-hq",
        "preferred_splits": ["train", "validation", "test"],
        "streaming": True,
    },
}


@dataclass
class EmotionSignal:
    source: str
    scores: Dict[str, float]
    confidence: float
    note: str
    evidence: str = ""

    @property
    def top_emotion(self) -> str:
        return max(self.scores, key=self.scores.get)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def ensure_runtime_dirs() -> None:
    DATASET_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    DOCS_DIR.mkdir(exist_ok=True)


def blank_scores() -> Dict[str, float]:
    return {emotion: 0.0 for emotion in UNIFIED_EMOTIONS}


def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    normalized = blank_scores()
    for emotion in UNIFIED_EMOTIONS:
        normalized[emotion] = max(0.0, float(scores.get(emotion, 0.0)))
    total = sum(normalized.values())
    if total <= 0:
        normalized["neutral"] = 1.0
        return normalized
    return {emotion: value / total for emotion, value in normalized.items()}


def top_prediction(scores: Dict[str, float]) -> Tuple[str, float]:
    normalized = normalize_scores(scores)
    emotion = max(normalized, key=normalized.get)
    return emotion, float(normalized[emotion])


def second_prediction_gap(scores: Dict[str, float]) -> float:
    values = sorted(normalize_scores(scores).values(), reverse=True)
    if len(values) < 2:
        return 0.0
    return float(values[0] - values[1])


def make_user_hash(username: str, private_key: str) -> str:
    return hashlib.sha256(f"{username.strip().lower()}::{private_key}".encode("utf-8")).hexdigest()[:16]


def map_label_to_unified(raw_label: Any) -> str:
    label = re.sub(r"[^a-z0-9_ -]", "", str(raw_label).strip().lower())
    label = label.replace("-", " ").replace("_", " ")
    aliases = {
        "anger": "angry",
        "angry": "angry",
        "disgust": "disgust",
        "disgusted": "disgust",
        "fear": "fear",
        "fearful": "fear",
        "scared": "fear",
        "happy": "happy",
        "happiness": "happy",
        "joy": "happy",
        "neutral": "neutral",
        "calm": "neutral",
        "relaxed": "neutral",
        "sad": "sad",
        "sadness": "sad",
        "surprise": "surprise",
        "surprised": "surprise",
        "stressed": "fear",
        "stress": "fear",
        "anxious": "fear",
        "bored": "sad",
        "contempt": "disgust",
    }
    return aliases.get(label, "neutral")


def tokenize_text(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def calculate_valence_arousal(scores: Dict[str, float]) -> Tuple[float, float]:
    normalized = normalize_scores(scores)
    valence = 0.0
    arousal = 0.0
    for emotion, weight in normalized.items():
        emotion_valence, emotion_arousal = EMOTION_GEOMETRY[emotion]
        valence += emotion_valence * weight
        arousal += emotion_arousal * weight
    return round(valence, 3), round(arousal, 3)


def estimate_stress_from_scores(scores: Dict[str, float]) -> float:
    normalized = normalize_scores(scores)
    stress = normalized["fear"] * 0.45 + normalized["angry"] * 0.35 + normalized["sad"] * 0.20
    return round(float(stress), 3)


def analyze_text_emotion(text: str, source: str = "text") -> Optional[EmotionSignal]:
    text = (text or "").strip()
    if not text:
        return None

    tokens = tokenize_text(text)
    scores = blank_scores()
    matched: List[str] = []

    for emotion, keywords in TEXT_KEYWORDS.items():
        for token in tokens:
            if token in keywords:
                scores[emotion] += 1.0
                matched.append(token)

    exclamations = text.count("!")
    upper_ratio = 0.0
    letters = [char for char in text if char.isalpha()]
    if letters:
        upper_ratio = sum(1 for char in letters if char.isupper()) / len(letters)
    if exclamations:
        scores["surprise"] += min(1.5, exclamations * 0.2)
    if upper_ratio > 0.30:
        scores["angry"] += 0.6
        scores["fear"] += 0.2

    if not any(scores.values()):
        scores["neutral"] = 1.0

    normalized = normalize_scores(scores)
    confidence = min(0.88, 0.30 + (len(matched) * 0.08) + exclamations * 0.03)
    note = (
        "Keyword and linguistic heuristic used for text emotion inference."
        if matched
        else "No strong emotion keywords found; text treated as mostly neutral."
    )
    evidence = ", ".join(sorted(set(matched))[:8]) if matched else "No salient tokens"
    return EmotionSignal(source=source, scores=normalized, confidence=round(confidence, 3), note=note, evidence=evidence)


def analyze_emoji_emotion(emoji: str) -> Optional[EmotionSignal]:
    if not emoji or emoji not in EMOJI_PRIORS or emoji == "None":
        return None
    scores = normalize_scores(EMOJI_PRIORS[emoji])
    return EmotionSignal(
        source="emoji",
        scores=scores,
        confidence=0.72,
        note="User-selected emoji prior.",
        evidence=emoji,
    )


def make_hf_client(token: Optional[str], model: Optional[str] = None):
    if InferenceClient is None:
        raise RuntimeError("huggingface_hub is not installed.")
    try:
        return InferenceClient(model=model, api_key=token) if token else InferenceClient(model=model)
    except TypeError:
        return InferenceClient(model=model, token=token) if token else InferenceClient(model=model)


def load_local_emotion_model() -> Tuple[Optional[Any], Dict[str, Any]]:
    if tf is None or not MODEL_PATH.exists() or not MODEL_META_PATH.exists():
        return None, {}
    model = keras.models.load_model(MODEL_PATH, compile=False)
    metadata = json.loads(MODEL_META_PATH.read_text(encoding="utf-8"))
    return model, metadata


def predict_local_face_emotion(image: Image.Image) -> Optional[EmotionSignal]:
    model, metadata = load_local_emotion_model()
    if model is None:
        return None

    image_size = int(metadata.get("image_size", DEFAULT_IMAGE_SIZE))
    class_names = metadata.get("class_names", UNIFIED_EMOTIONS)
    rgb = ImageOps.fit(image.convert("RGB"), (image_size, image_size))
    batch = np.expand_dims(np.array(rgb).astype("float32"), axis=0)
    probs = model.predict(batch, verbose=0)[0]
    scores = blank_scores()
    for class_name, score in zip(class_names, probs):
        scores[map_label_to_unified(class_name)] += float(score)
    normalized = normalize_scores(scores)
    _, confidence = top_prediction(normalized)
    return EmotionSignal(
        source="face",
        scores=normalized,
        confidence=round(confidence, 3),
        note="Prediction from local EfficientNetV2 emotion classifier.",
        evidence=metadata.get("backbone_name", "efficientnetv2"),
    )


def predict_remote_face_emotion(image: Image.Image, token: str, model_id: str) -> Optional[EmotionSignal]:
    if not token or InferenceClient is None:
        return None

    client = make_hf_client(token=token)
    try:
        output = client.image_classification(image.convert("RGB"), model=model_id)
    except Exception:
        return None

    scores = blank_scores()
    for item in output:
        label = getattr(item, "label", None)
        score = getattr(item, "score", None)
        if isinstance(item, dict):
            label = item.get("label")
            score = item.get("score")
        if label is None or score is None:
            continue
        scores[map_label_to_unified(label)] += float(score)

    normalized = normalize_scores(scores)
    _, confidence = top_prediction(normalized)
    return EmotionSignal(
        source="face",
        scores=normalized,
        confidence=round(confidence, 3),
        note=f"Remote facial emotion inference via Hugging Face model `{model_id}`.",
        evidence="huggingface-image-classification",
    )


def heuristic_face_emotion(image: Image.Image) -> EmotionSignal:
    rgb = ImageOps.fit(image.convert("RGB"), (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE))
    pixels = np.array(rgb).astype("float32")
    gray = pixels.mean(axis=2)
    brightness = gray.mean() / 255.0
    contrast = gray.std() / 255.0
    saturation = ((pixels.max(axis=2) - pixels.min(axis=2)).mean()) / 255.0
    edge_v = np.abs(np.diff(gray, axis=0)).mean() / 255.0
    edge_h = np.abs(np.diff(gray, axis=1)).mean() / 255.0
    warmth = (pixels[:, :, 0].mean() - pixels[:, :, 2].mean()) / 255.0

    scores = {
        "happy": max(0.0, brightness * 0.8 + saturation * 0.5 + max(0.0, warmth) * 0.4),
        "sad": max(0.0, (1.0 - brightness) * 0.85 + (1.0 - saturation) * 0.35),
        "angry": max(0.0, contrast * 0.85 + (edge_v + edge_h) * 0.55 + max(0.0, warmth) * 0.25),
        "fear": max(0.0, contrast * 0.45 + (edge_v + edge_h) * 0.75),
        "surprise": max(0.0, brightness * 0.45 + contrast * 0.65),
        "disgust": max(0.0, contrast * 0.25 + (1.0 - saturation) * 0.15),
        "neutral": 0.35 + (1.0 - abs(brightness - 0.5)) * 0.25,
    }
    normalized = normalize_scores(scores)
    return EmotionSignal(
        source="face",
        scores=normalized,
        confidence=0.36,
        note="Low-cost local heuristic used because no trained CNN or HF token is available.",
        evidence=f"brightness={brightness:.2f}, contrast={contrast:.2f}, saturation={saturation:.2f}",
    )


def analyze_face_image(image: Optional[Image.Image], hf_token: str, face_model_id: str) -> Optional[EmotionSignal]:
    if image is None:
        return None
    local_result = predict_local_face_emotion(image)
    if local_result is not None:
        return local_result
    remote_result = predict_remote_face_emotion(image, hf_token, face_model_id) if hf_token else None
    if remote_result is not None:
        return remote_result
    return heuristic_face_emotion(image)


def transcribe_audio_with_hf(audio_bytes: bytes, token: str, asr_model_id: str) -> Tuple[str, str]:
    if not token or not audio_bytes:
        return "", "No Hugging Face token was supplied, so voice transcription was skipped."
    if InferenceClient is None:
        return "", "huggingface_hub is not installed."

    client = make_hf_client(token=token)
    try:
        output = client.automatic_speech_recognition(audio_bytes, model=asr_model_id or None)
        if isinstance(output, dict):
            text = output.get("text", "")
        else:
            text = getattr(output, "text", "")
        return text.strip(), f"Audio transcribed with `{asr_model_id or 'recommended ASR model'}`."
    except Exception as exc:
        return "", f"ASR failed: {exc}"


def fuse_modalities(signals: Iterable[Optional[EmotionSignal]]) -> Dict[str, Any]:
    filtered = [signal for signal in signals if signal is not None]
    if not filtered:
        scores = normalize_scores({"neutral": 1.0})
        valence, arousal = calculate_valence_arousal(scores)
        return {
            "scores": scores,
            "final_emotion": "neutral",
            "confidence": 0.0,
            "valence": valence,
            "arousal": arousal,
            "stress": 0.0,
            "used_sources": [],
            "notes": ["No valid modalities were provided."],
            "gap": 0.0,
        }

    base_weight = {"face": 0.42, "text": 0.26, "voice": 0.18, "emoji": 0.14}
    fused = blank_scores()
    used_sources = []
    notes = []

    for signal in filtered:
        weight = base_weight.get(signal.source, 0.15) * (0.60 + signal.confidence)
        used_sources.append(signal.source)
        notes.append(f"{signal.source}: {signal.note}")
        for emotion, probability in signal.scores.items():
            fused[emotion] += weight * probability

    fused = normalize_scores(fused)
    final_emotion, confidence = top_prediction(fused)
    valence, arousal = calculate_valence_arousal(fused)
    return {
        "scores": fused,
        "final_emotion": final_emotion,
        "confidence": round(confidence, 3),
        "valence": valence,
        "arousal": arousal,
        "stress": estimate_stress_from_scores(fused),
        "used_sources": used_sources,
        "notes": notes,
        "gap": round(second_prediction_gap(fused), 3),
    }


def ethical_monitor(fusion: Dict[str, Any], signals: List[Optional[EmotionSignal]]) -> List[str]:
    warnings: List[str] = []
    signal_count = sum(1 for signal in signals if signal is not None)

    if signal_count < 2:
        warnings.append("Only one modality is active, so the fused emotion is low-context and should be treated cautiously.")
    if fusion["confidence"] < 0.55:
        warnings.append("Overall confidence is modest, so the system should be used for reflection and support rather than hard judgment.")
    if fusion["gap"] < 0.10:
        warnings.append("The top two emotions are close together, which indicates emotional ambiguity.")
    if any(signal and signal.source == "face" and "heuristic" in signal.note.lower() for signal in signals):
        warnings.append("Face analysis is currently heuristic because no local CNN or Hugging Face image model result was available.")
    return warnings


def build_resource_catalog() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for mood, blueprint in MOOD_BLUEPRINTS.items():
        for index, (title, url) in enumerate(blueprint["channels"], start=1):
            rows.append(
                {
                    "resource_id": f"{mood}-yt-direct-{index}",
                    "mood": mood,
                    "title": f"{title} channel for {mood} regulation",
                    "url": url,
                    "source": "YouTube",
                    "type": "direct_channel",
                    "offline_fallback": blueprint["offline"],
                }
            )

        search_entries = [
            ("YouTube", "search_music", blueprint["music"], "https://www.youtube.com/results?search_query={}"),
            ("YouTube", "search_breathing", blueprint["breathing"], "https://www.youtube.com/results?search_query={}"),
            ("Spotify", "search_music", blueprint["music"], "https://open.spotify.com/search/{}"),
            ("Spotify", "search_podcast", blueprint["podcast"], "https://open.spotify.com/search/{}"),
            ("YouTube Music", "search_focus", blueprint["focus"], "https://music.youtube.com/search?q={}"),
        ]

        for offset, (source, resource_type, query, template) in enumerate(search_entries, start=1):
            rows.append(
                {
                    "resource_id": f"{mood}-{source.lower().replace(' ', '-')}-{offset}",
                    "mood": mood,
                    "title": f"{source} search for {query}",
                    "url": template.format(quote_plus(query)),
                    "source": source,
                    "type": resource_type,
                    "offline_fallback": blueprint["offline"],
                }
            )

    return pd.DataFrame(rows)


def export_resource_catalog(path: Path = RESOURCE_CATALOG_PATH) -> Path:
    ensure_runtime_dirs()
    catalog = build_resource_catalog()
    catalog.to_csv(path, index=False)
    return path


def load_recommender_stats() -> pd.DataFrame:
    if RECOMMENDER_STATS_PATH.exists():
        return pd.read_csv(RECOMMENDER_STATS_PATH)
    return pd.DataFrame(
        columns=[
            "resource_id",
            "shown",
            "helpful",
            "skipped",
            "total_reward",
            "q_value",
            "last_served_at",
            "last_feedback",
        ]
    )


def save_recommender_stats(df: pd.DataFrame) -> None:
    df.to_csv(RECOMMENDER_STATS_PATH, index=False)


def mark_resources_served(resource_ids: List[str]) -> None:
    stats = load_recommender_stats()
    now = utc_now()
    for resource_id in resource_ids:
        if resource_id not in stats["resource_id"].values:
            stats = pd.concat(
                [
                    stats,
                    pd.DataFrame(
                        [
                            {
                                "resource_id": resource_id,
                                "shown": 0,
                                "helpful": 0,
                                "skipped": 0,
                                "total_reward": 0.0,
                                "q_value": 0.0,
                                "last_served_at": now,
                                "last_feedback": "",
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
        row_index = stats.index[stats["resource_id"] == resource_id][0]
        stats.at[row_index, "shown"] = int(stats.at[row_index, "shown"]) + 1
        stats.at[row_index, "last_served_at"] = now
    save_recommender_stats(stats)


def update_feedback(resource_id: str, feedback: str) -> None:
    reward = 1.0 if feedback == "helpful" else -0.40
    stats = load_recommender_stats()
    if resource_id not in stats["resource_id"].values:
        mark_resources_served([resource_id])
        stats = load_recommender_stats()

    row_index = stats.index[stats["resource_id"] == resource_id][0]
    shown = max(1, int(stats.at[row_index, "shown"]))
    helpful = int(stats.at[row_index, "helpful"])
    skipped = int(stats.at[row_index, "skipped"])
    total_reward = float(stats.at[row_index, "total_reward"])

    if feedback == "helpful":
        helpful += 1
    else:
        skipped += 1

    total_reward += reward
    q_value = total_reward / shown

    stats.at[row_index, "helpful"] = helpful
    stats.at[row_index, "skipped"] = skipped
    stats.at[row_index, "total_reward"] = round(total_reward, 3)
    stats.at[row_index, "q_value"] = round(q_value, 3)
    stats.at[row_index, "last_feedback"] = feedback
    save_recommender_stats(stats)


def recommend_resources(final_emotion: str, recent_ids: List[str], top_n: int = 5) -> pd.DataFrame:
    catalog = build_resource_catalog()
    stats = load_recommender_stats()
    mood_targets = EMOTION_TO_CATALOG_MOODS.get(final_emotion, ["neutral", "calm", "focused"])

    candidates = catalog[catalog["mood"].isin(mood_targets)].copy()
    merged = candidates.merge(stats, on="resource_id", how="left")
    merged["shown"] = merged["shown"].fillna(0)
    merged["helpful"] = merged["helpful"].fillna(0)
    merged["skipped"] = merged["skipped"].fillna(0)
    merged["q_value"] = merged["q_value"].fillna(0.0)
    merged["repeat_penalty"] = merged["resource_id"].isin(recent_ids).astype(int)
    merged["exploration_bonus"] = 1 / np.sqrt(1 + merged["shown"])
    merged["priority_bonus"] = merged["mood"].apply(lambda mood: 0.18 if mood == mood_targets[0] else 0.06)
    merged["ranking_score"] = (
        merged["q_value"]
        + merged["exploration_bonus"]
        + merged["priority_bonus"]
        - (merged["repeat_penalty"] * 0.65)
        - (merged["skipped"] * 0.08)
    )
    merged = merged.sort_values(["ranking_score", "q_value"], ascending=False).head(top_n)
    return merged[
        ["resource_id", "mood", "title", "url", "source", "type", "offline_fallback", "ranking_score"]
    ].reset_index(drop=True)


def empty_twin_snapshot() -> Dict[str, Any]:
    return DEFAULT_TWIN_SNAPSHOT.copy()


def twin_log_has_expected_header() -> bool:
    if not TWIN_LOG_PATH.exists() or TWIN_LOG_PATH.stat().st_size == 0:
        return True
    try:
        with TWIN_LOG_PATH.open("r", encoding="utf-8", newline="") as handle:
            return next(csv.reader(handle), []) == TWIN_LOG_FIELDNAMES
    except (OSError, csv.Error):
        return False


def normalize_twin_log_file() -> None:
    if not TWIN_LOG_PATH.exists() or twin_log_has_expected_header():
        return
    existing_rows = read_twin_log().to_dict("records")
    backup_path = TWIN_LOG_PATH.with_name(
        f"{TWIN_LOG_PATH.stem}.backup-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}{TWIN_LOG_PATH.suffix}"
    )
    TWIN_LOG_PATH.replace(backup_path)
    with TWIN_LOG_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TWIN_LOG_FIELDNAMES)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in TWIN_LOG_FIELDNAMES} for row in existing_rows)


def read_twin_log() -> pd.DataFrame:
    if not TWIN_LOG_PATH.exists() or TWIN_LOG_PATH.stat().st_size == 0:
        return pd.DataFrame(columns=TWIN_LOG_FIELDNAMES)

    rows: List[Dict[str, Any]] = []
    try:
        with TWIN_LOG_PATH.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, restkey="_extra_fields")
            if not reader.fieldnames:
                return pd.DataFrame(columns=TWIN_LOG_FIELDNAMES)
            known_fields = [field for field in reader.fieldnames if field in TWIN_LOG_FIELDNAMES]
            for raw_row in reader:
                if raw_row.get("_extra_fields"):
                    continue
                normalized_row = {key: "" for key in TWIN_LOG_FIELDNAMES}
                for key in known_fields:
                    normalized_row[key] = raw_row.get(key, "")
                rows.append(normalized_row)
    except (OSError, csv.Error):
        return pd.DataFrame(columns=TWIN_LOG_FIELDNAMES)

    return pd.DataFrame(rows, columns=TWIN_LOG_FIELDNAMES)


def append_twin_log(row: Dict[str, Any]) -> None:
    ensure_runtime_dirs()
    normalize_twin_log_file()
    file_exists = TWIN_LOG_PATH.exists()
    with TWIN_LOG_PATH.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TWIN_LOG_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in TWIN_LOG_FIELDNAMES})


def build_twin_snapshot(user_hash: str) -> Dict[str, Any]:
    if not TWIN_LOG_PATH.exists():
        return empty_twin_snapshot()

    df = read_twin_log()
    if df.empty or user_hash not in df["user_hash"].values:
        return empty_twin_snapshot()

    user_df = df[df["user_hash"] == user_hash].tail(30).copy()
    numeric_columns = ["valence", "arousal", "stress"]
    for column in numeric_columns:
        user_df[column] = pd.to_numeric(user_df[column], errors="coerce").fillna(0.0)
    dominant_mode = user_df["final_emotion_after_override"].replace("", "neutral").mode()
    dominant = dominant_mode.iloc[0] if not dominant_mode.empty else "neutral"
    volatility = float(user_df["valence"].std(ddof=0)) if len(user_df) > 1 else 0.0
    return {
        "dominant_emotion": dominant,
        "avg_valence": round(float(user_df["valence"].mean()), 3),
        "avg_arousal": round(float(user_df["arousal"].mean()), 3),
        "avg_stress": round(float(user_df["stress"].mean()), 3),
        "entries": int(len(user_df)),
        "volatility": round(volatility, 3),
    }


def generate_reflective_questions(final_emotion: str, twin_snapshot: Dict[str, Any]) -> List[str]:
    templates = {
        "happy": [
            "What activity helped you reach this positive state today?",
            "How can you preserve this mood for the next study or work block?",
            "Which habit should you repeat tomorrow because it clearly improved your energy?",
        ],
        "sad": [
            "What event or thought reduced your energy the most today?",
            "Would rest, connection, or completion of one small task help you most right now?",
            "Which person, place, or activity usually helps you recover when you feel low?",
        ],
        "angry": [
            "What triggered the strongest frustration and was it controllable or uncontrollable?",
            "What would a calm response look like if you waited five minutes before reacting?",
            "Which boundary or task change could reduce this trigger next time?",
        ],
        "fear": [
            "What is the exact uncertainty behind the anxiety right now?",
            "Which smallest next action would reduce the unknown by even ten percent?",
            "What evidence do you have that the situation may still be manageable?",
        ],
        "surprise": [
            "Did the unexpected event create opportunity, stress, or both?",
            "What new information should you capture before reacting quickly?",
            "How can you turn this surprise into a useful next step?",
        ],
        "neutral": [
            "Which one task would create the highest value in the next hour?",
            "What lifestyle habit should be protected while your mind is balanced?",
            "Do you want to use this stable state for planning, learning, or recovery?",
        ],
        "disgust": [
            "What exactly feels misaligned or unacceptable in the situation?",
            "Can you step away, reframe, or replace the trigger with a healthier option?",
            "What practical boundary would protect your wellbeing here?",
        ],
    }
    questions = templates.get(final_emotion, templates["neutral"]).copy()
    if twin_snapshot["entries"] >= 3:
        questions.append(
            f"Your recent dominant pattern is `{twin_snapshot['dominant_emotion']}` with stress {twin_snapshot['avg_stress']:.2f}. What changed today compared with your usual pattern?"
        )
    return questions


def local_project_answer(question: str, fused_emotion: str, twin_snapshot: Dict[str, Any], repeated: bool) -> str:
    q = question.lower()
    intro = (
        f"Current user-state perspective: the fused emotion is **{fused_emotion}** and the recent digital twin trend "
        f"shows dominant emotion **{twin_snapshot['dominant_emotion']}** with average stress **{twin_snapshot['avg_stress']:.2f}**.\n\n"
    )
    if repeated:
        intro += "You asked a similar question earlier, so the answer below approaches it from a deeper viva-defense angle.\n\n"

    if "grad" in q or "cam" in q or "explainable" in q:
        return intro + (
            "**Grad-CAM significance**\n"
            "1. Grad-CAM highlights the image regions that most influenced the CNN prediction.\n"
            "2. In this project it makes facial emotion recognition inspectable instead of opaque.\n"
            "3. It helps the evaluator see whether the model is focusing on the face rather than background noise.\n"
            "4. In the viva, the strongest answer is that explainability builds trust, supports debugging, and enables ethical review.\n"
        )
    if "research gap" in q or "research gaps" in q or ("earlier" in q and "research" in q) or ("2023" in q and "2025" in q):
        return intro + (
            "**How this project addresses research gaps from 2023-2025**\n"
            "1. Earlier systems were often unimodal; this one fuses face, emoji, text, and voice.\n"
            "2. Many projects gave predictions without explanations; this project adds Grad-CAM and confidence-aware abstention.\n"
            "3. Prior systems focused on one-shot classification; this project adds a history-aware digital emotional twin and bandit-style recommendation updates.\n"
            "4. Many prototypes ignored ethics; here the user can override the model, logging is consent-based, and low-confidence states are flagged.\n"
        )
    if "future" in q or "scope" in q:
        return intro + (
            "**Future scope**\n"
            "- Add physiological sensors such as heart rate or GSR for stronger multimodal affect recognition.\n"
            "- Replace heuristic lifestyle adaptation with full contextual reinforcement learning over long-term outcomes.\n"
            "- Add multilingual conversation, richer avatars, and on-device optimized models for mobile deployment.\n"
            "- Extend from lifestyle assistance to education support, stress analytics, and adaptive therapy-assist dashboards.\n"
        )
    if "real world" in q or "problem" in q:
        return intro + (
            "**Real-world problem solved by the system**\n"
            "This project helps users who struggle to understand or regulate their emotional state while choosing music, study strategy, or lifestyle actions. "
            "Instead of generic advice, it personalizes support from the user's present mood, recent emotional pattern, and explicit feedback. "
            "That makes it relevant for student wellbeing, mental wellness support, focus management, and emotion-aware digital assistance.\n"
        )
    if "protagonist" in q or "different" in q:
        return intro + (
            "**Project protagonist and uniqueness**\n"
            "The protagonist is the user-centered digital emotional twin: a persistent, consent-driven emotional profile that learns from multimodal evidence and feedback. "
            "What makes the project different is not only emotion detection, but the combination of explainability, adaptive recommendation, free AI integration, and ethical override in a single-file executable system.\n"
        )
    if "global average pooling" in q or "g.a.p" in q or re.search(r"\bgap\b", q):
        return intro + (
            "**Why Global Average Pooling matters**\n"
            "Global Average Pooling compresses spatial feature maps into one value per channel before classification. "
            "Compared with large fully connected layers, it reduces parameters, overfitting, and memory use, which is important for an 8 GB Windows laptop. "
            "It also works well with Grad-CAM because the last convolutional activations remain semantically meaningful.\n"
        )
    if "mobilenet" in q or "efficientnet" in q or "cnn" in q:
        return intro + (
            "**Why EfficientNetV2 + GAP instead of MobileNetV2**\n"
            "EfficientNetV2 is a newer and stronger family for transfer learning with better efficiency-quality trade-offs. "
            "For this project, EfficientNetV2B0 keeps the model lightweight enough for CPU-friendly demos while giving a stronger feature extractor for facial emotion recognition. "
            "Using GAP after the convolutional backbone keeps the design compact and explainable.\n"
        )
    if "ethical" in q:
        return intro + (
            "**Ethical AI monitoring**\n"
            "The system treats emotion inference as supportive and probabilistic, not as absolute truth. "
            "It warns when confidence is low, allows user override, supports opt-in logging, and avoids sensitive demographic profiling in the decision path. "
            "That makes the project more responsible than many earlier demo systems.\n"
        )
    if "hugging face" in q or "token" in q or "api" in q:
        return intro + (
            "**Why Hugging Face is the free integration choice**\n"
            "Hugging Face offers free account-based access tokens that can be used with open models for chat, image inference, and speech recognition. "
            "Unlike paid providers, it is more suitable for student projects because the same code can run with free public models or with no token at all using local fallbacks. "
            "The app therefore remains executable even on limited hardware.\n"
        )
    return intro + (
        "**Project explanation**\n"
        "This major project is a single-file Streamlit system that captures user mood from multiple inputs, fuses the evidence, explains facial predictions with Grad-CAM, "
        "maintains a digital emotional twin through CSV logs, and recommends lifestyle or media resources without repeating the same suggestions. "
        "When a Hugging Face token is provided, the chatbot and voice transcription upgrade automatically to free cloud inference.\n"
    )


def ask_hf_chat(
    question: str,
    token: str,
    model_id: str,
    chat_history: List[Dict[str, str]],
    fused_emotion: str,
    twin_snapshot: Dict[str, Any],
) -> str:
    if not token or InferenceClient is None:
        raise RuntimeError("No Hugging Face token available.")

    system_prompt = (
        "You are the avatar assistant for a final-year major project titled "
        "'Cognitive Emotion Intelligence & Adaptive Lifestyle System'. "
        "Answer in a clear, polished, human-style academic tone. "
        "Focus on the project context: multimodal emotion fusion, EfficientNetV2 CNN with Global Average Pooling, Grad-CAM explainability, "
        "Digital Emotional Twin logging, ethical AI monitoring, history-aware recommendation, free Hugging Face integrations, and viva preparation. "
        "Be practical, supportive, and avoid claiming medical diagnosis. "
        f"The current fused user emotion is {fused_emotion}. "
        f"The recent digital twin dominant emotion is {twin_snapshot['dominant_emotion']} with average stress {twin_snapshot['avg_stress']:.2f}."
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history[-6:])
    messages.append({"role": "user", "content": question})

    client = make_hf_client(token=token)
    response = client.chat_completion(
        messages=messages,
        model=model_id or None,
        max_tokens=500,
        temperature=0.35,
    )
    if isinstance(response, dict):
        return response["choices"][0]["message"]["content"]
    return response.choices[0].message.content


def question_fingerprint(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.lower())).strip()


def render_avatar(reply_text: str) -> None:
    safe_text = json.dumps(reply_text or "Your assistant reply will appear here.")
    html = f"""
    <style>
      .cei-wrap {{
        display:flex;
        flex-direction:column;
        align-items:center;
        gap:14px;
        padding:18px;
        border-radius:20px;
        background:linear-gradient(135deg,#0f172a,#1e293b);
        color:#e2e8f0;
        font-family:Arial,sans-serif;
      }}
      .cei-avatar {{
        width:150px;
        height:150px;
        border-radius:50%;
        background:radial-gradient(circle at 30% 30%, #60a5fa, #1d4ed8 55%, #0f172a 90%);
        position:relative;
        box-shadow:0 0 0 10px rgba(96,165,250,0.12), 0 18px 40px rgba(15,23,42,0.45);
        animation:pulse 2.2s infinite;
      }}
      .cei-eye {{
        position:absolute;
        top:52px;
        width:18px;
        height:18px;
        background:white;
        border-radius:50%;
      }}
      .left-eye {{ left:42px; }}
      .right-eye {{ right:42px; }}
      .cei-mouth {{
        position:absolute;
        left:50%;
        bottom:34px;
        width:62px;
        height:28px;
        margin-left:-31px;
        border-bottom:5px solid white;
        border-radius:0 0 60px 60px;
      }}
      .cei-note {{
        text-align:center;
        font-size:14px;
        line-height:1.5;
        max-width:480px;
      }}
      button {{
        background:#2563eb;
        color:white;
        border:none;
        padding:10px 16px;
        border-radius:10px;
        font-weight:600;
        cursor:pointer;
      }}
      @keyframes pulse {{
        0% {{ transform:scale(1); }}
        50% {{ transform:scale(1.03); }}
        100% {{ transform:scale(1); }}
      }}
    </style>
    <div class="cei-wrap">
      <div class="cei-avatar">
        <div class="cei-eye left-eye"></div>
        <div class="cei-eye right-eye"></div>
        <div class="cei-mouth"></div>
      </div>
      <div class="cei-note">Animated avatar assistant with optional browser voice output.</div>
      <button onclick="speakNow()">Speak latest answer</button>
      <div class="cei-note" id="cei-preview"></div>
    </div>
    <script>
      const latestText = {safe_text};
      document.getElementById("cei-preview").innerText = latestText.slice(0, 260);
      function speakNow() {{
        const utterance = new SpeechSynthesisUtterance(latestText);
        utterance.rate = 1.0;
        utterance.pitch = 1.0;
        utterance.volume = 1.0;
        speechSynthesis.cancel();
        speechSynthesis.speak(utterance);
      }}
    </script>
    """
    components.html(html, height=320)


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def bytes_to_pil_image(raw_bytes: bytes) -> Optional[Image.Image]:
    if not raw_bytes:
        return None
    return Image.open(io.BytesIO(raw_bytes)).convert("RGB")


def coerce_pil_image(value: Any) -> Optional[Image.Image]:
    if value is None:
        return None
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        if "bytes" in value and value["bytes"] is not None:
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if "path" in value and value["path"]:
            return Image.open(value["path"]).convert("RGB")
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(value)).convert("RGB")
    return None


def extract_label_from_example(example: Dict[str, Any], label_column: str, label_feature: Any = None) -> str:
    value = example.get(label_column)
    if ClassLabel is not None and isinstance(label_feature, ClassLabel) and isinstance(value, int):
        return str(label_feature.names[value])
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, dict):
        numeric_items = [(key, val) for key, val in value.items() if isinstance(val, (int, float))]
        if numeric_items:
            return max(numeric_items, key=lambda item: float(item[1]))[0]
        boolean_items = [key for key, val in value.items() if bool(val)]
        if boolean_items:
            return boolean_items[0]
    if isinstance(value, list) and value and isinstance(value[0], dict):
        first = value[0]
        numeric_items = [(key, val) for key, val in first.items() if isinstance(val, (int, float))]
        if numeric_items:
            return max(numeric_items, key=lambda item: float(item[1]))[0]
    return "neutral"


def discover_columns(dataset_obj: Any) -> Tuple[str, str, Any]:
    features = getattr(dataset_obj, "features", {}) or {}
    columns = list(getattr(dataset_obj, "column_names", []) or list(features.keys()))

    image_column = None
    for candidate in ["image", "img", "face", "frame", "pixels"]:
        if candidate in columns:
            image_column = candidate
            break
    if image_column is None and columns:
        image_column = columns[0]

    label_column = None
    for candidate in ["label", "emotion", "labels", "class", "annotations", "category"]:
        if candidate in columns:
            label_column = candidate
            break
    if label_column is None:
        label_column = columns[-1]

    return image_column, label_column, features.get(label_column)


def clear_existing_dataset() -> None:
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR, ignore_errors=True)
    DATASET_DIR.mkdir(exist_ok=True)


def prepare_hf_dataset(source_name: str, max_images: int = 900, seed: int = 42) -> Dict[str, Any]:
    ensure_runtime_dirs()
    if load_dataset is None:
        raise RuntimeError("The `datasets` package is required for automatic dataset preparation.")
    if source_name not in DATASET_SOURCES:
        raise ValueError(f"Unknown dataset source: {source_name}")

    clear_existing_dataset()
    source = DATASET_SOURCES[source_name]
    repo_id = source["repo_id"]
    preferred_splits = source["preferred_splits"]
    streaming = source["streaming"]

    dataset_collection = load_dataset(repo_id, streaming=streaming)
    split_name = next((name for name in preferred_splits if name in dataset_collection), list(dataset_collection.keys())[0])
    dataset_obj = dataset_collection[split_name]
    image_column, label_column, label_feature = discover_columns(dataset_obj)

    rows: List[Tuple[Image.Image, str]] = []
    limit = max(200, int(max_images))

    for example in dataset_obj:
        image = coerce_pil_image(example.get(image_column))
        if image is None:
            continue
        label = map_label_to_unified(extract_label_from_example(example, label_column, label_feature))
        rows.append((image, label))
        if len(rows) >= limit:
            break

    if len(rows) < 30:
        raise RuntimeError("The selected dataset source did not provide enough compatible image-label pairs.")

    random.Random(seed).shuffle(rows)
    labels = [label for _, label in rows]
    indices = list(range(len(rows)))

    train_indices, temp_indices = train_test_split(
        indices,
        test_size=0.30,
        random_state=seed,
        stratify=labels if len(set(labels)) > 1 else None,
    )
    temp_labels = [labels[index] for index in temp_indices]
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=0.50,
        random_state=seed,
        stratify=temp_labels if len(set(temp_labels)) > 1 else None,
    )

    split_map = {"train": train_indices, "val": val_indices, "test": test_indices}
    counts: Dict[str, Dict[str, int]] = {}

    for split_name, split_indices in split_map.items():
        counts[split_name] = {}
        for running_index, sample_index in enumerate(split_indices, start=1):
            image, label = rows[sample_index]
            split_dir = DATASET_DIR / split_name / label
            split_dir.mkdir(parents=True, exist_ok=True)
            target_path = split_dir / f"{label}_{running_index:05d}.png"
            ImageOps.fit(image.convert("RGB"), (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)).save(target_path)
            counts[split_name][label] = counts[split_name].get(label, 0) + 1

    manifest = {
        "created_at": utc_now(),
        "source_name": source_name,
        "repo_id": repo_id,
        "source_split": split_name,
        "max_images_requested": max_images,
        "actual_images_saved": len(rows),
        "image_column": image_column,
        "label_column": label_column,
        "counts": counts,
    }
    DATASET_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def get_last_conv_layer_name(model: Any) -> str:
    if tf is None:
        raise RuntimeError("TensorFlow is not available.")
    last_conv = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer.name
        elif isinstance(layer, tf.keras.Model):
            try:
                nested = get_last_conv_layer_name(layer)
            except Exception:
                nested = None
            if nested:
                last_conv = nested
    if not last_conv:
        raise RuntimeError("No convolutional layer was found for Grad-CAM.")
    return last_conv


def build_efficientnet_model(num_classes: int, image_size: int = DEFAULT_IMAGE_SIZE, fine_tune: bool = False):
    if tf is None:
        raise RuntimeError("TensorFlow is not installed.")

    inputs = keras.Input(shape=(image_size, image_size, 3), name="image")
    x = keras.layers.Rescaling(1.0 / 255.0, name="rescaling")(inputs)
    backbone = keras.applications.EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
        input_shape=(image_size, image_size, 3),
        pooling=None,
        name="efficientnetv2b0_backbone",
    )
    backbone.trainable = fine_tune
    x = backbone(x, training=False)
    x = keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = keras.layers.Dropout(0.25, name="dropout")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="emotion_head")(x)
    model = keras.Model(inputs, outputs, name="cei_emotion_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")],
    )
    return model, backbone, get_last_conv_layer_name(backbone)


def train_emotion_model(
    batch_size: int = 8,
    epochs: int = 2,
    image_size: int = DEFAULT_IMAGE_SIZE,
    fine_tune: bool = False,
) -> Dict[str, Any]:
    ensure_runtime_dirs()
    if tf is None:
        raise RuntimeError("TensorFlow is not installed, so local CNN training is unavailable.")

    train_dir = DATASET_DIR / "train"
    val_dir = DATASET_DIR / "val"
    test_dir = DATASET_DIR / "test"
    if not train_dir.exists() or not any(train_dir.iterdir()):
        raise RuntimeError("No prepared dataset found under dataset/train.")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=True,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=False,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=False,
    )

    class_names = list(train_ds.class_names)
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)

    model, backbone, gradcam_layer = build_efficientnet_model(len(class_names), image_size, fine_tune)
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=1)
    model.save(MODEL_PATH)

    probabilities: List[np.ndarray] = []
    y_true: List[int] = []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        probabilities.append(preds)
        y_true.extend(labels.numpy().tolist())

    y_prob = np.concatenate(probabilities, axis=0)
    y_pred = y_prob.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True)

    roc_auc = None
    try:
        y_true_one_hot = tf.keras.utils.to_categorical(y_true, num_classes=len(class_names))
        roc_auc = float(
            roc_auc_score(
                y_true_one_hot,
                y_prob,
                average="weighted",
                multi_class="ovr",
            )
        )
    except Exception:
        roc_auc = None

    metadata = {
        "trained_at": utc_now(),
        "image_size": image_size,
        "batch_size": batch_size,
        "epochs": epochs,
        "fine_tune": fine_tune,
        "class_names": class_names,
        "backbone_name": backbone.name,
        "gradcam_layer": gradcam_layer,
        "metrics": {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "roc_auc": round(float(roc_auc), 4) if roc_auc is not None else None,
        },
        "history": {key: [float(value) for value in values] for key, values in history.history.items()},
        "classification_report": report,
    }
    MODEL_META_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "class_names": class_names,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "metadata": metadata,
    }


def generate_gradcam_overlay(image: Image.Image) -> Tuple[Optional[Image.Image], str]:
    if tf is None:
        return None, "TensorFlow is not installed, so Grad-CAM is unavailable."
    model, metadata = load_local_emotion_model()
    if model is None:
        return None, "No local trained model was found. Train the CNN first to enable Grad-CAM."

    image_size = int(metadata.get("image_size", DEFAULT_IMAGE_SIZE))
    processed = ImageOps.fit(image.convert("RGB"), (image_size, image_size))
    array = np.expand_dims(np.array(processed).astype("float32"), axis=0)

    try:
        backbone = model.get_layer(metadata["backbone_name"])
        conv_layer = backbone.get_layer(metadata["gradcam_layer"])
        grad_model = tf.keras.models.Model(inputs=model.inputs, outputs=[conv_layer.output, model.output])
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(array)
            top_index = tf.argmax(predictions[0])
            top_class = predictions[:, top_index]
        gradients = tape.gradient(top_class, conv_outputs)
        pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(conv_outputs * pooled_gradients, axis=-1)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-9)
        heatmap = heatmap.numpy()
    except Exception as exc:
        return None, f"Grad-CAM generation failed: {exc}"

    cmap = plt.get_cmap("jet")
    colored = cmap(heatmap)[..., :3]
    overlay = Image.fromarray(np.uint8(np.clip(colored, 0, 1) * 255)).resize(processed.size)
    base = np.array(processed).astype("float32") / 255.0
    overlay_arr = np.array(overlay).astype("float32") / 255.0
    combined = (base * 0.55) + (overlay_arr * 0.45)
    combined = np.uint8(np.clip(combined, 0, 1) * 255)
    return Image.fromarray(combined), "Grad-CAM generated from the local EfficientNetV2 model."


def plot_scores(scores: Dict[str, float], title: str):
    fig, axis = plt.subplots(figsize=(8, 3.5))
    series = pd.Series(normalize_scores(scores)).sort_values(ascending=False)
    sns.barplot(x=series.index, y=series.values, ax=axis, palette="crest")
    axis.set_title(title)
    axis.set_ylabel("Probability")
    axis.set_xlabel("Emotion")
    axis.set_ylim(0, 1)
    axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm: np.ndarray, labels: List[str]):
    fig, axis = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=axis)
    axis.set_title("Confusion Matrix")
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    fig.tight_layout()
    return fig


def is_streamlit_context() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def render_header() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption(APP_TAGLINE)
    st.info(
        "Design target: Windows 11 laptop, Intel i5, 8 GB RAM. The app stays usable with or without TensorFlow, "
        "and it upgrades automatically when a free Hugging Face token is provided."
    )


def render_sidebar() -> Dict[str, Any]:
    st.sidebar.header("Profile and runtime")
    default_hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or ""
    username = st.sidebar.text_input("Username / demo login", value=st.session_state.get("username", "StudentUser"))
    private_key = st.sidebar.text_input(
        "Private session key", value=st.session_state.get("private_key", "major-project-demo"), type="password"
    )
    hf_token = st.sidebar.text_input(
        "Hugging Face access token (optional)",
        value=st.session_state.get("hf_token", default_hf_token),
        type="password",
        help="Create a free token in your Hugging Face account settings.",
    )
    llm_model = st.sidebar.text_input("HF chat model", value="Qwen/Qwen2.5-7B-Instruct")
    asr_model = st.sidebar.text_input("HF ASR model", value="openai/whisper-large-v3-turbo")
    face_model = st.sidebar.text_input("HF face model", value="dima806/facial_emotions_image_detection")
    consent_to_log = st.sidebar.checkbox("Allow Digital Emotional Twin CSV logging", value=True)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Runtime status")
    st.sidebar.write(f"TensorFlow available: {'Yes' if tf is not None else 'No'}")
    st.sidebar.write(f"Hugging Face client available: {'Yes' if InferenceClient is not None else 'No'}")
    st.sidebar.write(f"Local CNN present: {'Yes' if MODEL_PATH.exists() else 'No'}")
    st.sidebar.write(f"Catalog size: {len(build_resource_catalog())} resources")

    state = {
        "username": username.strip() or "StudentUser",
        "private_key": private_key or "major-project-demo",
        "hf_token": hf_token.strip(),
        "llm_model": llm_model.strip(),
        "asr_model": asr_model.strip(),
        "face_model": face_model.strip(),
        "consent_to_log": consent_to_log,
    }
    st.session_state.update(state)
    return state


def run_streamlit_app() -> None:
    ensure_runtime_dirs()
    render_header()
    runtime = render_sidebar()
    user_hash = make_user_hash(runtime["username"], runtime["private_key"])
    twin_snapshot = build_twin_snapshot(user_hash)

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Twin entries", twin_snapshot["entries"])
    metric_2.metric("Dominant emotion", twin_snapshot["dominant_emotion"])
    metric_3.metric("Avg stress", twin_snapshot["avg_stress"])
    metric_4.metric("Valence volatility", twin_snapshot["volatility"])

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "question_memory" not in st.session_state:
        st.session_state.question_memory = set()
    if "recent_recommendation_ids" not in st.session_state:
        st.session_state.recent_recommendation_ids = []

    tabs = st.tabs(
        [
            "Emotion Fusion Studio",
            "Training and Grad-CAM",
            "Adaptive Recommender",
            "Avatar Chatbot",
            "Research and Viva Support",
        ]
    )

    with tabs[0]:
        st.subheader("Multimodal emotion fusion")
        left, right = st.columns([1.1, 1.0])

        with left:
            uploaded_image = st.file_uploader("Upload a face image", type=["png", "jpg", "jpeg"])
            camera_image = st.camera_input("Or capture a live face image")
            emoji_choice = st.selectbox("Emoji mood input", list(EMOJI_PRIORS.keys()), index=0)
            text_context = st.text_area(
                "Free-text context",
                placeholder="Describe how you feel, what happened, and what kind of support or recommendation you want.",
                height=180,
            )

            audio_bytes = b""
            audio_transcript = ""
            audio_note = "Voice input not used."
            if hasattr(st, "audio_input"):
                audio_capture = st.audio_input("Voice-to-text input (optional, requires HF token)")
                if audio_capture is not None:
                    audio_bytes = audio_capture.read()
                    audio_transcript, audio_note = transcribe_audio_with_hf(
                        audio_bytes, runtime["hf_token"], runtime["asr_model"]
                    )
                    if audio_transcript:
                        st.success(f"Voice transcript: {audio_transcript}")
                    else:
                        st.caption(audio_note)
            else:
                st.caption("Your Streamlit version does not expose st.audio_input, so voice upload is unavailable in this environment.")

            analyze_clicked = st.button("Fuse emotion state", type="primary")

        with right:
            image_for_analysis = None
            if camera_image is not None:
                image_for_analysis = Image.open(camera_image).convert("RGB")
            elif uploaded_image is not None:
                image_for_analysis = Image.open(uploaded_image).convert("RGB")

            if image_for_analysis is not None:
                st.image(image_for_analysis, caption="Input face image", use_container_width=True)
            else:
                st.info("Add a face image or use text, emoji, and voice alone.")

        if analyze_clicked:
            face_signal = analyze_face_image(image_for_analysis, runtime["hf_token"], runtime["face_model"])
            text_signal = analyze_text_emotion(text_context, source="text")
            voice_signal = analyze_text_emotion(audio_transcript, source="voice")
            emoji_signal = analyze_emoji_emotion(emoji_choice)
            signals = [face_signal, text_signal, voice_signal, emoji_signal]
            fusion = fuse_modalities(signals)
            warnings = ethical_monitor(fusion, signals)

            st.session_state.current_fusion = fusion
            st.session_state.current_signals = signals
            st.session_state.latest_audio_transcript = audio_transcript
            if image_for_analysis is not None:
                st.session_state.latest_face_image_bytes = image_to_png_bytes(image_for_analysis)

            st.success(
                f"Fused emotion: **{fusion['final_emotion']}** | confidence **{fusion['confidence']:.2f}** | "
                f"valence **{fusion['valence']:.2f}** | arousal **{fusion['arousal']:.2f}**"
            )

            override_choice = st.selectbox(
                "User override / ethical correction",
                ["No override"] + UNIFIED_EMOTIONS,
                help="If the AI is wrong, the user should be able to correct it.",
            )
            final_after_override = (
                fusion["final_emotion"] if override_choice == "No override" else override_choice
            )
            st.info(f"Recommendation and chat modules will use: **{final_after_override}**")
            st.session_state.final_emotion_after_override = final_after_override

            signal_columns = st.columns(2)
            rendered = 0
            for signal in signals:
                if signal is None:
                    continue
                with signal_columns[rendered % 2]:
                    st.markdown(f"**{signal.source.title()} signal**")
                    st.write(f"Top emotion: `{signal.top_emotion}`")
                    st.write(f"Confidence: `{signal.confidence:.2f}`")
                    st.caption(signal.note)
                    if signal.evidence:
                        st.caption(f"Evidence: {signal.evidence}")
                    st.pyplot(plot_scores(signal.scores, f"{signal.source.title()} emotion distribution"))
                rendered += 1

            st.markdown("**Fused emotion distribution**")
            st.pyplot(plot_scores(fusion["scores"], "Fused multimodal probabilities"))

            if warnings:
                st.warning("Ethical AI monitoring flags:\n\n- " + "\n- ".join(warnings))
            else:
                st.success("Ethical AI monitoring: no major warning flags for the current fusion cycle.")

            reflective_questions = generate_reflective_questions(final_after_override, twin_snapshot)
            st.markdown("**Suggested reflective questions based on current mindset**")
            for question in reflective_questions:
                st.markdown(f"- {question}")

            if runtime["consent_to_log"]:
                append_twin_log(
                    {
                        "timestamp": utc_now(),
                        "username": runtime["username"],
                        "user_hash": user_hash,
                        "face_emotion": face_signal.top_emotion if face_signal else "",
                        "face_confidence": face_signal.confidence if face_signal else "",
                        "text_emotion": text_signal.top_emotion if text_signal else "",
                        "voice_emotion": voice_signal.top_emotion if voice_signal else "",
                        "emoji_emotion": emoji_signal.top_emotion if emoji_signal else "",
                        "fused_emotion": fusion["final_emotion"],
                        "final_emotion_after_override": final_after_override,
                        "confidence": fusion["confidence"],
                        "valence": fusion["valence"],
                        "arousal": fusion["arousal"],
                        "stress": fusion["stress"],
                        "used_sources": ", ".join(fusion["used_sources"]),
                        "override_emotion": "" if override_choice == "No override" else override_choice,
                        "context_excerpt": text_context[:160],
                        "voice_excerpt": audio_transcript[:160],
                        "consent_to_log": runtime["consent_to_log"],
                    }
                )
                st.caption("Digital Emotional Twin log updated in `cei_twin_log.csv`.")
            else:
                st.caption("Logging is disabled for this run.")

    with tabs[1]:
        st.subheader("Dataset preparation, local CNN training, and Grad-CAM")
        prep_col, train_col = st.columns(2)

        with prep_col:
            st.markdown("**Automatic dataset preparation**")
            source_name = st.selectbox("Dataset source", list(DATASET_SOURCES.keys()), index=0)
            sample_count = st.slider("Sample size for low-RAM training", min_value=600, max_value=1200, value=900, step=100)
            if st.button("Prepare sampled dataset"):
                try:
                    manifest = prepare_hf_dataset(source_name=source_name, max_images=sample_count)
                    st.success(f"Dataset prepared successfully from `{manifest['repo_id']}`.")
                    st.json(manifest)
                except Exception as exc:
                    st.error(f"Dataset preparation failed: {exc}")

            if DATASET_MANIFEST_PATH.exists():
                st.markdown("**Current manifest**")
                st.code(DATASET_MANIFEST_PATH.read_text(encoding="utf-8"), language="json")

        with train_col:
            st.markdown("**EfficientNetV2 + Global Average Pooling training**")
            batch_size = st.selectbox("Batch size", [4, 8], index=1)
            epochs = st.selectbox("Epochs", [1, 2, 3], index=1)
            fine_tune = st.checkbox("Fine-tune backbone", value=False)
            if st.button("Train local emotion model"):
                try:
                    results = train_emotion_model(batch_size=batch_size, epochs=epochs, fine_tune=fine_tune)
                    st.success("Training complete. Model and metadata saved under `models/`.")
                    metrics = st.columns(4)
                    metrics[0].metric("Precision", f"{results['precision']:.3f}")
                    metrics[1].metric("Recall", f"{results['recall']:.3f}")
                    metrics[2].metric("F1", f"{results['f1']:.3f}")
                    metrics[3].metric("ROC-AUC", f"{results['roc_auc']:.3f}" if results["roc_auc"] is not None else "N/A")
                    st.pyplot(plot_confusion_matrix(results["confusion_matrix"], results["class_names"]))
                except Exception as exc:
                    st.error(f"Training failed: {exc}")

        st.markdown("---")
        st.markdown("**Grad-CAM explainability**")
        gradcam_input = None
        if "current_fusion" in st.session_state:
            if camera_image is not None:
                gradcam_input = Image.open(camera_image).convert("RGB")
            elif uploaded_image is not None:
                gradcam_input = Image.open(uploaded_image).convert("RGB")
            elif st.session_state.get("latest_face_image_bytes"):
                gradcam_input = bytes_to_pil_image(st.session_state.latest_face_image_bytes)
        if gradcam_input is None:
            gradcam_upload = st.file_uploader("Upload an image for Grad-CAM", type=["png", "jpg", "jpeg"], key="gradcam_uploader")
            if gradcam_upload is not None:
                gradcam_input = Image.open(gradcam_upload).convert("RGB")

        if st.button("Generate Grad-CAM overlay"):
            if gradcam_input is None:
                st.warning("Please provide an image first.")
            else:
                overlay, message = generate_gradcam_overlay(gradcam_input)
                if overlay is None:
                    st.warning(message)
                else:
                    gc_left, gc_right = st.columns(2)
                    with gc_left:
                        st.image(gradcam_input, caption="Original image", use_container_width=True)
                    with gc_right:
                        st.image(overlay, caption="Grad-CAM overlay", use_container_width=True)
                    st.caption(message)

    with tabs[2]:
        st.subheader("History-aware adaptive recommender")
        active_emotion = st.session_state.get("final_emotion_after_override")
        if not active_emotion:
            st.info("Run the Emotion Fusion Studio first to generate a personalized recommendation state.")
        else:
            recommendations = recommend_resources(active_emotion, st.session_state.recent_recommendation_ids, top_n=5)
            current_ids = recommendations["resource_id"].tolist()
            if current_ids != st.session_state.recent_recommendation_ids:
                mark_resources_served(current_ids)
                st.session_state.recent_recommendation_ids = current_ids

            st.caption(
                "The recommender uses a simple bandit-style score with helpful/skip feedback and repeat-avoidance penalties."
            )
            for index, row in recommendations.iterrows():
                with st.container(border=True):
                    st.markdown(f"**{index + 1}. {row['title']}**")
                    st.write(f"Mood bucket: `{row['mood']}` | Source: `{row['source']}` | Type: `{row['type']}`")
                    st.markdown(f"[Open resource]({row['url']})")
                    st.caption(f"Offline fallback: {row['offline_fallback']}")
                    fb_cols = st.columns([1, 1, 4])
                    if fb_cols[0].button("Helpful", key=f"helpful-{row['resource_id']}"):
                        update_feedback(row["resource_id"], "helpful")
                        st.success("Feedback saved.")
                    if fb_cols[1].button("Skip", key=f"skip-{row['resource_id']}"):
                        update_feedback(row["resource_id"], "skip")
                        st.warning("Skip feedback saved.")

            export_col1, export_col2 = st.columns(2)
            if export_col1.button("Export resource catalog CSV"):
                path = export_resource_catalog()
                st.success(f"Catalog exported to `{path.name}` with {len(build_resource_catalog())} entries.")
            if export_col2.button("Refresh recommendations"):
                st.session_state.recent_recommendation_ids = []
                st.rerun()

            stats_df = load_recommender_stats().sort_values("q_value", ascending=False).head(10)
            if not stats_df.empty:
                st.markdown("**Recommender stats snapshot**")
                st.dataframe(stats_df, use_container_width=True)

    with tabs[3]:
        st.subheader("Animated avatar chatbot")
        latest_answer = st.session_state.get(
            "latest_answer",
            "Ask about the major project, future scope, Grad-CAM, EfficientNetV2, ethical AI, research gaps, or viva preparation.",
        )
        render_avatar(latest_answer)

        suggested = generate_reflective_questions(
            st.session_state.get("final_emotion_after_override", "neutral"),
            twin_snapshot,
        )
        st.markdown("**Suggested questions to ask the bot right now**")
        for question in suggested[:3]:
            st.markdown(f"- {question}")

        voice_question = ""
        voice_question_note = ""
        if hasattr(st, "audio_input"):
            voice_for_bot = st.audio_input("Record a question for the bot (optional)", key="bot_audio")
            if voice_for_bot is not None:
                voice_question, voice_question_note = transcribe_audio_with_hf(
                    voice_for_bot.read(), runtime["hf_token"], runtime["asr_model"]
                )
                if voice_question:
                    st.caption(f"Voice question transcript: {voice_question}")
                elif voice_question_note:
                    st.caption(voice_question_note)

        question = st.chat_input("Ask your project or viva question here")
        if not question and voice_question:
            question = voice_question

        for message in st.session_state.chat_history[-8:]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if question:
            with st.chat_message("user"):
                st.markdown(question)

            fingerprint = question_fingerprint(question)
            repeated = fingerprint in st.session_state.question_memory
            st.session_state.question_memory.add(fingerprint)

            try:
                if runtime["hf_token"]:
                    answer = ask_hf_chat(
                        question=question,
                        token=runtime["hf_token"],
                        model_id=runtime["llm_model"],
                        chat_history=st.session_state.chat_history,
                        fused_emotion=st.session_state.get("final_emotion_after_override", "neutral"),
                        twin_snapshot=twin_snapshot,
                    )
                else:
                    answer = local_project_answer(
                        question,
                        st.session_state.get("final_emotion_after_override", "neutral"),
                        twin_snapshot,
                        repeated,
                    )
            except Exception:
                answer = local_project_answer(
                    question,
                    st.session_state.get("final_emotion_after_override", "neutral"),
                    twin_snapshot,
                    repeated,
                )

            with st.chat_message("assistant"):
                st.markdown(answer)

            st.session_state.chat_history.append({"role": "user", "content": question})
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.session_state.latest_answer = answer

    with tabs[4]:
        st.subheader("Research, viva, and deployment support")
        st.markdown("**Top viva questions likely to be asked**")
        for question in TOP_VIVA_QUESTIONS:
            st.markdown(f"- {question}")

        st.markdown("---")
        st.markdown("**Why this project is different**")
        st.write(
            "The project's central protagonist is the user-governed Digital Emotional Twin: a running, consent-based emotional profile "
            "that converts multimodal perception into explainable, adaptive lifestyle support. "
            "Most student projects stop at emotion detection; this one extends the pipeline into ethical monitoring, adaptive recommendation, and a conversational support layer."
        )

        st.markdown("**Recommended free deployment path**")
        st.write(
            "For a zero-cost installable experience, host the Streamlit app, open it on Android in Chrome, and use **Add to Home Screen**. "
            "For an APK-style wrapper, deploy the app publicly and wrap the URL in a WebView shell using Android Studio or another free PWA/WebView approach."
        )

        doc_links = {
            "Windows + VS Code setup guide": DOCS_DIR / "VS_CODE_WINDOWS_SETUP.md",
            "IEEE-style research paper draft": DOCS_DIR / "RESEARCH_PAPER_IEEE.md",
            "Literature extraction matrix": DOCS_DIR / "LITERATURE_MATRIX.md",
            "Free Hugging Face token setup": DOCS_DIR / "HUGGING_FACE_FREE_API_SETUP.md",
            "Viva questions and answers": DOCS_DIR / "VIVA_GUIDE.md",
            "APK conversion guide": DOCS_DIR / "APK_CONVERSION_GUIDE.md",
        }
        st.markdown("**Repository documents**")
        for label, path in doc_links.items():
            status = "available" if path.exists() else "not created yet"
            st.write(f"- {label}: `{path.name}` ({status})")


def cli_export_catalog() -> int:
    path = export_resource_catalog()
    print(f"Catalog exported to {path} with {len(build_resource_catalog())} entries.")
    return 0


def cli_prepare_dataset(source_name: str, sample_size: int) -> int:
    manifest = prepare_hf_dataset(source_name=source_name, max_images=sample_size)
    print(json.dumps(manifest, indent=2))
    return 0


def main_cli() -> int:
    parser = argparse.ArgumentParser(description="Single-file major project utilities")
    parser.add_argument("--export-catalog", action="store_true", help="Export the 100+ item resource catalog")
    parser.add_argument("--prepare-dataset", action="store_true", help="Prepare a sampled dataset under dataset/")
    parser.add_argument("--dataset-source", default="FER2013", choices=sorted(DATASET_SOURCES.keys()))
    parser.add_argument("--sample-size", type=int, default=900)
    args, _ = parser.parse_known_args()

    if args.export_catalog:
        return cli_export_catalog()
    if args.prepare_dataset:
        return cli_prepare_dataset(args.dataset_source, args.sample_size)

    print("Run `streamlit run app.py` for the UI, `python app.py --export-catalog`, or `python app.py --prepare-dataset`.")
    return 0


if is_streamlit_context():
    run_streamlit_app()
elif __name__ == "__main__":
    raise SystemExit(main_cli())
