from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import math
import os
import random
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote_plus

import numpy as np
import requests

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional runtime dependency
    pd = None

try:
    from PIL import Image, ImageEnhance, ImageOps
except Exception:  # pragma: no cover - optional runtime dependency
    Image = None
    ImageEnhance = None
    ImageOps = None

try:
    import matplotlib.pyplot as plt
    from matplotlib import cm
except Exception:  # pragma: no cover - optional runtime dependency
    plt = None
    cm = None

try:
    import streamlit as st
except Exception:  # pragma: no cover - optional runtime dependency
    st = None

try:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
        roc_auc_score,
    )
    from sklearn.preprocessing import label_binarize
except Exception:  # pragma: no cover - optional runtime dependency
    accuracy_score = None
    confusion_matrix = None
    precision_recall_fscore_support = None
    roc_auc_score = None
    label_binarize = None

try:
    import tensorflow as tf
except Exception:  # pragma: no cover - optional runtime dependency
    tf = None

try:
    from datasets import ClassLabel, DatasetDict, load_dataset
except Exception:  # pragma: no cover - optional runtime dependency
    ClassLabel = None
    DatasetDict = None
    load_dataset = None

try:
    from huggingface_hub import InferenceClient
except Exception:  # pragma: no cover - optional runtime dependency
    InferenceClient = None


APP_TITLE = "Cognitive Emotion Intelligence & Adaptive Lifestyle System"
APP_TAGLINE = (
    "Single-file Streamlit major-project prototype with multimodal emotion fusion, "
    "adaptive recommendations, digital emotional twin logging, ethical AI monitoring, "
    "EfficientNetV2 + GAP training, and Grad-CAM explainability."
)

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "efficientnetv2_cei.keras"
MODEL_METADATA_PATH = MODELS_DIR / "efficientnetv2_cei_metadata.json"
DATASET_MANIFEST_PATH = DATASET_DIR / "dataset_manifest.json"
TWIN_LOG_PATH = ROOT / "cei_twin_log.csv"
RECOMMENDER_STATS_PATH = ROOT / "recommender_stats.csv"
RESOURCE_CATALOG_PATH = ROOT / "resource_catalog.csv"
DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_EXPORT_COUNT = 120

EMOTIONS = ["happy", "sad", "angry", "neutral", "surprise", "fear", "calm"]
PRIMARY_EMOTIONS = ["happy", "sad", "angry", "neutral"]
EMOJI_TO_EMOTION = {
    "😀": "happy",
    "🙂": "happy",
    "😌": "calm",
    "😴": "calm",
    "😐": "neutral",
    "😕": "sad",
    "😢": "sad",
    "😡": "angry",
    "😠": "angry",
    "😨": "fear",
    "😲": "surprise",
}

TEXT_LEXICON = {
    "happy": {
        "words": {
            "happy",
            "joy",
            "excited",
            "hopeful",
            "confident",
            "good",
            "great",
            "fresh",
            "motivated",
            "enthusiastic",
            "calmly",
            "positive",
        },
        "base": 0.12,
    },
    "sad": {
        "words": {
            "sad",
            "low",
            "tired",
            "lonely",
            "down",
            "cry",
            "empty",
            "upset",
            "hurt",
            "unhappy",
            "depressed",
            "exhausted",
        },
        "base": 0.10,
    },
    "angry": {
        "words": {
            "angry",
            "frustrated",
            "annoyed",
            "mad",
            "rage",
            "irritated",
            "stress",
            "stressed",
            "pressure",
            "overloaded",
            "burnout",
            "furious",
        },
        "base": 0.09,
    },
    "neutral": {
        "words": {
            "okay",
            "fine",
            "normal",
            "routine",
            "balanced",
            "steady",
            "usual",
            "neutral",
            "average",
            "stable",
        },
        "base": 0.11,
    },
    "surprise": {
        "words": {"surprised", "shocked", "unexpected", "wow", "amazed"},
        "base": 0.07,
    },
    "fear": {
        "words": {"fear", "afraid", "anxious", "worried", "panic", "nervous"},
        "base": 0.08,
    },
    "calm": {
        "words": {"peaceful", "relaxed", "mindful", "grounded", "rested", "calm"},
        "base": 0.10,
    },
}

HF_DATASET_OPTIONS = {
    "FER2013 mirror": "Jeneral/fer-2013",
    "FER2013 alternate mirror": "AutumnQiu/fer2013",
    "Facial emotion image set": "UniqueData/facial-emotion-recognition-dataset",
}


@dataclass
class EmotionEstimate:
    source: str
    scores: Dict[str, float]
    top_emotion: str
    confidence: float
    notes: str


def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def ensure_directories() -> None:
    for folder in [DATASET_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR, MODELS_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    filtered = {emotion: max(float(value), 0.0) for emotion, value in scores.items()}
    total = sum(filtered.values())
    if total <= 0:
        uniform = 1.0 / len(EMOTIONS)
        return {emotion: uniform for emotion in EMOTIONS}
    return {emotion: filtered.get(emotion, 0.0) / total for emotion in EMOTIONS}


def ranked_emotions(scores: Dict[str, float]) -> List[Tuple[str, float]]:
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def to_counter_text(scores: Dict[str, float], limit: int = 3) -> str:
    parts = [f"{name}:{value:.2f}" for name, value in ranked_emotions(scores)[:limit]]
    return ", ".join(parts)


def csv_append(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def read_csv_preview(path: Path, limit: int = 25):
    if not path.exists():
        return [] if pd is None else pd.DataFrame()
    if pd is None:
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = []
            for index, row in enumerate(reader):
                rows.append(row)
                if index + 1 >= limit:
                    break
            return rows
    return pd.read_csv(path).tail(limit)


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def make_base_scores() -> Dict[str, float]:
    return {emotion: 0.01 for emotion in EMOTIONS}


def estimate_from_emoji(emoji: str) -> EmotionEstimate:
    scores = make_base_scores()
    detected = EMOJI_TO_EMOTION.get(emoji, "neutral")
    scores[detected] += 0.9
    normalized = normalize_scores(scores)
    top_emotion, confidence = ranked_emotions(normalized)[0]
    return EmotionEstimate(
        source="emoji",
        scores=normalized,
        top_emotion=top_emotion,
        confidence=confidence,
        notes=f"Emoji '{emoji}' mapped to {detected}.",
    )


def estimate_from_text(text: str, source_name: str = "text") -> EmotionEstimate:
    tokens = {
        token.strip(".,!?;:-_()[]{}<>\"'").lower()
        for token in text.split()
        if token.strip()
    }
    scores = {emotion: config["base"] for emotion, config in TEXT_LEXICON.items()}
    for emotion, config in TEXT_LEXICON.items():
        overlap = len(tokens.intersection(config["words"]))
        scores[emotion] += overlap * 0.18
    if not tokens:
        scores["neutral"] += 0.25
    normalized = normalize_scores(scores)
    top_emotion, confidence = ranked_emotions(normalized)[0]
    return EmotionEstimate(
        source=source_name,
        scores=normalized,
        top_emotion=top_emotion,
        confidence=confidence,
        notes=f"Keyword-based affect score from {source_name}: {to_counter_text(normalized)}",
    )


def pil_required() -> None:
    if Image is None or ImageOps is None:
        raise RuntimeError("Pillow is required for image-based emotion analysis.")


def open_image(input_file) -> "Image.Image":
    pil_required()
    if isinstance(input_file, Image.Image):
        image = input_file
    else:
        image = Image.open(input_file)
    return ImageOps.exif_transpose(image).convert("RGB")


def preprocess_pil_for_model(image: "Image.Image", image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE):
    pil_required()
    resized = ImageOps.fit(image.convert("RGB"), image_size)
    array = np.asarray(resized).astype("float32")
    return array


def heuristic_face_scores(image: "Image.Image") -> Dict[str, float]:
    pil_required()
    gray = image.convert("L").resize((96, 96))
    arr = np.asarray(gray).astype("float32") / 255.0
    brightness = float(arr.mean())
    contrast = float(arr.std())
    upper_mean = float(arr[:48].mean())
    lower_mean = float(arr[48:].mean())
    left_mean = float(arr[:, :48].mean())
    right_mean = float(arr[:, 48:].mean())
    symmetry = 1.0 - abs(left_mean - right_mean)

    scores = make_base_scores()
    scores["happy"] += max(brightness - 0.50, 0.0) * 1.8 + max(lower_mean - upper_mean, 0.0) * 1.0
    scores["sad"] += max(0.55 - brightness, 0.0) * 1.6 + max(upper_mean - lower_mean, 0.0) * 0.8
    scores["angry"] += max(contrast - 0.20, 0.0) * 2.0 + max(upper_mean - lower_mean, 0.0) * 0.7
    scores["neutral"] += symmetry * 0.9 + (0.20 - abs(brightness - 0.50)) * 0.8
    scores["surprise"] += max(contrast - 0.18, 0.0) * 0.7 + max(brightness - 0.60, 0.0) * 0.4
    scores["fear"] += max(contrast - 0.22, 0.0) * 0.8 + max(0.47 - brightness, 0.0) * 0.5
    scores["calm"] += max(0.18 - contrast, 0.0) * 1.4 + symmetry * 0.5
    return normalize_scores(scores)


def tensorflow_available() -> bool:
    return tf is not None


def load_saved_model():
    if not tensorflow_available() or not MODEL_PATH.exists():
        return None
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception:
        return None


def estimate_from_face_image(image: "Image.Image", model=None) -> EmotionEstimate:
    if model is not None and tensorflow_available():
        prepared = preprocess_pil_for_model(image)
        batch = np.expand_dims(prepared, axis=0)
        probabilities = model.predict(batch, verbose=0)[0]
        class_names = load_json(MODEL_METADATA_PATH).get("class_names", PRIMARY_EMOTIONS)
        scores = make_base_scores()
        for name, score in zip(class_names, probabilities.tolist()):
            scores[name] = float(score)
        normalized = normalize_scores(scores)
        top_emotion, confidence = ranked_emotions(normalized)[0]
        return EmotionEstimate(
            source="face-image",
            scores=normalized,
            top_emotion=top_emotion,
            confidence=confidence,
            notes="Prediction generated by saved EfficientNetV2 + GAP model.",
        )

    normalized = heuristic_face_scores(image)
    top_emotion, confidence = ranked_emotions(normalized)[0]
    return EmotionEstimate(
        source="face-image",
        scores=normalized,
        top_emotion=top_emotion,
        confidence=confidence,
        notes="Heuristic fallback used because no trained TensorFlow model was available.",
    )


def fuse_multimodal_estimates(
    estimates: List[EmotionEstimate],
    weights: Dict[str, float],
) -> EmotionEstimate:
    combined = defaultdict(float)
    total_weight = 0.0
    notes = []
    for estimate in estimates:
        weight = max(weights.get(estimate.source, 0.0), 0.0)
        if weight <= 0:
            continue
        total_weight += weight
        for emotion, score in estimate.scores.items():
            combined[emotion] += score * weight
        notes.append(f"{estimate.source}:{weight:.2f}")
    if total_weight <= 0:
        return EmotionEstimate(
            source="fusion",
            scores=normalize_scores(make_base_scores()),
            top_emotion="neutral",
            confidence=1.0 / len(EMOTIONS),
            notes="No active modalities were selected.",
        )
    normalized = normalize_scores(dict(combined))
    top_emotion, confidence = ranked_emotions(normalized)[0]
    return EmotionEstimate(
        source="fusion",
        scores=normalized,
        top_emotion=top_emotion,
        confidence=confidence,
        notes=f"Weighted fusion of modalities ({', '.join(notes)}).",
    )


def build_adaptive_plan(emotion: str, energy: int, stress: int, sleep_hours: float) -> List[Dict[str, str]]:
    base_actions = {
        "happy": [
            ("Study mode", "Use high-focus tasks while motivation is naturally high."),
            ("Movement", "Add a short walk or stretch between long work blocks."),
            ("Reflection", "Capture what is going well in a daily win log."),
        ],
        "sad": [
            ("Reset", "Start with a low-friction task for 10 minutes to regain momentum."),
            ("Support", "Use gentle music, journaling, or a trusted peer check-in."),
            ("Sleep hygiene", "Reduce blue-light exposure and target a fixed bedtime."),
        ],
        "angry": [
            ("Cooling strategy", "Delay major decisions for 10 minutes and use paced breathing."),
            ("Task slicing", "Break large work into micro tasks to reduce overload."),
            ("Environment", "Lower notification noise and remove obvious stress triggers."),
        ],
        "neutral": [
            ("Planning", "Use the stable state to prioritize one deep-work task."),
            ("Review", "Check goals, deadlines, and progress tracker."),
            ("Balance", "Add hydration, posture correction, and a movement break."),
        ],
        "surprise": [
            ("Stabilize", "Write down the new event, risk, impact, and next action."),
            ("Buffer", "Avoid overcommitting before the situation is clear."),
            ("Check-in", "Confirm facts and dependencies before responding."),
        ],
        "fear": [
            ("Grounding", "Use 5-4-3-2-1 grounding or box breathing."),
            ("Risk framing", "Write controllable vs. uncontrollable factors."),
            ("Confidence recovery", "Complete one small task to rebuild agency."),
        ],
        "calm": [
            ("Deep work", "Schedule the most conceptually demanding task here."),
            ("Learning", "Use the calm state for revision and retention."),
            ("Maintenance", "Lock in the routine that produced this state."),
        ],
    }
    plan = []
    for title, action in base_actions.get(emotion, base_actions["neutral"]):
        plan.append({"title": title, "action": action})
    if energy <= 4:
        plan.append(
            {
                "title": "Low-energy safeguard",
                "action": "Reduce batch size, shorten work sprints, and avoid aggressive multitasking.",
            }
        )
    if stress >= 7:
        plan.append(
            {
                "title": "High-stress safeguard",
                "action": "Switch to single-task mode, slow breathing for 2 minutes, and defer nonessential work.",
            }
        )
    if sleep_hours < 6:
        plan.append(
            {
                "title": "Recovery note",
                "action": "Treat outputs as lower-confidence until rest improves; fatigue can distort emotional signals.",
            }
        )
    return plan


def build_ethical_ai_report(
    fusion: EmotionEstimate,
    active_modalities: List[str],
    consent: bool,
    dataset_manifest: Dict[str, object],
) -> List[Dict[str, str]]:
    report = [
        {
            "area": "Consent and logging",
            "status": "enabled" if consent else "disabled",
            "detail": (
                "Digital Emotional Twin logging is allowed by the user."
                if consent
                else "No personal state should be stored until consent is enabled."
            ),
        },
        {
            "area": "Model transparency",
            "status": "watch",
            "detail": "Explainability is available through Grad-CAM when a trained CNN exists.",
        },
        {
            "area": "Multimodal reliability",
            "status": "watch" if len(active_modalities) < 2 else "good",
            "detail": (
                "Using a single modality raises uncertainty and should be treated as advisory only."
                if len(active_modalities) < 2
                else "Multiple modalities reduce single-source bias, but not perfectly."
            ),
        },
    ]
    if fusion.confidence < 0.42:
        report.append(
            {
                "area": "Prediction confidence",
                "status": "warning",
                "detail": "Low confidence detected. Ask for self-report confirmation before acting on the result.",
            }
        )
    class_balance = dataset_manifest.get("class_distribution", {})
    if class_balance:
        counts = list(class_balance.values())
        if counts and max(counts) > 2.5 * max(min(counts), 1):
            report.append(
                {
                    "area": "Training fairness",
                    "status": "warning",
                    "detail": "Dataset appears imbalanced. Consider resampling or class weighting before final reporting.",
                }
            )
    return report


def spotify_client_token(client_id: str, client_secret: str) -> Optional[str]:
    if not client_id or not client_secret:
        return None
    raw = f"{client_id}:{client_secret}".encode("utf-8")
    headers = {"Authorization": "Basic " + base64.b64encode(raw).decode("utf-8")}
    data = {"grant_type": "client_credentials"}
    try:
        response = requests.post(
            "https://accounts.spotify.com/api/token",
            headers=headers,
            data=data,
            timeout=20,
        )
        response.raise_for_status()
        return response.json().get("access_token")
    except Exception:
        return None


def spotify_search(query: str, access_token: str, limit: int = 5) -> List[Dict[str, str]]:
    if not access_token:
        return []
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {"q": query, "type": "track,playlist", "limit": str(limit)}
    try:
        response = requests.get(
            "https://api.spotify.com/v1/search",
            headers=headers,
            params=params,
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return []

    results = []
    for track in payload.get("tracks", {}).get("items", []):
        results.append(
            {
                "name": track.get("name", "Unknown track"),
                "artist": ", ".join(artist["name"] for artist in track.get("artists", [])),
                "url": track.get("external_urls", {}).get("spotify", ""),
                "kind": "track",
            }
        )
    for playlist in payload.get("playlists", {}).get("items", []):
        results.append(
            {
                "name": playlist.get("name", "Unknown playlist"),
                "artist": playlist.get("owner", {}).get("display_name", "Spotify"),
                "url": playlist.get("external_urls", {}).get("spotify", ""),
                "kind": "playlist",
            }
        )
    return results[:limit]


def build_resource_catalog() -> List[Dict[str, str]]:
    mood_queries = {
        "happy": ["uplifting focus", "morning motivation", "celebration coding"],
        "sad": ["healing calm", "gentle piano", "soft reflection"],
        "angry": ["stress release", "calm breathing", "reset focus"],
        "neutral": ["deep work", "study beats", "instrumental concentration"],
        "surprise": ["steady recovery", "clarity playlist", "adaptive focus"],
        "fear": ["grounding audio", "confidence building", "anxiety relief"],
        "calm": ["meditation focus", "ambient coding", "slow clarity"],
    }
    themes = [
        "focus",
        "study",
        "relaxation",
        "motivation",
        "breathing",
        "journaling",
        "fitness",
        "sleep",
        "mindfulness",
        "coding",
    ]
    catalog = []
    for mood, seeds in mood_queries.items():
        for theme in themes:
            for seed in seeds:
                query = f"{mood} {theme} {seed}"
                title = f"{mood.title()} {theme.title()} - {seed.title()}"
                catalog.append(
                    {
                        "mood": mood,
                        "title": title,
                        "url": f"https://www.youtube.com/results?search_query={quote_plus(query)}",
                        "source": "YouTube",
                        "type": "search-link",
                        "offline_fallback": f"Create an offline list named '{title}' with 3 locally saved notes or activities.",
                    }
                )
                catalog.append(
                    {
                        "mood": mood,
                        "title": title,
                        "url": f"https://open.spotify.com/search/{quote_plus(query)}",
                        "source": "Spotify",
                        "type": "search-link",
                        "offline_fallback": f"Save 5 local song names tagged with '{mood}' and '{theme}'.",
                    }
                )
                catalog.append(
                    {
                        "mood": mood,
                        "title": title,
                        "url": f"https://music.youtube.com/search?q={quote_plus(query)}",
                        "source": "YouTube Music",
                        "type": "search-link",
                        "offline_fallback": "Use a local breathing, study, or meditation timer instead.",
                    }
                )
                catalog.append(
                    {
                        "mood": mood,
                        "title": f"{title} Routine",
                        "url": "offline://adaptive-routine",
                        "source": "Offline",
                        "type": "fallback-plan",
                        "offline_fallback": (
                            f"10 minutes {theme}, 25 minutes single-task work, 2 minutes reflection for mood {mood}."
                        ),
                    }
                )
    return catalog


RESOURCE_CATALOG = build_resource_catalog()


def export_resource_catalog(path: Path = RESOURCE_CATALOG_PATH) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(RESOURCE_CATALOG[0].keys()))
        writer.writeheader()
        writer.writerows(RESOURCE_CATALOG)
    return len(RESOURCE_CATALOG)


def recommend_resources(
    emotion: str,
    history: Iterable[str],
    energy: int,
    goal: str,
    limit: int = 8,
) -> List[Dict[str, str]]:
    history_set = set(history)
    scored = []
    for item in RESOURCE_CATALOG:
        score = 0.0
        if item["mood"] == emotion:
            score += 3.0
        if goal and goal.lower() in item["title"].lower():
            score += 1.2
        if energy <= 4 and "sleep" in item["title"].lower():
            score += 0.8
        if energy >= 7 and "motivation" in item["title"].lower():
            score += 0.7
        if item["title"] in history_set:
            score -= 2.2
        scored.append((score, item))
    ranked = [item for _, item in sorted(scored, key=lambda entry: entry[0], reverse=True)]
    unique = []
    seen_titles = set()
    for item in ranked:
        if item["title"] in seen_titles:
            continue
        seen_titles.add(item["title"])
        unique.append(item)
        if len(unique) >= limit:
            break
    return unique


def build_local_viva_questions() -> List[Dict[str, object]]:
    return [
        {
            "question": "Why did you replace MobileNetV2 with EfficientNetV2 and Global Average Pooling?",
            "answer_points": [
                "EfficientNetV2 is a newer family with stronger accuracy-efficiency trade-offs.",
                "Global Average Pooling reduces parameters and overfitting compared to large dense heads.",
                "The combination is better aligned with explainable CNN pipelines and low-memory deployment.",
            ],
        },
        {
            "question": "What is the role of Grad-CAM in this project?",
            "answer_points": [
                "Grad-CAM highlights the facial regions that influenced the CNN prediction.",
                "It improves trust, debugging, and reportability in a sensitive affective AI system.",
                "It supports ethical AI monitoring by exposing when the model attends to irrelevant regions.",
            ],
        },
        {
            "question": "What is meant by Digital Emotional Twin?",
            "answer_points": [
                "It is a structured longitudinal log of user state, context, outputs, and feedback.",
                "The twin helps the adaptive system learn routine patterns without storing unnecessary raw media.",
                "It supports explainability, personalization, and auditability.",
            ],
        },
        {
            "question": "How does reinforcement-learning logic appear in this system?",
            "answer_points": [
                "The current version uses a reinforcement-inspired reward loop rather than a heavy RL agent.",
                "User feedback changes recommendation preference and novelty penalties.",
                "This keeps the system adaptive while staying lightweight for a major-project deployment.",
            ],
        },
        {
            "question": "Why is ethical AI monitoring necessary in emotion recognition?",
            "answer_points": [
                "Emotion inference is sensitive and can be wrong in high-stakes contexts.",
                "Ethical monitoring flags low confidence, missing consent, imbalance, and transparency gaps.",
                "It prevents blind automation and keeps the system human-in-the-loop.",
            ],
        },
        {
            "question": "What are the main research gaps your project addresses?",
            "answer_points": [
                "Many systems are unimodal, not explainable, and not deployment-aware.",
                "Several prior works ignore low-resource practical deployment and digital twin audit trails.",
                "This project combines multimodal fusion, adaptive recommendations, explainability, and ethics in one prototype.",
            ],
        },
        {
            "question": "What is the practical difference between mood detection and mood-aware recommendation?",
            "answer_points": [
                "Mood detection estimates the user's present affective state.",
                "Mood-aware recommendation uses that estimate to change outputs such as music, routines, or questions.",
                "The second task must also consider safety, novelty, and user control.",
            ],
        },
        {
            "question": "How does this project handle the cold-start problem?",
            "answer_points": [
                "It uses immediate multimodal signals instead of relying only on long listening history.",
                "Self-report, emoji, text, and image inputs bootstrap personalization for new users.",
                "Recommendation history and feedback then refine the adaptive layer over time.",
            ],
        },
        {
            "question": "Why did you keep the project as a single-file Streamlit application?",
            "answer_points": [
                "It reduces setup complexity for evaluation and viva demonstration.",
                "A single-file architecture is easy to run in VS Code and on low-end machines.",
                "The design still separates logic into functions so it can be refactored later if needed.",
            ],
        },
        {
            "question": "Can this project be converted into an APK?",
            "answer_points": [
                "Yes, the easiest free route is to deploy the Streamlit app and wrap it as a PWA-based Android package.",
                "The repo documents a PWABuilder route and also mentions native alternatives if needed.",
                "A true offline native APK would require a different GUI stack such as Kivy or Flutter.",
            ],
        },
    ]


def build_mood_questions_local(emotion: str, project_stage: str) -> List[str]:
    prompts = {
        "happy": [
            "Which ambitious feature can you complete now while your confidence is high?",
            "What result from your project are you most proud of, and how will you explain it clearly?",
            "How can you convert your current energy into a polished demo or paper improvement?",
        ],
        "sad": [
            "What is the smallest achievable task that can help you regain momentum today?",
            "Which part of the project feels heavy right now, and how can you simplify it?",
            "What evidence shows that your work still has value even if progress feels slow today?",
        ],
        "angry": [
            "What exactly is causing friction: code, time pressure, or expectations?",
            "Which issue should be parked for later so it does not damage your main delivery?",
            "What calm, factual explanation will you give if a viva question challenges your design choice?",
        ],
        "fear": [
            "Which viva question scares you most, and what three-point answer can neutralize it?",
            "What backup demo path will you use if the trained model cannot run on the presentation machine?",
            "What can you verify right now to reduce uncertainty before submission?",
        ],
        "neutral": [
            "What is the next most rational task for the current project stage?",
            "Which document or code section needs better clarity before review?",
            "What one metric or screenshot will strengthen your demonstration?",
        ],
        "surprise": [
            "What changed, and does it alter your scope, risk, or presentation strategy?",
            "Which unexpected result is worth discussing as a learning point?",
            "How will you convert this new event into a stronger discussion answer?",
        ],
        "calm": [
            "Which deep concept can you revise now for a stronger viva performance?",
            "What architecture decision deserves a clean, diagram-driven explanation?",
            "How can you use this stable state to improve both code quality and the paper?",
        ],
    }
    questions = prompts.get(emotion, prompts["neutral"])
    stage_prompt = f"Project stage focus: {project_stage}."
    return [f"{stage_prompt} {question}" for question in questions]


def hf_text_generation(prompt: str, token: str, model_name: str) -> str:
    if not token or InferenceClient is None:
        return ""
    try:
        client = InferenceClient(model=model_name, token=token)
        if hasattr(client, "chat_completion"):
            response = client.chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a concise academic mentor for a final-year engineering major project.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=700,
            )
            choice = response.choices[0]
            return choice.message.content.strip()
        if hasattr(client, "text_generation"):
            return client.text_generation(prompt, max_new_tokens=700).strip()
    except Exception:
        return ""
    return ""


def to_markdown_bullets(items: List[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def detect_dataset_image_and_label_columns(dataset_split) -> Tuple[Optional[str], Optional[str]]:
    sample = dataset_split[0]
    image_column = None
    label_column = None
    for key, value in sample.items():
        if image_column is None and (
            (Image is not None and isinstance(value, Image.Image))
            or isinstance(value, dict)
            or (isinstance(value, str) and value.lower().endswith((".png", ".jpg", ".jpeg")))
        ):
            image_column = key
        key_lower = key.lower()
        if label_column is None and any(token in key_lower for token in ["label", "emotion", "class", "target"]):
            label_column = key
    return image_column, label_column


def dataset_value_to_pil(value) -> "Image.Image":
    pil_required()
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes"):
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path"):
            return Image.open(value["path"]).convert("RGB")
    if isinstance(value, str) and Path(value).exists():
        return Image.open(value).convert("RGB")
    raise ValueError("Could not convert dataset image value to a PIL image.")


def sanitize_label(label: object) -> str:
    label_text = str(label).strip().lower().replace("/", "-").replace(" ", "_")
    return "".join(ch for ch in label_text if ch.isalnum() or ch in {"_", "-"})


def prepare_sampled_hf_dataset(
    dataset_id: str,
    sample_size: int = 800,
    seed: int = 42,
) -> Dict[str, object]:
    if load_dataset is None:
        raise RuntimeError("The 'datasets' package is required for automatic dataset preparation.")
    if Image is None:
        raise RuntimeError("Pillow is required for image dataset preparation.")

    ensure_directories()
    dataset = load_dataset(dataset_id)
    if isinstance(dataset, DatasetDict):
        split_name = "train" if "train" in dataset else list(dataset.keys())[0]
        split = dataset[split_name]
    else:
        split_name = "train"
        split = dataset

    image_column, label_column = detect_dataset_image_and_label_columns(split)
    if not image_column or not label_column:
        raise RuntimeError("Could not detect compatible image and label columns in the selected dataset.")

    total = len(split)
    sample_size = min(sample_size, total)
    indexes = list(range(total))
    rng = random.Random(seed)
    rng.shuffle(indexes)
    selected_indexes = indexes[:sample_size]
    sampled_rows = [split[int(index)] for index in selected_indexes]

    label_names = None
    if ClassLabel is not None:
        features = getattr(split, "features", {})
        feature = features.get(label_column) if features else None
        if isinstance(feature, ClassLabel):
            label_names = feature.names

    for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        if folder.exists():
            for item in folder.iterdir():
                if item.is_dir():
                    for child in item.iterdir():
                        child.unlink()
                    item.rmdir()

    rows_by_label = defaultdict(list)
    for row in sampled_rows:
        label_value = row[label_column]
        if label_names and isinstance(label_value, int):
            label = sanitize_label(label_names[label_value])
        else:
            label = sanitize_label(label_value)
        rows_by_label[label].append(row)

    manifest = {
        "dataset_id": dataset_id,
        "prepared_at": now_iso(),
        "split_used": split_name,
        "sample_size": sample_size,
        "image_column": image_column,
        "label_column": label_column,
        "class_distribution": {},
    }

    for label, rows in rows_by_label.items():
        rng.shuffle(rows)
        train_cut = max(1, int(len(rows) * 0.70))
        val_cut = max(train_cut + 1, int(len(rows) * 0.85)) if len(rows) > 2 else len(rows)
        split_map = {
            TRAIN_DIR: rows[:train_cut],
            VAL_DIR: rows[train_cut:val_cut],
            TEST_DIR: rows[val_cut:],
        }
        manifest["class_distribution"][label] = len(rows)
        for base_path, chunk in split_map.items():
            label_dir = base_path / label
            label_dir.mkdir(parents=True, exist_ok=True)
            for index, row in enumerate(chunk):
                image = dataset_value_to_pil(row[image_column])
                output_path = label_dir / f"{label}_{index:04d}.png"
                image.save(output_path)

    save_json(DATASET_MANIFEST_PATH, manifest)
    return manifest


def build_efficientnetv2_model(num_classes: int):
    if not tensorflow_available():
        raise RuntimeError("TensorFlow is required for training.")
    inputs = tf.keras.Input(shape=(DEFAULT_IMAGE_SIZE[0], DEFAULT_IMAGE_SIZE[1], 3))
    base = tf.keras.applications.EfficientNetV2B0(
        include_top=False,
        input_tensor=None,
        input_shape=(DEFAULT_IMAGE_SIZE[0], DEFAULT_IMAGE_SIZE[1], 3),
        weights="imagenet",
    )
    base.trainable = False
    x = tf.keras.applications.efficientnet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="emotion_head")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def collect_labels_from_dataset(dataset) -> np.ndarray:
    labels = []
    for _, batch_labels in dataset:
        labels.extend(batch_labels.numpy().tolist())
    return np.array(labels)


def train_emotion_model(batch_size: int = 8, epochs: int = 2) -> Dict[str, object]:
    if not tensorflow_available():
        raise RuntimeError("TensorFlow is required for training.")
    if not TRAIN_DIR.exists() or not any(TRAIN_DIR.glob("*/*")):
        raise RuntimeError("Prepared dataset not found. Use the dataset preparation section first.")

    ensure_directories()
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=DEFAULT_IMAGE_SIZE,
        batch_size=batch_size,
        label_mode="int",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=DEFAULT_IMAGE_SIZE,
        batch_size=batch_size,
        label_mode="int",
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=DEFAULT_IMAGE_SIZE,
        batch_size=batch_size,
        label_mode="int",
        shuffle=False,
    )

    class_names = train_ds.class_names
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)

    model = build_efficientnetv2_model(num_classes=len(class_names))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=2,
            restore_best_weights=True,
        )
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=1)
    model.save(MODEL_PATH)

    prediction_array = model.predict(test_ds, verbose=0)
    predicted_labels = np.argmax(prediction_array, axis=1)
    true_labels = collect_labels_from_dataset(test_ds)

    metrics = {
        "class_names": class_names,
        "history": {key: [float(value) for value in values] for key, values in history.history.items()},
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "roc_auc_ovr": None,
        "confusion_matrix": None,
        "trained_at": now_iso(),
        "architecture": "EfficientNetV2B0 + GlobalAveragePooling2D + Dense",
        "batch_size": batch_size,
        "epochs": epochs,
        "model_path": str(MODEL_PATH.name),
    }
    if accuracy_score is not None:
        metrics["accuracy"] = float(accuracy_score(true_labels, predicted_labels))
    if precision_recall_fscore_support is not None:
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels,
            predicted_labels,
            average="weighted",
            zero_division=0,
        )
        metrics["precision"] = float(precision)
        metrics["recall"] = float(recall)
        metrics["f1"] = float(f1)
    if confusion_matrix is not None:
        metrics["confusion_matrix"] = confusion_matrix(true_labels, predicted_labels).tolist()
    if roc_auc_score is not None and label_binarize is not None and len(class_names) > 2:
        try:
            y_true = label_binarize(true_labels, classes=list(range(len(class_names))))
            metrics["roc_auc_ovr"] = float(roc_auc_score(y_true, prediction_array, multi_class="ovr"))
        except Exception:
            metrics["roc_auc_ovr"] = None

    save_json(MODEL_METADATA_PATH, metrics)
    return metrics


def find_last_conv_layer_name(model) -> Optional[str]:
    if not tensorflow_available() or model is None:
        return None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        if isinstance(layer, tf.keras.Model):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    return sub_layer.name
    return None


def generate_gradcam_overlay(model, image: "Image.Image") -> Optional["Image.Image"]:
    if not tensorflow_available() or model is None or cm is None:
        return None
    last_conv_name = find_last_conv_layer_name(model)
    if not last_conv_name:
        return None

    input_array = preprocess_pil_for_model(image)
    input_batch = np.expand_dims(input_array, axis=0)
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_name).output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_batch)
        class_index = tf.argmax(predictions[0])
        class_channel = predictions[:, class_index]

    gradients = tape.gradient(class_channel, conv_outputs)
    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_gradients, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap.numpy(), 0)
    max_value = np.max(heatmap) or 1.0
    heatmap = heatmap / max_value

    heatmap_image = Image.fromarray(np.uint8(cm.jet(heatmap)[:, :, :3] * 255)).resize(image.size)
    overlay = Image.blend(image.convert("RGB"), heatmap_image, alpha=0.38)
    return overlay


def figure_from_confusion_matrix(matrix: List[List[int]], labels: List[str]):
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            ax.text(col_index, row_index, str(value), ha="center", va="center")
    fig.tight_layout()
    return fig


def build_summary_cards() -> List[Tuple[str, str]]:
    return [
        ("Protagonist", "A single-file CEI system that fuses mood signals and adapts lifestyle guidance."),
        ("Difference", "It combines multimodal fusion, explainability, ethical AI checks, and adaptive logging."),
        ("Research gap", "Many 2023-2025 systems remain unimodal, opaque, or hard to deploy on low-end devices."),
        ("Patent angle", "The strongest novelty is the integrated digital emotional twin + adaptive routine loop."),
    ]


def render_dataframe_or_table(data, height: int = 280) -> None:
    if st is None:
        return
    if pd is not None and isinstance(data, pd.DataFrame):
        st.dataframe(data, use_container_width=True, height=height)
    elif pd is not None and isinstance(data, list) and data and isinstance(data[0], dict):
        st.dataframe(pd.DataFrame(data), use_container_width=True, height=height)
    else:
        st.write(data)


def app_sidebar_defaults():
    if "recommendation_history" not in st.session_state:
        st.session_state.recommendation_history = []
    if "latest_fusion" not in st.session_state:
        st.session_state.latest_fusion = None
    if "latest_metrics" not in st.session_state:
        st.session_state.latest_metrics = load_json(MODEL_METADATA_PATH)


def render_streamlit_app() -> None:
    ensure_directories()
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    app_sidebar_defaults()

    st.title(APP_TITLE)
    st.caption(APP_TAGLINE)

    with st.sidebar:
        st.header("Run configuration")
        consent = st.checkbox("Allow Digital Emotional Twin logging", value=True)
        user_id = st.text_input("User or demo ID", value="demo-user")
        hf_token = st.text_input("Hugging Face token (optional)", type="password")
        hf_model = st.text_input(
            "Hugging Face text model",
            value="mistralai/Mistral-7B-Instruct-v0.3",
        )
        spotify_client_id = st.text_input("Spotify client ID (optional)")
        spotify_client_secret = st.text_input("Spotify client secret (optional)", type="password")
        st.divider()
        st.subheader("Fusion weights")
        face_weight = st.slider("Face image", 0.0, 1.0, 0.45, 0.05)
        emoji_weight = st.slider("Emoji", 0.0, 1.0, 0.15, 0.05)
        text_weight = st.slider("Free text", 0.0, 1.0, 0.25, 0.05)
        voice_weight = st.slider("Voice transcript", 0.0, 1.0, 0.15, 0.05)
        weights = {
            "face-image": face_weight,
            "emoji": emoji_weight,
            "text": text_weight,
            "voice-transcript": voice_weight,
        }

    overview_tab, fusion_tab, mentor_tab, train_tab, results_tab, rec_tab, catalog_tab = st.tabs(
        [
            "Overview",
            "Emotion Fusion",
            "AI Mentor & Viva",
            "Dataset & Training",
            "Results & Explainability",
            "Recommendations",
            "Resource Catalog",
        ]
    )

    with overview_tab:
        st.subheader("What this repository now contains")
        left, right = st.columns([1, 1])
        with left:
            for title, description in build_summary_cards():
                st.markdown(f"**{title}:** {description}")
            st.markdown(
                """
                **Core stack**
                - Single-file Streamlit app
                - Optional TensorFlow training path
                - EfficientNetV2 + Global Average Pooling
                - Grad-CAM explainability
                - CSV-based Digital Emotional Twin
                - Free local fallback plus optional Hugging Face and Spotify APIs
                """
            )
        with right:
            st.info(
                "OpenAI is not required here. The default free execution path uses local heuristics and optional "
                "Hugging Face inference. Spotify developer credentials are free to create, but not required."
            )
            st.warning(
                "No project can honestly guarantee a patent, publication, or a specific SGPA. "
                "This prototype is designed to strengthen the technical and presentation quality of the work."
            )

    with fusion_tab:
        st.subheader("Multimodal emotion fusion")
        col1, col2 = st.columns([1.0, 1.1])
        with col1:
            uploaded_image = st.file_uploader("Upload face image", type=["png", "jpg", "jpeg"])
            selected_emoji = st.selectbox("Select current emoji", list(EMOJI_TO_EMOTION.keys()), index=0)
            free_text = st.text_area(
                "Context text",
                placeholder="Describe your current workload, mood, challenges, and what you need from the system.",
            )
            voice_transcript = st.text_area(
                "Voice-to-text fallback",
                placeholder="Paste speech transcript here if you do not want to use a live speech API.",
            )
            goal = st.selectbox(
                "Current goal",
                [
                    "deep work",
                    "major project coding",
                    "paper writing",
                    "viva preparation",
                    "stress reset",
                    "sleep recovery",
                ],
            )
            energy = st.slider("Energy level", 1, 10, 6)
            stress = st.slider("Stress level", 1, 10, 5)
            sleep_hours = st.slider("Sleep last night", 0.0, 10.0, 6.5, 0.5)
            analyze_clicked = st.button("Analyze current state", type="primary")

        with col2:
            estimates = []
            saved_model = load_saved_model()
            if analyze_clicked:
                if uploaded_image is not None:
                    image = open_image(uploaded_image)
                    st.image(image, caption="Uploaded face image", width=300)
                    estimates.append(estimate_from_face_image(image, saved_model))
                estimates.append(estimate_from_emoji(selected_emoji))
                estimates.append(estimate_from_text(free_text, source_name="text"))
                estimates.append(estimate_from_text(voice_transcript, source_name="voice-transcript"))
                fusion = fuse_multimodal_estimates(estimates, weights)
                st.session_state.latest_fusion = {
                    "scores": fusion.scores,
                    "top_emotion": fusion.top_emotion,
                    "confidence": fusion.confidence,
                    "notes": fusion.notes,
                    "goal": goal,
                    "energy": energy,
                    "stress": stress,
                    "sleep_hours": sleep_hours,
                    "active_modalities": [estimate.source for estimate in estimates if estimate.confidence > 0],
                }

                st.success(
                    f"Detected state: {fusion.top_emotion.upper()} "
                    f"(confidence {fusion.confidence:.2f})"
                )
                if pd is not None:
                    scores_df = pd.DataFrame(
                        [{"emotion": emotion, "score": score} for emotion, score in fusion.scores.items()]
                    )
                    st.bar_chart(scores_df.set_index("emotion"))
                else:
                    st.write(fusion.scores)

                with st.expander("Per-modality details", expanded=True):
                    for estimate in estimates:
                        st.markdown(
                            f"**{estimate.source}** -> {estimate.top_emotion} ({estimate.confidence:.2f})"
                        )
                        st.caption(estimate.notes)

                plan = build_adaptive_plan(fusion.top_emotion, energy, stress, sleep_hours)
                st.markdown("**Adaptive lifestyle plan**")
                for step_index, item in enumerate(plan, start=1):
                    st.markdown(f"{step_index}. **{item['title']}** - {item['action']}")

                ethical_report = build_ethical_ai_report(
                    fusion,
                    [estimate.source for estimate in estimates if estimate.source],
                    consent,
                    load_json(DATASET_MANIFEST_PATH),
                )
                st.markdown("**Ethical AI monitoring**")
                render_dataframe_or_table(ethical_report)

                if consent:
                    csv_append(
                        TWIN_LOG_PATH,
                        {
                            "timestamp": now_iso(),
                            "user_id": user_id,
                            "emotion": fusion.top_emotion,
                            "confidence": round(fusion.confidence, 4),
                            "goal": goal,
                            "energy": energy,
                            "stress": stress,
                            "sleep_hours": sleep_hours,
                            "modalities": "|".join([estimate.source for estimate in estimates]),
                            "scores_json": json.dumps(fusion.scores),
                        },
                    )
                    st.caption("Digital Emotional Twin log updated.")

            elif st.session_state.latest_fusion:
                fusion = st.session_state.latest_fusion
                st.info(
                    f"Latest fused state: {fusion['top_emotion']} "
                    f"(confidence {fusion['confidence']:.2f})"
                )

    with mentor_tab:
        st.subheader("Free mentor mode and viva preparation")
        current_emotion = "neutral"
        current_goal = "major project coding"
        if st.session_state.latest_fusion:
            current_emotion = st.session_state.latest_fusion["top_emotion"]
            current_goal = st.session_state.latest_fusion["goal"]

        project_stage = st.selectbox(
            "Project stage",
            ["proposal", "coding", "model training", "paper writing", "ppt preparation", "viva revision"],
            index=1,
        )
        extra_context = st.text_area(
            "Additional mentor context",
            value="My final-year project is about Cognitive Emotion Intelligence and Adaptive Lifestyle Support.",
        )

        local_questions = build_mood_questions_local(current_emotion, project_stage)
        st.markdown("**Mindset-aware important questions**")
        st.markdown(to_markdown_bullets(local_questions))

        viva_questions = build_local_viva_questions()
        with st.expander("Top HOD and interviewer viva questions", expanded=True):
            for question in viva_questions:
                st.markdown(f"**Q. {question['question']}**")
                for point in question["answer_points"]:
                    st.markdown(f"- {point}")

        if hf_token:
            prompt = textwrap.dedent(
                f"""
                Create 8 concise but high-value viva questions and answer pointers for an engineering major project.
                Project title: {APP_TITLE}
                Detected mood: {current_emotion}
                Goal: {current_goal}
                Stage: {project_stage}
                Context: {extra_context}
                Keep the style practical, viva-oriented, and easy to memorize.
                """
            ).strip()
            generated = hf_text_generation(prompt, hf_token, hf_model)
            if generated:
                st.markdown("**Optional Hugging Face generated coaching output**")
                st.code(generated)
            else:
                st.caption("Hugging Face generation was requested but no response was returned. Local fallback remains active.")

    with train_tab:
        st.subheader("Automatic dataset preparation and EfficientNetV2 training")
        st.markdown(
            """
            **Recommended Windows 11 low-RAM settings**
            - sample size: 600 to 1200 images
            - batch size: 4 or 8
            - epochs: 1 to 3 for demonstration
            - start with fewer classes before scaling up
            """
        )

        dataset_choice = st.selectbox("Hugging Face dataset", list(HF_DATASET_OPTIONS.keys()))
        sample_size = st.slider("Sample size", 100, 1500, 800, 100)
        batch_size = st.selectbox("Batch size", [4, 8, 16], index=1)
        epochs = st.slider("Epochs", 1, 5, 2)

        col_prepare, col_train = st.columns(2)
        with col_prepare:
            if st.button("Prepare sampled dataset"):
                try:
                    manifest = prepare_sampled_hf_dataset(HF_DATASET_OPTIONS[dataset_choice], sample_size=sample_size)
                    st.success("Dataset prepared successfully.")
                    render_dataframe_or_table([manifest])
                except Exception as exc:
                    st.error(str(exc))
        with col_train:
            if st.button("Train EfficientNetV2 + GAP model"):
                try:
                    metrics = train_emotion_model(batch_size=batch_size, epochs=epochs)
                    st.session_state.latest_metrics = metrics
                    st.success("Training finished and model saved.")
                    render_dataframe_or_table([metrics])
                except Exception as exc:
                    st.error(str(exc))

        manifest = load_json(DATASET_MANIFEST_PATH)
        if manifest:
            st.markdown("**Current dataset manifest**")
            render_dataframe_or_table([manifest])

    with results_tab:
        st.subheader("Results, discussion metrics, and Grad-CAM")
        metrics = st.session_state.latest_metrics or load_json(MODEL_METADATA_PATH)
        if metrics:
            metric_cards = {
                "Accuracy": metrics.get("accuracy"),
                "Precision": metrics.get("precision"),
                "Recall": metrics.get("recall"),
                "F1": metrics.get("f1"),
                "ROC-AUC OVR": metrics.get("roc_auc_ovr"),
            }
            cols = st.columns(len(metric_cards))
            for column, (label, value) in zip(cols, metric_cards.items()):
                display_value = "N/A" if value in (None, "") else f"{float(value):.4f}"
                column.metric(label, display_value)
            st.caption(metrics.get("architecture", "No architecture metadata available."))

            if metrics.get("confusion_matrix") and metrics.get("class_names"):
                fig = figure_from_confusion_matrix(metrics["confusion_matrix"], metrics["class_names"])
                if fig is not None:
                    st.pyplot(fig)
            with st.expander("Training history"):
                render_dataframe_or_table(
                    pd.DataFrame(metrics.get("history", {})) if pd is not None else metrics.get("history", {})
                )
        else:
            st.info("Train the model first to populate evaluation metrics.")

        st.divider()
        st.markdown("**Grad-CAM explainability**")
        explain_image = st.file_uploader("Upload an image for Grad-CAM", type=["png", "jpg", "jpeg"], key="gradcam")
        if st.button("Generate Grad-CAM overlay"):
            model = load_saved_model()
            if explain_image is None:
                st.warning("Upload an image first.")
            elif model is None:
                st.warning("Train or load a saved TensorFlow model first.")
            else:
                overlay = generate_gradcam_overlay(model, open_image(explain_image))
                if overlay is None:
                    st.error("Could not produce a Grad-CAM overlay for the current model.")
                else:
                    st.image(overlay, caption="Grad-CAM overlay", width=360)

    with rec_tab:
        st.subheader("Adaptive recommendation engine")
        if st.session_state.latest_fusion:
            emotion = st.session_state.latest_fusion["top_emotion"]
            goal = st.session_state.latest_fusion["goal"]
            energy = st.session_state.latest_fusion["energy"]
        else:
            emotion = "neutral"
            goal = "deep work"
            energy = 5

        recommendations = recommend_resources(
            emotion=emotion,
            history=st.session_state.recommendation_history,
            energy=energy,
            goal=goal,
        )
        st.markdown(f"**Current adaptive target:** mood `{emotion}` and goal `{goal}`")
        render_dataframe_or_table(recommendations)

        if recommendations:
            selected_title = st.selectbox("Select a recommendation to rate", [item["title"] for item in recommendations])
            feedback = st.radio("Recommendation feedback", ["helpful", "neutral", "needs-improvement"], horizontal=True)
            if st.button("Save recommendation feedback"):
                st.session_state.recommendation_history.append(selected_title)
                csv_append(
                    RECOMMENDER_STATS_PATH,
                    {
                        "timestamp": now_iso(),
                        "user_id": user_id,
                        "emotion": emotion,
                        "goal": goal,
                        "recommended_title": selected_title,
                        "feedback": feedback,
                    },
                )
                st.success("Recommendation feedback saved.")

        st.divider()
        query = f"{emotion} {goal} playlist"
        spotify_token = spotify_client_token(spotify_client_id, spotify_client_secret)
        spotify_items = spotify_search(query, spotify_token, limit=5) if spotify_token else []
        st.markdown("**Spotify integration**")
        if spotify_items:
            render_dataframe_or_table(spotify_items)
        else:
            st.write(
                {
                    "search_url": f"https://open.spotify.com/search/{quote_plus(query)}",
                    "note": "Free fallback mode uses search links when no Spotify API credentials are supplied.",
                }
            )

        st.markdown("**Digital Emotional Twin preview**")
        render_dataframe_or_table(read_csv_preview(TWIN_LOG_PATH))

    with catalog_tab:
        st.subheader("100+ item resource catalog")
        st.write(f"Catalog size: **{len(RESOURCE_CATALOG)}** entries")
        mood_filter = st.selectbox("Filter by mood", ["all"] + sorted({item["mood"] for item in RESOURCE_CATALOG}))
        source_filter = st.selectbox(
            "Filter by source",
            ["all"] + sorted({item["source"] for item in RESOURCE_CATALOG}),
        )
        filtered = []
        for item in RESOURCE_CATALOG:
            if mood_filter != "all" and item["mood"] != mood_filter:
                continue
            if source_filter != "all" and item["source"] != source_filter:
                continue
            filtered.append(item)
        render_dataframe_or_table(filtered)

        if st.button("Export catalog to CSV"):
            count = export_resource_catalog()
            st.success(f"Exported {count} catalog entries to {RESOURCE_CATALOG_PATH.name}")
        if RESOURCE_CATALOG_PATH.exists():
            st.download_button(
                label="Download resource_catalog.csv",
                data=RESOURCE_CATALOG_PATH.read_bytes(),
                file_name=RESOURCE_CATALOG_PATH.name,
                mime="text/csv",
            )


def run_cli_or_streamlit() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--export-catalog", action="store_true")
    parser.add_argument("--catalog-output", default=str(RESOURCE_CATALOG_PATH))
    args, _ = parser.parse_known_args()

    ensure_directories()
    if args.export_catalog:
        output_path = Path(args.catalog_output)
        count = export_resource_catalog(output_path)
        print(f"Exported {count} catalog entries to {output_path}")
        return

    if st is None:
        print("Streamlit is not installed. Run 'pip install -r requirements.txt' and then 'streamlit run app.py'.")
        return
    render_streamlit_app()


if __name__ == "__main__":
    run_cli_or_streamlit()
