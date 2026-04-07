from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import shutil
import sys
import tempfile
import textwrap
import zipfile
from collections import defaultdict
from datetime import datetime
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any

try:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
except Exception:
    matplotlib = None
    cm = None
    plt = None

import numpy as np
import pandas as pd
import requests
try:
    import seaborn as sns
except Exception:
    sns = None
from PIL import Image
try:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
        roc_auc_score,
    )
    from sklearn.preprocessing import label_binarize
except Exception:
    accuracy_score = None
    confusion_matrix = None
    precision_recall_fscore_support = None
    roc_auc_score = None
    label_binarize = None

try:
    import streamlit as st
except Exception:
    st = None

try:
    import speech_recognition as sr
except Exception:
    sr = None

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except Exception:
    tf = None
    keras = None
    layers = None

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None


ROOT = Path(__file__).resolve().parent
DATASET_ROOT = ROOT / "dataset"
TRAIN_DIR = DATASET_ROOT / "train"
VAL_DIR = DATASET_ROOT / "val"
TEST_DIR = DATASET_ROOT / "test"
MODELS_ROOT = ROOT / "models"
MODEL_FILE = MODELS_ROOT / "cei_cnn_gap.keras"
MODEL_META_FILE = MODELS_ROOT / "cei_cnn_gap_metadata.json"
CATALOG_FILE = ROOT / "resource_catalog.csv"
TWIN_LOG_FILE = ROOT / "cei_twin_log.csv"
RECOMMENDER_FILE = ROOT / "recommender_stats.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PROJECT_EMOTIONS = [
    "happy",
    "sad",
    "angry",
    "anxious",
    "calm",
    "focused",
    "tired",
    "neutral",
]

EMOJI_MAP = {
    "None": "neutral",
    "😀": "happy",
    "😌": "calm",
    "😐": "neutral",
    "😓": "anxious",
    "😞": "sad",
    "😡": "angry",
    "🥱": "tired",
    "🤓": "focused",
}

TEXT_KEYWORDS = {
    "happy": {
        "happy",
        "joy",
        "joyful",
        "excited",
        "grateful",
        "smile",
        "celebrate",
        "good",
        "great",
        "awesome",
        "love",
        "fun",
        "success",
        "confident",
    },
    "sad": {
        "sad",
        "down",
        "upset",
        "cry",
        "hurt",
        "lonely",
        "miss",
        "gloomy",
        "broken",
        "disappointed",
        "empty",
    },
    "angry": {
        "angry",
        "mad",
        "annoyed",
        "frustrated",
        "hate",
        "furious",
        "irritated",
        "rage",
        "argument",
        "fight",
        "burnout",
    },
    "anxious": {
        "anxious",
        "stress",
        "stressed",
        "tense",
        "worry",
        "worried",
        "panic",
        "nervous",
        "deadline",
        "pressure",
        "afraid",
        "fear",
        "uncertain",
    },
    "calm": {
        "calm",
        "peace",
        "peaceful",
        "relaxed",
        "relax",
        "breathe",
        "quiet",
        "steady",
        "balanced",
        "mindful",
        "grounded",
    },
    "focused": {
        "focus",
        "focused",
        "study",
        "learn",
        "discipline",
        "plan",
        "goal",
        "project",
        "code",
        "exam",
        "productive",
        "work",
    },
    "tired": {
        "tired",
        "sleepy",
        "exhausted",
        "fatigue",
        "drained",
        "burned",
        "burnedout",
        "rest",
        "nap",
        "lazy",
        "weak",
    },
    "neutral": {
        "okay",
        "fine",
        "normal",
        "neutral",
        "average",
        "stable",
        "usual",
        "regular",
    },
}

QUESTION_BANK = {
    "happy": [
        "What contributed most to this positive mood today?",
        "How can you preserve this energy for the next few hours?",
        "Which task can you complete right now while your motivation is high?",
    ],
    "sad": [
        "What happened just before this feeling became stronger?",
        "Is there one small action that would make the next hour easier?",
        "Who is one person you could message for support or company?",
    ],
    "angry": [
        "What boundary or expectation feels violated right now?",
        "Would a short pause change the way you want to respond?",
        "What outcome do you actually want from this situation?",
    ],
    "anxious": [
        "Which part of the situation is real, and which part is only imagined?",
        "What is the smallest next step you can control in the next 10 minutes?",
        "What would you tell a friend who described this same concern?",
    ],
    "calm": [
        "What habit helped you reach this balanced state today?",
        "How can you use this calm period to prepare for a busy moment later?",
        "Which person or environment is supporting your stability right now?",
    ],
    "focused": [
        "What is the single most important task to finish first?",
        "What distraction should you remove before the next work session?",
        "How will you measure progress by the end of this session?",
    ],
    "tired": [
        "Is your tiredness physical, emotional, or both?",
        "What recovery action is realistic right now: water, stretch, food, or sleep?",
        "Which task can be postponed so you avoid low-quality work?",
    ],
    "neutral": [
        "What kind of support would help you shift from neutral to better?",
        "Would you benefit more right now from rest, movement, or focus?",
        "Which activity usually improves your mood without much effort?",
    ],
}

ACTION_BANK = {
    "happy": [
        "Use the positive momentum to finish one pending academic task.",
        "Capture what worked today in a short note for future repetition.",
        "Share the good moment with a teammate or family member.",
    ],
    "sad": [
        "Choose a low-friction task and complete it for momentum.",
        "Use slow breathing for two minutes before making decisions.",
        "Avoid isolating yourself for too long if support is available.",
    ],
    "angry": [
        "Pause for 90 seconds before replying to messages or emails.",
        "Switch to a physical reset: water, walking, stretching, or deep breathing.",
        "Write the issue privately before confronting anyone.",
    ],
    "anxious": [
        "Break the situation into one immediate action and one later action.",
        "Reduce information overload by closing extra tabs and notifications.",
        "Use a timer and focus only on the current micro-task.",
    ],
    "calm": [
        "Schedule a demanding task while your mind is steady.",
        "Maintain your current environment if it is helping you stay balanced.",
        "Use this state for revision, planning, or reflective journaling.",
    ],
    "focused": [
        "Work in a 25-minute sprint with one clearly defined target.",
        "Silence non-essential notifications during the current session.",
        "Record what you completed to strengthen future habit loops.",
    ],
    "tired": [
        "Reduce screen brightness and take a hydration break.",
        "Avoid starting cognitively heavy work without a recovery pause.",
        "Prioritize sleep hygiene if tiredness is recurring.",
    ],
    "neutral": [
        "Pick one purposeful action to avoid drifting into procrastination.",
        "Choose a background track or resource that matches your next task.",
        "Review your schedule and set one achievable target for the day.",
    ],
}

HF_DEFAULT_EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
HF_DEFAULT_GENERATION_MODEL = "google/flan-t5-base"


def ensure_runtime_dirs() -> None:
    DATASET_ROOT.mkdir(exist_ok=True)
    MODELS_ROOT.mkdir(exist_ok=True)
    for split_dir in (TRAIN_DIR, VAL_DIR, TEST_DIR):
        split_dir.mkdir(exist_ok=True)
    if not CATALOG_FILE.exists():
        export_catalog(CATALOG_FILE)


def empty_scores() -> dict[str, float]:
    return {emotion: 0.0 for emotion in PROJECT_EMOTIONS}


def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    filtered = {emotion: float(max(0.0, scores.get(emotion, 0.0))) for emotion in PROJECT_EMOTIONS}
    total = sum(filtered.values())
    if total <= 0:
        return {emotion: 1.0 / len(PROJECT_EMOTIONS) for emotion in PROJECT_EMOTIONS}
    return {emotion: value / total for emotion, value in filtered.items()}


def top_emotion(scores: dict[str, float]) -> tuple[str, float]:
    normalized = normalize_scores(scores)
    label = max(normalized, key=normalized.get)
    return label, normalized[label]


def emotion_alias(label: str) -> str:
    cleaned = re.sub(r"[^a-z]+", "", label.lower().strip())
    alias_map = {
        "anger": "angry",
        "angry": "angry",
        "disgust": "angry",
        "contempt": "angry",
        "annoyed": "angry",
        "happy": "happy",
        "happiness": "happy",
        "joy": "happy",
        "joyful": "happy",
        "sad": "sad",
        "sadness": "sad",
        "fear": "anxious",
        "afraid": "anxious",
        "anxiety": "anxious",
        "anxious": "anxious",
        "stress": "anxious",
        "stressed": "anxious",
        "calm": "calm",
        "relaxed": "calm",
        "peaceful": "calm",
        "focused": "focused",
        "concentrated": "focused",
        "surprise": "focused",
        "surprised": "focused",
        "tired": "tired",
        "sleepy": "tired",
        "fatigue": "tired",
        "neutral": "neutral",
        "normal": "neutral",
    }
    return alias_map.get(cleaned, "neutral")


def map_external_scores(raw_scores: dict[str, float]) -> dict[str, float]:
    merged = empty_scores()
    for label, value in raw_scores.items():
        merged[emotion_alias(label)] += float(value)
    return normalize_scores(merged)


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def lexicon_emotion_scores(text: str) -> dict[str, float]:
    tokens = tokenize(text)
    scores = empty_scores()
    if not tokens:
        return normalize_scores(scores)

    for token in tokens:
        for emotion, words in TEXT_KEYWORDS.items():
            if token in words:
                scores[emotion] += 1.0

    lowered = text.lower()
    if "!" in text:
        scores["happy"] += 0.3
        scores["angry"] += 0.2
    if any(phrase in lowered for phrase in ("can't sleep", "too much work", "overthinking", "what if")):
        scores["anxious"] += 1.2
    if any(phrase in lowered for phrase in ("need to finish", "must submit", "major project", "viva")):
        scores["focused"] += 1.0
    if any(phrase in lowered for phrase in ("need a break", "so exhausted", "very tired")):
        scores["tired"] += 1.0
    if not any(value > 0 for value in scores.values()):
        scores["neutral"] = 1.0

    return normalize_scores(scores)


def emoji_scores(emoji_value: str) -> dict[str, float]:
    scores = empty_scores()
    mapped = EMOJI_MAP.get(emoji_value, "neutral")
    scores[mapped] = 1.0
    return normalize_scores(scores)


def hf_headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def hf_post(model_id: str, payload: dict[str, Any], token: str, timeout: int = 90) -> Any:
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    response = requests.post(url, headers=hf_headers(token), json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def hf_emotion_scores(text: str, token: str, model_id: str = HF_DEFAULT_EMOTION_MODEL) -> tuple[dict[str, float] | None, str | None]:
    try:
        response = hf_post(
            model_id=model_id,
            payload={"inputs": text, "options": {"wait_for_model": True}},
            token=token,
            timeout=60,
        )
    except Exception as exc:
        return None, str(exc)

    if isinstance(response, dict) and response.get("error"):
        return None, response["error"]

    if isinstance(response, list) and response and isinstance(response[0], list):
        response = response[0]

    if not isinstance(response, list):
        return None, "Unexpected Hugging Face response for emotion analysis."

    raw_scores: dict[str, float] = {}
    for item in response:
        label = str(item.get("label", "")).lower()
        score = float(item.get("score", 0.0))
        raw_scores[label] = score

    return map_external_scores(raw_scores), None


def local_supportive_questions(emotion: str) -> list[str]:
    return QUESTION_BANK.get(emotion, QUESTION_BANK["neutral"])


def hf_supportive_questions(
    emotion: str,
    context: str,
    token: str,
    model_id: str = HF_DEFAULT_GENERATION_MODEL,
) -> tuple[list[str] | None, str | None]:
    prompt = textwrap.dedent(
        f"""
        Generate exactly three short reflective questions for a university student.
        Mood: {emotion}
        Context: {context or "No extra context provided."}
        Constraints:
        - supportive
        - practical
        - no diagnosis
        - one sentence each
        - plain text only
        """
    ).strip()
    try:
        response = hf_post(
            model_id=model_id,
            payload={
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 120,
                    "temperature": 0.6,
                    "return_full_text": False,
                },
                "options": {"wait_for_model": True},
            },
            token=token,
            timeout=90,
        )
    except Exception as exc:
        return None, str(exc)

    if isinstance(response, dict) and response.get("error"):
        return None, response["error"]

    generated_text = ""
    if isinstance(response, list) and response:
        first_item = response[0]
        if isinstance(first_item, dict):
            generated_text = str(first_item.get("generated_text", ""))
        elif isinstance(first_item, str):
            generated_text = first_item
    elif isinstance(response, dict):
        generated_text = str(response.get("generated_text", ""))

    questions = []
    for line in generated_text.splitlines():
        stripped = line.strip(" -0123456789.")
        if stripped:
            questions.append(stripped)

    if not questions and generated_text:
        questions = [segment.strip() for segment in re.split(r"[?]\s*", generated_text) if segment.strip()]
        questions = [f"{segment}?" for segment in questions[:3]]

    return questions[:3] if questions else None, None


def transcribe_audio(uploaded_file: Any) -> tuple[str, str | None]:
    if uploaded_file is None:
        return "", None
    if sr is None:
        return "", "SpeechRecognition is not installed. Use manual transcript text instead."

    suffix = Path(uploaded_file.name).suffix or ".wav"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            handle.write(uploaded_file.getvalue())
            temp_path = handle.name

        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_path) as source:
            audio_data = recognizer.record(source)
        transcript = recognizer.recognize_google(audio_data)
        return clean_text(transcript), None
    except Exception as exc:
        return "", f"Audio transcription fallback failed: {exc}"
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def heuristic_image_scores(image: Image.Image) -> dict[str, float]:
    sample = image.convert("RGB").resize((128, 128))
    array = np.asarray(sample, dtype=np.float32) / 255.0
    brightness = float(array.mean())
    contrast = float(array.std())
    channel_means = array.mean(axis=(0, 1))
    red, green, blue = [float(value) for value in channel_means]
    saturation = float((array.max(axis=2) - array.min(axis=2)).mean())

    scores = empty_scores()
    scores["happy"] = max(0.0, 0.35 * brightness + 0.35 * saturation + 0.15 * green)
    scores["sad"] = max(0.0, 0.50 * (1.0 - brightness) + 0.15 * blue)
    scores["angry"] = max(0.0, 0.50 * red + 0.20 * saturation - 0.10 * blue)
    scores["anxious"] = max(0.0, 0.30 * contrast + 0.20 * abs(red - blue))
    scores["calm"] = max(0.0, 0.30 * blue + 0.25 * (1.0 - saturation) + 0.10 * brightness)
    scores["focused"] = max(0.0, 0.35 * contrast + 0.10 * brightness)
    scores["tired"] = max(0.0, 0.20 * (1.0 - contrast) + 0.25 * (1.0 - brightness))
    scores["neutral"] = 0.15
    return normalize_scores(scores)


@lru_cache(maxsize=1)
def load_trained_model_bundle() -> tuple[Any | None, dict[str, Any]]:
    if tf is None or not MODEL_FILE.exists():
        return None, {}
    model = keras.models.load_model(MODEL_FILE)
    metadata = {}
    if MODEL_META_FILE.exists():
        metadata = json.loads(MODEL_META_FILE.read_text(encoding="utf-8"))
    return model, metadata


def predict_image_emotion(image: Image.Image) -> tuple[dict[str, float], str]:
    model, metadata = load_trained_model_bundle()
    if model is None:
        return heuristic_image_scores(image), "Heuristic visual fallback (no trained CNN model found)."

    image_size = int(metadata.get("image_size", 128))
    class_names = metadata.get("class_names", PROJECT_EMOTIONS)
    array = np.asarray(image.convert("RGB").resize((image_size, image_size)), dtype=np.float32)
    batch = np.expand_dims(array, axis=0)
    probabilities = model.predict(batch, verbose=0)[0]
    raw_scores = {str(label): float(score) for label, score in zip(class_names, probabilities)}
    return map_external_scores(raw_scores), "Custom CNN + GAP prediction."


def fuse_modalities(modality_results: list[dict[str, Any]], weights: dict[str, float]) -> dict[str, Any]:
    fused = empty_scores()
    contributions: list[str] = []
    active_modalities = []

    for result in modality_results:
        name = result["name"]
        scores = normalize_scores(result["scores"])
        confidence = float(result.get("confidence", 0.5))
        effective_weight = float(weights.get(name, 1.0)) * max(0.2, confidence)
        if effective_weight <= 0:
            continue
        active_modalities.append(name)
        label, label_score = top_emotion(scores)
        contributions.append(f"{name}: {label} ({label_score:.2f})")
        for emotion, value in scores.items():
            fused[emotion] += value * effective_weight

    fused = normalize_scores(fused)
    dominant, confidence = top_emotion(fused)
    return {
        "scores": fused,
        "dominant_emotion": dominant,
        "confidence": confidence,
        "active_modalities": active_modalities,
        "contributions": contributions,
    }


def ethical_ai_checks(fusion_result: dict[str, Any]) -> list[str]:
    flags = []
    active_count = len(fusion_result.get("active_modalities", []))
    confidence = float(fusion_result.get("confidence", 0.0))
    emotion = fusion_result.get("dominant_emotion", "neutral")

    if active_count < 2:
        flags.append("Only one modality influenced this decision, so confidence should be treated carefully.")
    if confidence < 0.45:
        flags.append("Low-confidence prediction detected. Ask the user to confirm or override the mood manually.")
    if emotion in {"sad", "anxious", "angry", "tired"}:
        flags.append("This output is supportive, not clinical. It should not be used for diagnosis or emergency decision-making.")
    flags.append("Store only minimal logs and avoid collecting emotion data without informed consent.")
    return flags


def action_plan_for_emotion(emotion: str) -> list[str]:
    return ACTION_BANK.get(emotion, ACTION_BANK["neutral"])


def build_resource_catalog() -> pd.DataFrame:
    mood_queries = {
        "happy": [
            "upbeat focus playlist",
            "celebration pop mix",
            "feel good study music",
            "confidence affirmations",
            "joy journaling prompts",
        ],
        "sad": [
            "soft healing piano",
            "gentle reflective music",
            "self compassion meditation",
            "comfort acoustic session",
            "uplifting recovery playlist",
        ],
        "angry": [
            "calming breathing audio",
            "stress release instrumental",
            "boxing workout without lyrics",
            "de escalation reflection prompts",
            "focus reset ambient mix",
        ],
        "anxious": [
            "deep breathing audio",
            "anxiety relief instrumental",
            "grounding meditation",
            "exam stress focus music",
            "slow heartbeat relaxation",
        ],
        "calm": [
            "ambient coding music",
            "mindful productivity playlist",
            "nature rain sounds",
            "minimal piano concentration",
            "gratitude reflection session",
        ],
        "focused": [
            "deep work instrumental",
            "pomodoro focus beats",
            "coding flow music",
            "exam concentration mix",
            "productive lofi session",
        ],
        "tired": [
            "light energizing playlist",
            "gentle stretch guidance",
            "morning motivation audio",
            "hydration and recovery routine",
            "soft wake up instrumental",
        ],
        "neutral": [
            "balanced mood playlist",
            "light focus instrumental",
            "daily planning guidance",
            "calm and clarity mix",
            "steady background study music",
        ],
    }

    sources = [
        ("YouTube Search", "youtube", "https://www.youtube.com/results?search_query={query}"),
        ("YouTube Music Search", "youtube_music", "https://music.youtube.com/search?q={query}"),
        ("Spotify Search", "spotify", "https://open.spotify.com/search/{query}"),
    ]

    rows = []
    for mood, queries in mood_queries.items():
        offline_hint = ACTION_BANK[mood][0]
        for query in queries:
            for source_name, source_type, template in sources:
                encoded_query = requests.utils.quote(query)
                rows.append(
                    {
                        "mood": mood,
                        "title": query.title(),
                        "url": template.format(query=encoded_query),
                        "source": source_name,
                        "type": source_type,
                        "offline_fallback_guidance": offline_hint,
                    }
                )
    return pd.DataFrame(rows)


def export_catalog(path: Path) -> pd.DataFrame:
    catalog = build_resource_catalog()
    catalog.to_csv(path, index=False)
    return catalog


def load_catalog() -> pd.DataFrame:
    if CATALOG_FILE.exists():
        return pd.read_csv(CATALOG_FILE)
    return export_catalog(CATALOG_FILE)


def read_recommender_history() -> pd.DataFrame:
    if RECOMMENDER_FILE.exists():
        return pd.read_csv(RECOMMENDER_FILE)
    return pd.DataFrame(
        columns=["timestamp", "emotion", "source", "title", "url", "reward", "event"]
    )


def recommend_resources(emotion: str, top_n: int = 6) -> pd.DataFrame:
    catalog = load_catalog().copy()
    history = read_recommender_history()
    recent_titles = set(history.tail(15)["title"].tolist()) if not history.empty else set()

    emotion_rows = catalog[catalog["mood"] == emotion].copy()
    if emotion_rows.empty:
        emotion_rows = catalog.copy()

    emotion_rows["score"] = 1.0
    if not history.empty:
        liked = history[(history["emotion"] == emotion) & (history["event"] == "feedback")]
        if not liked.empty:
            source_reward = liked.groupby("source")["reward"].mean().to_dict()
            emotion_rows["score"] += emotion_rows["source"].map(source_reward).fillna(0.0) / 5.0

    emotion_rows["score"] -= emotion_rows["title"].isin(recent_titles).astype(float) * 0.4
    emotion_rows = emotion_rows.sort_values(by=["score", "source", "title"], ascending=[False, True, True])
    return emotion_rows.head(top_n).drop(columns=["score"])


def log_digital_twin(row: dict[str, Any]) -> None:
    fieldnames = [
        "timestamp",
        "user_name",
        "context",
        "transcript",
        "emoji",
        "dominant_emotion",
        "confidence",
        "active_modalities",
        "scores_json",
        "ethical_flags",
    ]
    write_csv_row(TWIN_LOG_FILE, fieldnames, row)


def log_recommender_event(row: dict[str, Any]) -> None:
    fieldnames = ["timestamp", "emotion", "source", "title", "url", "reward", "event"]
    write_csv_row(RECOMMENDER_FILE, fieldnames, row)


def write_csv_row(path: Path, fieldnames: list[str], row: dict[str, Any]) -> None:
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fieldnames})


def dataset_status() -> dict[str, Any]:
    summary = {}
    for split_name, split_dir in (("train", TRAIN_DIR), ("val", VAL_DIR), ("test", TEST_DIR)):
        counts = {}
        if split_dir.exists():
            for class_dir in sorted([path for path in split_dir.iterdir() if path.is_dir()]):
                counts[class_dir.name] = len(
                    [file for file in class_dir.rglob("*") if file.suffix.lower() in IMAGE_EXTENSIONS]
                )
        summary[split_name] = counts
    return summary


def clear_dataset_dirs() -> None:
    for split_dir in (TRAIN_DIR, VAL_DIR, TEST_DIR):
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)


def write_split_records(records: list[tuple[str, Path]], seed: int = 42) -> dict[str, int]:
    clear_dataset_dirs()
    grouped: dict[str, list[Path]] = defaultdict(list)
    for label, image_path in records:
        grouped[emotion_alias(label)].append(image_path)

    random.seed(seed)
    total_written = {"train": 0, "val": 0, "test": 0}

    for label, files in grouped.items():
        files = files[:]
        random.shuffle(files)
        total = len(files)
        if total == 1:
            train_files, val_files, test_files = files, [], []
        else:
            train_cutoff = max(1, int(total * 0.7))
            val_cutoff = max(train_cutoff + 1, int(total * 0.85))
            train_files = files[:train_cutoff]
            val_files = files[train_cutoff:val_cutoff]
            test_files = files[val_cutoff:]
            if not test_files and val_files:
                test_files = [val_files.pop()]

        for split_name, split_files, split_dir in (
            ("train", train_files, TRAIN_DIR),
            ("val", val_files, VAL_DIR),
            ("test", test_files, TEST_DIR),
        ):
            target_dir = split_dir / label
            target_dir.mkdir(parents=True, exist_ok=True)
            for index, source_file in enumerate(split_files):
                extension = source_file.suffix.lower() or ".jpg"
                target_file = target_dir / f"{label}_{index:04d}{extension}"
                shutil.copy2(source_file, target_file)
                total_written[split_name] += 1

    return total_written


def import_zip_dataset(uploaded_file: Any, sample_limit: int = 900) -> dict[str, int]:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        with zipfile.ZipFile(BytesIO(uploaded_file.getvalue())) as archive:
            archive.extractall(temp_path)

        records = []
        for file_path in temp_path.rglob("*"):
            if file_path.suffix.lower() in IMAGE_EXTENSIONS:
                label = emotion_alias(file_path.parent.name)
                records.append((label, file_path))

        if not records:
            raise ValueError("No class-wise images were found in the ZIP file.")

        random.shuffle(records)
        records = records[:sample_limit]
        return write_split_records(records)


def infer_hf_columns(dataset: Any) -> tuple[str | None, str | None]:
    image_column = None
    label_column = None
    for column_name, feature in dataset.features.items():
        feature_name = type(feature).__name__.lower()
        lowered = column_name.lower()
        if image_column is None and ("image" in lowered or feature_name == "image"):
            image_column = column_name
        if label_column is None and any(key in lowered for key in ("label", "emotion", "class")):
            label_column = column_name
    return image_column, label_column


def resolve_hf_label(dataset: Any, label_column: str, raw_value: Any) -> str:
    feature = dataset.features[label_column]
    if hasattr(feature, "names") and isinstance(raw_value, int):
        return emotion_alias(feature.names[raw_value])
    return emotion_alias(str(raw_value))


def prepare_hf_image_dataset(
    dataset_id: str,
    split_name: str,
    image_column: str | None,
    label_column: str | None,
    sample_limit: int = 900,
) -> dict[str, int]:
    if load_dataset is None:
        raise RuntimeError("Install the datasets package to enable Hugging Face dataset preparation.")

    dataset = load_dataset(dataset_id, split=split_name)
    auto_image_column, auto_label_column = infer_hf_columns(dataset)
    image_column = image_column or auto_image_column
    label_column = label_column or auto_label_column
    if not image_column or not label_column:
        raise ValueError("Unable to infer image and label columns. Please provide both manually.")

    sample_count = min(sample_limit, len(dataset))
    dataset = dataset.shuffle(seed=42).select(range(sample_count))

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        records = []
        for index, item in enumerate(dataset):
            image_obj = item[image_column]
            if hasattr(image_obj, "convert"):
                image = image_obj.convert("RGB")
            elif isinstance(image_obj, dict) and "bytes" in image_obj:
                image = Image.open(BytesIO(image_obj["bytes"])).convert("RGB")
            else:
                continue

            label = resolve_hf_label(dataset, label_column, item[label_column])
            save_path = temp_root / f"{label}_{index:04d}.jpg"
            image.save(save_path, quality=95)
            records.append((label, save_path))

        if not records:
            raise ValueError("No valid images were extracted from the Hugging Face dataset.")
        return write_split_records(records)


def build_gap_cnn(input_shape: tuple[int, int, int], num_classes: int) -> Any:
    inputs = keras.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 255.0)(inputs)
    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.15)(x)
    x = layers.Conv2D(256, 3, padding="same", activation="relu", name="last_conv")(x)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = layers.Dropout(0.30)(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="cei_gap_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=8e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_directory_dataset(split_dir: Path, image_size: int, batch_size: int, shuffle: bool) -> Any:
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing dataset split: {split_dir}")
    return tf.keras.utils.image_dataset_from_directory(
        split_dir,
        label_mode="int",
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=shuffle,
        seed=42,
    )


def train_cnn_model(image_size: int = 128, batch_size: int = 8, epochs: int = 3) -> dict[str, Any]:
    if tf is None:
        raise RuntimeError("TensorFlow is not installed. Training is unavailable.")

    train_ds = load_directory_dataset(TRAIN_DIR, image_size=image_size, batch_size=batch_size, shuffle=True)
    val_ds = load_directory_dataset(VAL_DIR, image_size=image_size, batch_size=batch_size, shuffle=False)

    class_names = train_ds.class_names
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)

    model = build_gap_cnn((image_size, image_size, 3), len(class_names))
    callbacks = [
        keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True, monitor="val_loss"),
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=0)

    MODELS_ROOT.mkdir(exist_ok=True)
    model.save(MODEL_FILE)
    MODEL_META_FILE.write_text(
        json.dumps(
            {
                "class_names": class_names,
                "image_size": image_size,
                "saved_at": datetime.utcnow().isoformat(),
                "architecture": "Custom CNN with Global Average Pooling and Grad-CAM support",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    load_trained_model_bundle.cache_clear()
    return {"history": history.history, "class_names": class_names}


def evaluate_saved_model(batch_size: int = 8) -> dict[str, Any]:
    if tf is None:
        raise RuntimeError("TensorFlow is not installed. Evaluation is unavailable.")
    if any(item is None for item in (accuracy_score, confusion_matrix, precision_recall_fscore_support)):
        raise RuntimeError("scikit-learn is required for evaluation metrics.")

    model, metadata = load_trained_model_bundle()
    if model is None:
        raise RuntimeError("No saved model is available for evaluation.")

    image_size = int(metadata.get("image_size", 128))
    class_names = metadata.get("class_names", PROJECT_EMOTIONS)
    test_ds = load_directory_dataset(TEST_DIR, image_size=image_size, batch_size=batch_size, shuffle=False)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    y_true = []
    y_prob = []
    for batch_images, batch_labels in test_ds:
        probabilities = model.predict(batch_images, verbose=0)
        y_true.extend(batch_labels.numpy().tolist())
        y_prob.extend(probabilities.tolist())

    y_true_array = np.array(y_true)
    y_prob_array = np.array(y_prob)
    y_pred_array = np.argmax(y_prob_array, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_array,
        y_pred_array,
        average="weighted",
        zero_division=0,
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true_array, y_pred_array)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

    try:
        y_true_bin = label_binarize(y_true_array, classes=list(range(len(class_names))))
        metrics["roc_auc_ovr"] = float(
            roc_auc_score(y_true_bin, y_prob_array, multi_class="ovr", average="weighted")
        )
    except Exception:
        metrics["roc_auc_ovr"] = float("nan")

    matrix = confusion_matrix(y_true_array, y_pred_array)
    matrix_df = pd.DataFrame(matrix, index=class_names, columns=class_names)
    return {
        "metrics": metrics,
        "confusion_matrix": matrix_df,
        "class_names": class_names,
        "y_prob": y_prob_array,
        "y_true": y_true_array,
    }


def preprocess_image_for_model(image: Image.Image, image_size: int) -> np.ndarray:
    array = np.asarray(image.convert("RGB").resize((image_size, image_size)), dtype=np.float32)
    return np.expand_dims(array, axis=0)


def build_gradcam_overlay(image: Image.Image) -> tuple[Image.Image, str]:
    if tf is None:
        raise RuntimeError("TensorFlow is not installed. Grad-CAM is unavailable.")
    if cm is None:
        raise RuntimeError("matplotlib is required for Grad-CAM visualization.")

    model, metadata = load_trained_model_bundle()
    if model is None:
        raise RuntimeError("Train the CNN model before running Grad-CAM.")

    image_size = int(metadata.get("image_size", 128))
    class_names = metadata.get("class_names", PROJECT_EMOTIONS)
    batch = preprocess_image_for_model(image, image_size)

    grad_model = keras.models.Model(
        [model.inputs],
        [model.get_layer("last_conv").output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(batch)
        pred_index = int(tf.argmax(predictions[0]))
        loss = predictions[:, pred_index]

    gradients = tape.gradient(loss, conv_outputs)
    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_gradients, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-9)
    heatmap = heatmap.numpy()

    rescaled = Image.fromarray(np.uint8(heatmap * 255.0)).resize(image.size)
    colored = cm.jet(np.asarray(rescaled, dtype=np.float32) / 255.0)[:, :, :3]
    colored_image = Image.fromarray(np.uint8(colored * 255))
    overlay = Image.blend(image.convert("RGB"), colored_image, alpha=0.35)
    return overlay, class_names[pred_index]


def history_figure(history: dict[str, list[float]]) -> Any:
    if plt is None:
        raise RuntimeError("matplotlib is required to plot training history.")
    figure, axis = plt.subplots(figsize=(7, 4))
    if "accuracy" in history:
        axis.plot(history["accuracy"], label="Train accuracy")
    if "val_accuracy" in history:
        axis.plot(history["val_accuracy"], label="Validation accuracy")
    if "loss" in history:
        axis.plot(history["loss"], label="Train loss")
    if "val_loss" in history:
        axis.plot(history["val_loss"], label="Validation loss")
    axis.set_title("Training history")
    axis.set_xlabel("Epoch")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    return figure


def confusion_matrix_figure(matrix_df: pd.DataFrame) -> Any:
    if plt is None or sns is None:
        raise RuntimeError("matplotlib and seaborn are required to plot the confusion matrix.")
    figure, axis = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix_df, annot=True, fmt="d", cmap="Blues", ax=axis)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title("Confusion matrix")
    figure.tight_layout()
    return figure


def flowchart_text() -> str:
    return textwrap.dedent(
        """
        User Input
            |
            +--> Free text context
            +--> Emoji self-check
            +--> Voice note -> transcription fallback
            +--> Face image
                    |
        Preprocessing + Per-modality scoring
                    |
        Multimodal fusion engine
                    |
        Ethical AI monitor + confidence check
                    |
        Digital Emotional Twin logger
                    |
        Adaptive output layer
            +--> reflective questions
            +--> music/resource links
            +--> action plan
            +--> feedback reward update
        """
    ).strip()


def viva_items() -> list[dict[str, str]]:
    return [
        {
            "question": "What is the main protagonist of this major project?",
            "answer": (
                "The protagonist is the Cognitive Emotion Intelligence and Adaptive Lifestyle System itself. "
                "Instead of acting like a simple music player or a static chatbot, it works as a human-centered "
                "decision support layer that senses mood, explains why a mood was inferred, recommends safe next "
                "actions, and learns from feedback through a lightweight reward loop."
            ),
        },
        {
            "question": "How is this project different from earlier 2023-2025 emotion-based systems?",
            "answer": (
                "Many earlier systems depended on only one modality, mostly text or face images. This project "
                "combines text, emoji, optional voice transcript, and visual input, then fuses them with "
                "confidence-aware weighting. It also adds explainable AI, ethical monitoring, CSV-based digital "
                "twin logging, and adaptive recommendation updates instead of a one-shot static output."
            ),
        },
        {
            "question": "Why did you choose a custom CNN with Global Average Pooling instead of MobileNetV2?",
            "answer": (
                "MobileNetV2 is still useful, but this project intentionally demonstrates a transparent, compact, "
                "from-scratch CNN tailored for academic explainability and low-resource retraining. Global Average "
                "Pooling reduces parameter count, lowers overfitting risk, keeps the architecture lightweight, and "
                "supports cleaner Grad-CAM heatmaps because the last convolutional features remain directly linked "
                "to class activation."
            ),
        },
        {
            "question": "What is the significance of Grad-CAM in this system?",
            "answer": (
                "Grad-CAM provides visual evidence for model attention. In emotion recognition, that matters because "
                "a facial classifier should focus on meaningful facial regions instead of background noise. It helps "
                "during debugging, model auditing, viva explanation, and trust building by showing why the predicted "
                "emotion was produced."
            ),
        },
        {
            "question": "What is Global Average Pooling and why is it useful here?",
            "answer": (
                "Global Average Pooling converts each final convolutional feature map into a single representative "
                "value by averaging across spatial dimensions. That removes the need for a large flatten layer, "
                "reduces trainable parameters, improves generalization on small datasets, and keeps the network "
                "lighter for limited hardware such as an Intel i5 system with 4 GB RAM."
            ),
        },
        {
            "question": "What do you mean by Digital Emotional Twin?",
            "answer": (
                "A Digital Emotional Twin is a lightweight evolving record of a user's recent emotional interactions. "
                "In this project it is represented through CSV logs that capture detected mood, confidence, active "
                "modalities, and feedback. This allows trend analysis, adaptive recommendations, and traceability "
                "without requiring a heavy database."
            ),
        },
        {
            "question": "Where is the reinforcement learning logic in the project?",
            "answer": (
                "The project uses a practical reward-update loop rather than a heavy deep RL setup. When users rate "
                "recommended resources, the system updates future ranking preferences by emotion and source. This is "
                "a lightweight policy adaptation mechanism suitable for a major project and easy to explain in a viva."
            ),
        },
        {
            "question": "Why is Ethical AI monitoring important for emotion intelligence systems?",
            "answer": (
                "Emotion data is sensitive. Ethical AI monitoring ensures the system reports low-confidence cases, "
                "avoids overclaiming, reminds the user that outputs are supportive rather than diagnostic, and "
                "encourages minimal data retention. This is critical because emotion recognition can be biased, "
                "misinterpreted, or misused if not governed carefully."
            ),
        },
        {
            "question": "What real-world problem does this project solve?",
            "answer": (
                "The project addresses the gap between raw emotion detection and practical daily support. Students, "
                "professionals, and wellness users often need adaptive suggestions aligned with mood, workload, and "
                "attention level. The system converts multimodal affect signals into explainable recommendations, "
                "reflection prompts, and safer lifestyle guidance."
            ),
        },
        {
            "question": "Why is Hugging Face used as the free API option?",
            "answer": (
                "OpenAI and Spotify developer integrations may require billing or more restrictive quotas. Hugging Face "
                "offers a large ecosystem of public models and a simple token-based workflow. In this project, Hugging "
                "Face is optional: if the token or model is unavailable, the system still works with local rule-based "
                "emotion scoring and template-based reflective questions."
            ),
        },
    ]


def render_home_page() -> None:
    st.title("Cognitive Emotion Intelligence & Adaptive Lifestyle System")
    st.caption("Single-file Streamlit major project with multimodal mood understanding, explainable AI, and adaptive recommendation support.")

    st.subheader("Project highlights")
    st.markdown(
        """
        - Single-source-code Python application in `app.py`
        - Multimodal mood intake: text, emoji, voice transcript fallback, and face image
        - Free-first workflow with optional Hugging Face token support
        - Custom CNN with Global Average Pooling and Grad-CAM support
        - CSV-based Digital Emotional Twin logging
        - No MP3 downloads; recommendations are link-based and VS Code friendly
        - Lightweight feedback loop for adaptive recommendation ranking
        """
    )

    st.subheader("Method flow")
    st.code(flowchart_text())

    st.subheader("Important compatibility note")
    st.info(
        "For Windows 11 with 4 GB RAM, keep sample datasets small, use batch size 4 or 8, and prefer Python 3.10 or 3.11 for TensorFlow compatibility."
    )

    status = dataset_status()
    st.subheader("Dataset status")
    st.json(status)


def render_analysis_page(hf_token: str, hf_emotion_model: str, hf_generation_model: str) -> None:
    st.header("Multimodal mood analysis and adaptive support")
    st.write("Use any combination of text, emoji, audio, and image. The app will fuse available modalities and generate supportive outputs.")

    with st.form("analysis_form"):
        user_name = st.text_input("User name", value="Demo Student")
        context = st.text_area(
            "Free-text context",
            placeholder="Example: I have my major project viva next week and I feel stressed but I still want to stay productive.",
            height=140,
        )
        emoji_value = st.selectbox("Emoji self-check", list(EMOJI_MAP.keys()), index=2)
        image_file = st.file_uploader("Optional face image", type=["jpg", "jpeg", "png"], key="image_input")
        audio_file = st.file_uploader("Optional voice note", type=["wav", "flac", "aiff", "aif"], key="audio_input")

        st.markdown("**Fusion weights**")
        text_weight = st.slider("Text weight", 0.2, 2.0, 1.0, 0.1)
        emoji_weight = st.slider("Emoji weight", 0.2, 2.0, 0.8, 0.1)
        audio_weight = st.slider("Audio transcript weight", 0.2, 2.0, 1.0, 0.1)
        image_weight = st.slider("Image weight", 0.2, 2.0, 1.0, 0.1)

        submitted = st.form_submit_button("Analyse and generate support")

    if not submitted:
        return

    modality_results = []
    transcript = ""
    transcription_note = None
    full_context = clean_text(context)

    if context.strip():
        text_scores = lexicon_emotion_scores(context)
        text_source = "Local lexicon"
        if hf_token.strip():
            hf_scores, hf_error = hf_emotion_scores(context, hf_token.strip(), hf_emotion_model)
            if hf_scores is not None:
                text_scores = hf_scores
                text_source = "Hugging Face emotion model"
            elif hf_error:
                st.warning(f"Hugging Face emotion inference fallback used: {hf_error}")
        text_label, text_conf = top_emotion(text_scores)
        modality_results.append(
            {
                "name": "text",
                "scores": text_scores,
                "confidence": text_conf,
                "note": text_source,
                "label": text_label,
            }
        )

    if emoji_value != "None":
        scores = emoji_scores(emoji_value)
        label, confidence = top_emotion(scores)
        modality_results.append(
            {
                "name": "emoji",
                "scores": scores,
                "confidence": confidence,
                "note": "Self-reported emoji prior",
                "label": label,
            }
        )

    if audio_file is not None:
        transcript, transcription_note = transcribe_audio(audio_file)
        if transcript:
            audio_scores = lexicon_emotion_scores(transcript)
            audio_label, audio_conf = top_emotion(audio_scores)
            modality_results.append(
                {
                    "name": "audio",
                    "scores": audio_scores,
                    "confidence": audio_conf,
                    "note": "SpeechRecognition transcript -> lexicon scoring",
                    "label": audio_label,
                }
            )
            if full_context:
                full_context = f"{full_context} {transcript}"
            else:
                full_context = transcript

    image = None
    if image_file is not None:
        image = Image.open(image_file).convert("RGB")
        image_scores, image_note = predict_image_emotion(image)
        image_label, image_conf = top_emotion(image_scores)
        modality_results.append(
            {
                "name": "image",
                "scores": image_scores,
                "confidence": image_conf,
                "note": image_note,
                "label": image_label,
            }
        )

    if not modality_results:
        st.error("Provide at least one input modality before analysis.")
        return

    weights = {
        "text": text_weight,
        "emoji": emoji_weight,
        "audio": audio_weight,
        "image": image_weight,
    }
    fusion_result = fuse_modalities(modality_results, weights)
    flags = ethical_ai_checks(fusion_result)

    questions = local_supportive_questions(fusion_result["dominant_emotion"])
    if hf_token.strip():
        hf_questions, hf_error = hf_supportive_questions(
            fusion_result["dominant_emotion"],
            full_context,
            hf_token.strip(),
            hf_generation_model,
        )
        if hf_questions:
            questions = hf_questions
        elif hf_error:
            st.warning(f"Hugging Face question generation fallback used: {hf_error}")

    recommendations = recommend_resources(fusion_result["dominant_emotion"], top_n=6)
    actions = action_plan_for_emotion(fusion_result["dominant_emotion"])

    st.subheader("Detected mood")
    col1, col2, col3 = st.columns(3)
    col1.metric("Dominant emotion", fusion_result["dominant_emotion"].title())
    col2.metric("Confidence", f"{fusion_result['confidence']:.2f}")
    col3.metric("Modalities used", len(fusion_result["active_modalities"]))

    score_df = pd.DataFrame(
        {"emotion": list(fusion_result["scores"].keys()), "score": list(fusion_result["scores"].values())}
    )
    st.bar_chart(score_df.set_index("emotion"))

    st.subheader("Per-modality interpretation")
    for result in modality_results:
        st.markdown(
            f"- **{result['name'].title()}** -> {result['label'].title()} ({result['confidence']:.2f}) | {result['note']}"
        )

    if transcript:
        st.subheader("Voice transcript fallback")
        st.write(transcript)
    elif transcription_note:
        st.info(transcription_note)

    if image is not None:
        st.subheader("Uploaded face image")
        st.image(image, width=240)

    st.subheader("Reflective questions")
    for item in questions:
        st.write(f"- {item}")

    st.subheader("Adaptive action plan")
    for action in actions:
        st.write(f"- {action}")

    st.subheader("Recommended resources")
    st.dataframe(recommendations, use_container_width=True)
    for _, row in recommendations.iterrows():
        log_recommender_event(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "emotion": fusion_result["dominant_emotion"],
                "source": row["source"],
                "title": row["title"],
                "url": row["url"],
                "reward": "",
                "event": "shown",
            }
        )

    st.subheader("Ethical AI monitoring")
    for flag in flags:
        st.warning(flag)

    log_digital_twin(
        {
            "timestamp": datetime.utcnow().isoformat(),
            "user_name": user_name,
            "context": context,
            "transcript": transcript,
            "emoji": emoji_value,
            "dominant_emotion": fusion_result["dominant_emotion"],
            "confidence": round(fusion_result["confidence"], 4),
            "active_modalities": ",".join(fusion_result["active_modalities"]),
            "scores_json": json.dumps(fusion_result["scores"]),
            "ethical_flags": " | ".join(flags),
        }
    )

    st.subheader("Feedback reward update")
    resource_titles = recommendations["title"].tolist()
    with st.form("feedback_form"):
        chosen_title = st.selectbox("Which recommended resource felt most useful?", resource_titles)
        reward = st.slider("Usefulness rating", 1, 5, 4)
        save_feedback = st.form_submit_button("Save feedback")
    if save_feedback:
        chosen_row = recommendations[recommendations["title"] == chosen_title].iloc[0]
        log_recommender_event(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "emotion": fusion_result["dominant_emotion"],
                "source": chosen_row["source"],
                "title": chosen_row["title"],
                "url": chosen_row["url"],
                "reward": reward,
                "event": "feedback",
            }
        )
        st.success("Feedback saved. Future recommendations will adapt to this reward signal.")


def render_dataset_page() -> None:
    st.header("Dataset tools")
    st.write("Prepare a small, class-wise image dataset for CNN training. This is designed for sample mode on low-RAM systems.")

    st.subheader("Option A: Import a class-wise ZIP dataset")
    zip_file = st.file_uploader(
        "Upload ZIP with folders such as happy/, sad/, angry/, neutral/",
        type=["zip"],
        key="zip_dataset",
    )
    zip_limit = st.slider("ZIP sample limit", 100, 1500, 900, 50)
    if st.button("Prepare dataset from ZIP") and zip_file is not None:
        result = import_zip_dataset(zip_file, sample_limit=zip_limit)
        st.success(f"Dataset prepared. Split sizes: {result}")

    st.subheader("Option B: Download a public Hugging Face image dataset sample")
    hf_dataset_id = st.text_input("Hugging Face dataset id", value="")
    hf_split = st.text_input("Split name", value="train")
    hf_image_column = st.text_input("Image column (leave blank to infer)", value="")
    hf_label_column = st.text_input("Label column (leave blank to infer)", value="")
    hf_limit = st.slider("HF sample limit", 100, 1500, 600, 50)
    if st.button("Prepare dataset from Hugging Face") and hf_dataset_id.strip():
        try:
            result = prepare_hf_image_dataset(
                dataset_id=hf_dataset_id.strip(),
                split_name=hf_split.strip() or "train",
                image_column=hf_image_column.strip() or None,
                label_column=hf_label_column.strip() or None,
                sample_limit=hf_limit,
            )
            st.success(f"Hugging Face dataset prepared. Split sizes: {result}")
        except Exception as exc:
            st.error(str(exc))

    st.subheader("Current dataset structure")
    st.json(dataset_status())


def render_training_page() -> None:
    st.header("Training, evaluation, and Grad-CAM")
    st.write("Train a lightweight custom CNN with Global Average Pooling. Keep the settings small for Windows systems with limited RAM.")

    if tf is None:
        st.error("TensorFlow is not installed in the current environment. Install the requirements and restart the app.")
        return

    summary = dataset_status()
    total_train = sum(summary["train"].values())
    if total_train == 0:
        st.info("No training dataset found yet. Use the Dataset Tools page first.")
        return

    with st.form("training_form"):
        image_size = st.select_slider("Image size", options=[96, 112, 128, 160, 192], value=128)
        batch_size = st.select_slider("Batch size", options=[4, 8, 12, 16], value=8)
        epochs = st.slider("Epochs", 1, 5, 2)
        run_training = st.form_submit_button("Train model")

    if run_training:
        try:
            result = train_cnn_model(image_size=image_size, batch_size=batch_size, epochs=epochs)
            st.success("Training completed and model saved under models/.")
            st.pyplot(history_figure(result["history"]))
        except Exception as exc:
            st.error(str(exc))

    if MODEL_FILE.exists():
        st.subheader("Saved model metadata")
        if MODEL_META_FILE.exists():
            st.json(json.loads(MODEL_META_FILE.read_text(encoding="utf-8")))

        if st.button("Run evaluation on test split"):
            try:
                result = evaluate_saved_model(batch_size=8)
                metric_df = pd.DataFrame([result["metrics"]]).T.reset_index()
                metric_df.columns = ["metric", "value"]
                st.dataframe(metric_df, use_container_width=True)
                st.pyplot(confusion_matrix_figure(result["confusion_matrix"]))
            except Exception as exc:
                st.error(str(exc))

        st.subheader("Grad-CAM explainability")
        gradcam_image = st.file_uploader("Upload an image for Grad-CAM", type=["jpg", "jpeg", "png"], key="gradcam_input")
        if st.button("Generate Grad-CAM") and gradcam_image is not None:
            try:
                uploaded_image = Image.open(gradcam_image).convert("RGB")
                overlay, predicted_label = build_gradcam_overlay(uploaded_image)
                st.write(f"Predicted class: **{predicted_label.title()}**")
                col1, col2 = st.columns(2)
                col1.image(uploaded_image, caption="Original image", use_column_width=True)
                col2.image(overlay, caption="Grad-CAM overlay", use_column_width=True)
            except Exception as exc:
                st.error(str(exc))


def render_logs_page() -> None:
    st.header("Logs and exports")

    st.subheader("Resource catalog")
    catalog = load_catalog()
    st.dataframe(catalog.head(20), use_container_width=True)
    st.download_button(
        "Download resource_catalog.csv",
        data=catalog.to_csv(index=False).encode("utf-8"),
        file_name="resource_catalog.csv",
        mime="text/csv",
    )

    st.subheader("Digital Emotional Twin log")
    if TWIN_LOG_FILE.exists():
        twin_df = pd.read_csv(TWIN_LOG_FILE)
        st.dataframe(twin_df.tail(20), use_container_width=True)
        st.download_button(
            "Download cei_twin_log.csv",
            data=twin_df.to_csv(index=False).encode("utf-8"),
            file_name="cei_twin_log.csv",
            mime="text/csv",
        )
    else:
        st.info("No emotional twin logs created yet.")

    st.subheader("Recommendation feedback log")
    if RECOMMENDER_FILE.exists():
        rec_df = pd.read_csv(RECOMMENDER_FILE)
        st.dataframe(rec_df.tail(20), use_container_width=True)
        st.download_button(
            "Download recommender_stats.csv",
            data=rec_df.to_csv(index=False).encode("utf-8"),
            file_name="recommender_stats.csv",
            mime="text/csv",
        )
    else:
        st.info("No recommender feedback logs created yet.")


def render_viva_page() -> None:
    st.header("Major project viva support")
    st.write("These questions and answers are designed to help you defend the system design, research gap, explainability choices, and practical relevance.")

    for item in viva_items():
        with st.expander(item["question"]):
            st.write(item["answer"])

    st.subheader("Future scope")
    st.markdown(
        """
        1. Add privacy-preserving federated learning for personalization without centralizing sensitive data.
        2. Extend the Digital Emotional Twin with longer-term trend visualization and anomaly alerts.
        3. Add multimodal transformer fusion once higher compute and a larger labeled dataset are available.
        """
    )


def streamlit_app() -> None:
    ensure_runtime_dirs()
    st.set_page_config(page_title="CEI Adaptive Lifestyle System", layout="wide")

    with st.sidebar:
        st.title("CEI Major Project")
        page = st.radio(
            "Navigate",
            [
                "Overview",
                "Mood Analysis",
                "Dataset Tools",
                "Training Lab",
                "Logs and Exports",
                "Viva Support",
            ],
        )
        st.markdown("---")
        st.caption("Optional free API integration")
        hf_token = st.text_input(
            "Hugging Face token",
            value=os.getenv("HUGGINGFACE_API_TOKEN", ""),
            type="password",
        )
        hf_emotion_model = st.text_input("HF emotion model", value=HF_DEFAULT_EMOTION_MODEL)
        hf_generation_model = st.text_input("HF text-generation model", value=HF_DEFAULT_GENERATION_MODEL)

    if page == "Overview":
        render_home_page()
    elif page == "Mood Analysis":
        render_analysis_page(hf_token, hf_emotion_model, hf_generation_model)
    elif page == "Dataset Tools":
        render_dataset_page()
    elif page == "Training Lab":
        render_training_page()
    elif page == "Logs and Exports":
        render_logs_page()
    else:
        render_viva_page()


def cli_main() -> int:
    parser = argparse.ArgumentParser(description="CEI Adaptive Lifestyle System utilities")
    parser.add_argument("--export-catalog", action="store_true", help="Export the generated resource catalog CSV")
    parser.add_argument("--catalog-path", default=str(CATALOG_FILE), help="Output path for the catalog CSV")
    args, _ = parser.parse_known_args()

    ensure_runtime_dirs()

    if args.export_catalog:
        output_path = Path(args.catalog_path)
        catalog = export_catalog(output_path)
        print(f"Exported {len(catalog)} resources to {output_path}")
        return 0

    if st is None:
        print("Streamlit is not installed. Run `streamlit run app.py` after installing requirements.")
        return 0

    streamlit_app()
    return 0


if __name__ == "__main__":
    raise SystemExit(cli_main())
