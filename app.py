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
import tempfile
import textwrap
import urllib.error
import urllib.request
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote, quote_plus


APP_ROOT = Path(__file__).resolve().parent
MODELS_DIR = APP_ROOT / "models"
DATASET_DIR = APP_ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"
MANUAL_IMPORT_DIR = DATASET_DIR / "manual_import"
MODEL_PATH = MODELS_DIR / "efficientnetv2_emotion.keras"
MODEL_META_PATH = MODELS_DIR / "efficientnetv2_metadata.json"
BANDIT_STATE_PATH = MODELS_DIR / "recommender_bandit_state.json"
DATASET_MANIFEST_PATH = DATASET_DIR / "dataset_manifest.json"
RESOURCE_CATALOG_PATH = APP_ROOT / "resource_catalog.csv"
CEI_LOG_PATH = APP_ROOT / "cei_twin_log.csv"
RECOMMENDER_STATS_PATH = APP_ROOT / "recommender_stats.csv"

EMOTION_LABELS = ["joy", "calm", "neutral", "sadness", "anger", "fear", "surprise"]
MODALITY_WEIGHTS = {"image": 0.45, "text": 0.25, "voice": 0.20, "emoji": 0.10}
HF_CHAT_MODELS = [
    os.getenv("HF_PRIMARY_MODEL", "").strip(),
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
]

CANONICAL_LABELS = {
    "happy": "joy",
    "happiness": "joy",
    "joy": "joy",
    "positive": "joy",
    "relaxed": "calm",
    "calm": "calm",
    "peace": "calm",
    "peaceful": "calm",
    "neutral": "neutral",
    "normal": "neutral",
    "sad": "sadness",
    "sadness": "sadness",
    "depressed": "sadness",
    "anger": "anger",
    "angry": "anger",
    "frustrated": "anger",
    "disgust": "anger",
    "contempt": "anger",
    "fear": "fear",
    "fearful": "fear",
    "anxious": "fear",
    "anxiety": "fear",
    "stress": "fear",
    "stressed": "fear",
    "surprise": "surprise",
    "surprised": "surprise",
    "amazed": "surprise",
}

EMOJI_MAP = {
    "😀": "joy",
    "😊": "joy",
    "😌": "calm",
    "😐": "neutral",
    "😔": "sadness",
    "😡": "anger",
    "😰": "fear",
    "😮": "surprise",
}

TEXT_EMOTION_LEXICON = {
    "joy": [
        "happy",
        "excited",
        "grateful",
        "hopeful",
        "awesome",
        "great",
        "good",
        "energized",
        "proud",
        "love",
        "motivated",
    ],
    "calm": [
        "calm",
        "peaceful",
        "relaxed",
        "steady",
        "balanced",
        "okay",
        "fine",
        "composed",
        "settled",
        "easy",
    ],
    "neutral": [
        "normal",
        "usual",
        "routine",
        "regular",
        "average",
        "neutral",
    ],
    "sadness": [
        "sad",
        "down",
        "lonely",
        "crying",
        "hurt",
        "tired",
        "hopeless",
        "upset",
        "loss",
        "broken",
    ],
    "anger": [
        "angry",
        "frustrated",
        "annoyed",
        "irritated",
        "mad",
        "furious",
        "resentful",
        "disgusted",
        "rage",
        "tense",
    ],
    "fear": [
        "worried",
        "anxious",
        "fear",
        "scared",
        "nervous",
        "overwhelmed",
        "panic",
        "stressed",
        "uncertain",
        "afraid",
    ],
    "surprise": [
        "surprised",
        "shocked",
        "unexpected",
        "amazed",
        "suddenly",
        "wow",
        "astonished",
    ],
}

LIFESTYLE_ACTIONS = {
    "joy": [
        "Use the positive energy for one focused 25-minute work sprint.",
        "Capture one gratitude point so the emotional state becomes reusable later.",
        "Choose an uplifting but not distracting music stream during routine work.",
        "Share one encouraging message with a friend or teammate.",
    ],
    "calm": [
        "Protect this state with a low-stimulation playlist and deep work block.",
        "Drink water and keep the next task small, clear, and concrete.",
        "Do a two-minute breathing routine before switching to the next activity.",
        "Use this stable mood for study, reading, or quiet reflection.",
    ],
    "neutral": [
        "Start with one low-friction task to create momentum.",
        "Pair the next task with soft background audio or ambient sound.",
        "Add a short body movement break to prevent emotional drift.",
        "Write one sentence about what would make the next hour successful.",
    ],
    "sadness": [
        "Lower task difficulty for the next 20 minutes and focus on completion, not perfection.",
        "Play comforting, low-tempo music or listen to a calming spoken resource.",
        "Reach out to one trusted person if you need support or grounding.",
        "Try a gentle breathing or stretching routine before resuming work.",
    ],
    "anger": [
        "Delay major decisions until the body settles and breathing slows down.",
        "Use a cooldown playlist or guided breathing resource for five minutes.",
        "Convert frustration into one actionable problem statement and one next step.",
        "Step away from the trigger briefly before returning with a clearer plan.",
    ],
    "fear": [
        "Shrink the next task into the smallest safe step you can finish today.",
        "Use grounding audio and a written checklist to reduce uncertainty.",
        "Replace open-ended worry with a short plan of three realistic actions.",
        "If needed, pause notifications and create a calm, low-noise environment.",
    ],
    "surprise": [
        "Take one minute to label whether the surprise feels exciting, stressful, or both.",
        "Convert the new information into a quick note or action item.",
        "Use curiosity productively by exploring one reliable resource, not ten random links.",
        "Keep a calm soundtrack in the background so attention does not scatter.",
    ],
}

DIRECT_RESOURCES = {
    "joy": [
        ("TED ideas and motivation", "https://www.youtube.com/@TED"),
        ("KEXP live performance channel", "https://www.youtube.com/@kexp"),
    ],
    "calm": [
        ("LoFi Girl live channel", "https://www.youtube.com/@LofiGirl"),
        ("Meditative Mind channel", "https://www.youtube.com/@MeditativeMind"),
    ],
    "neutral": [
        ("Khan Academy study channel", "https://www.youtube.com/@khanacademy"),
        ("CrashCourse learning channel", "https://www.youtube.com/@crashcourse"),
    ],
    "sadness": [
        ("Headspace guided calm", "https://www.youtube.com/@Headspace"),
        ("Yoga With Adriene gentle reset", "https://www.youtube.com/@yogawithadriene"),
    ],
    "anger": [
        ("Yoga With Adriene breathwork", "https://www.youtube.com/@yogawithadriene"),
        ("TED perspective talks", "https://www.youtube.com/@TED"),
    ],
    "fear": [
        ("Headspace anxiety support", "https://www.youtube.com/@Headspace"),
        ("Meditative Mind grounding audio", "https://www.youtube.com/@MeditativeMind"),
    ],
    "surprise": [
        ("Kurzgesagt curiosity channel", "https://www.youtube.com/@kurzgesagt"),
        ("TED ideas worth sharing", "https://www.youtube.com/@TED"),
    ],
}

SEARCH_THEMES = {
    "joy": [
        "uplifting songs",
        "celebration playlist",
        "feel good instrumental",
        "happy focus music",
        "confidence boost songs",
    ],
    "calm": [
        "calm instrumental focus",
        "stress relief music",
        "deep breathing music",
        "ambient relax playlist",
        "mindful evening music",
    ],
    "neutral": [
        "balanced work playlist",
        "steady productivity music",
        "light instrumental study",
        "daily reset music",
        "gentle background music",
    ],
    "sadness": [
        "comfort songs",
        "healing piano music",
        "self compassion music",
        "gentle acoustic support",
        "mood lifting calm songs",
    ],
    "anger": [
        "cool down music",
        "release tension playlist",
        "breathing reset music",
        "low tempo focus music",
        "anger management meditation",
    ],
    "fear": [
        "anxiety relief music",
        "grounding music",
        "confidence meditation",
        "safe calm playlist",
        "worry reset ambient music",
    ],
    "surprise": [
        "curious discovery playlist",
        "cinematic wonder music",
        "fresh ideas soundtrack",
        "creative spark music",
        "exploration playlist",
    ],
}

CEI_LOG_FIELDS = [
    "timestamp",
    "user_id",
    "fused_emotion",
    "fused_confidence",
    "image_emotion",
    "text_emotion",
    "voice_emotion",
    "emoji_emotion",
    "modalities_used",
    "ethical_risk",
    "ethical_flags",
    "question_source",
    "top_recommendation",
    "context_excerpt",
]

RECOMMENDER_STATS_FIELDS = [
    "timestamp",
    "user_id",
    "emotion",
    "title",
    "url",
    "source",
    "reward",
    "feedback_note",
]

_MODEL_CACHE: Dict[str, Any] = {}


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def slugify(text: Any) -> str:
    clean = re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower()).strip("_")
    return clean or "neutral"


def canonical_emotion(label: Any) -> str:
    slug = slugify(label)
    return CANONICAL_LABELS.get(slug, slug if slug in EMOTION_LABELS else "neutral")


def make_peaked_distribution(label: str, peak: float = 0.80) -> Dict[str, float]:
    label = canonical_emotion(label)
    peak = max(0.0, min(0.98, peak))
    remainder = 1.0 - peak
    base = remainder / max(1, len(EMOTION_LABELS) - 1)
    distribution = {emotion: base for emotion in EMOTION_LABELS}
    distribution[label] = peak
    return distribution


def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    normalized = {emotion: max(0.0, float(scores.get(emotion, 0.0))) for emotion in EMOTION_LABELS}
    total = sum(normalized.values())
    if total <= 0:
        return make_peaked_distribution("neutral", peak=0.55)
    return {emotion: value / total for emotion, value in normalized.items()}


def emotion_result(
    label: str,
    confidence: float,
    method: str,
    distribution: Optional[Dict[str, float]] = None,
    note: str = "",
) -> Dict[str, Any]:
    canonical = canonical_emotion(label)
    dist = distribution or make_peaked_distribution(canonical, peak=max(0.45, min(0.95, confidence)))
    normalized_dist = normalize_scores(dist)
    return {
        "label": canonical,
        "confidence": round(float(confidence), 4),
        "method": method,
        "distribution": normalized_dist,
        "note": note,
    }


def ensure_runtime_dirs() -> None:
    for path in [MODELS_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR, MANUAL_IMPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def ensure_csv_header(path: Path, fieldnames: Sequence[str]) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()


def write_csv_row(path: Path, row: Dict[str, Any], fieldnames: Sequence[str]) -> None:
    ensure_csv_header(path, fieldnames)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writerow({name: row.get(name, "") for name in fieldnames})


def read_csv_rows(path: Path, limit: int = 25) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-limit:]


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def bootstrap_project() -> Dict[str, Any]:
    ensure_runtime_dirs()
    ensure_csv_header(CEI_LOG_PATH, CEI_LOG_FIELDS)
    ensure_csv_header(RECOMMENDER_STATS_PATH, RECOMMENDER_STATS_FIELDS)
    catalog_path, catalog_count = export_resource_catalog()
    return {
        "runtime_directories": [str(MODELS_DIR), str(DATASET_DIR)],
        "generated_files": [str(catalog_path), str(CEI_LOG_PATH), str(RECOMMENDER_STATS_PATH)],
        "catalog_items": catalog_count,
    }


def generate_resource_catalog() -> List[Dict[str, str]]:
    catalog: List[Dict[str, str]] = []
    for mood, items in DIRECT_RESOURCES.items():
        for title, url in items:
            catalog.append(
                {
                    "mood": mood,
                    "title": title,
                    "url": url,
                    "source": "youtube_direct",
                    "type": "direct",
                    "offline_fallback": "Open a locally saved focus playlist or use a breathing timer for 3 minutes.",
                }
            )
    for mood, themes in SEARCH_THEMES.items():
        for theme in themes:
            query = f"{mood} {theme}"
            catalog.extend(
                [
                    {
                        "mood": mood,
                        "title": f"{theme.title()} - YouTube Search",
                        "url": f"https://www.youtube.com/results?search_query={quote_plus(query)}",
                        "source": "youtube_search",
                        "type": "search",
                        "offline_fallback": "Search your local device for similar music or use a saved study playlist.",
                    },
                    {
                        "mood": mood,
                        "title": f"{theme.title()} - Spotify Search",
                        "url": f"https://open.spotify.com/search/{quote(query, safe='')}",
                        "source": "spotify_search",
                        "type": "search",
                        "offline_fallback": "Use an offline playlist with the same mood tag if Spotify is unavailable.",
                    },
                    {
                        "mood": mood,
                        "title": f"{theme.title()} - YouTube Music Search",
                        "url": f"https://music.youtube.com/search?q={quote_plus(query)}",
                        "source": "youtube_music_search",
                        "type": "search",
                        "offline_fallback": "Switch to downloaded instrumental audio or ambient sound on your device.",
                    },
                ]
            )
    return catalog


def export_resource_catalog(path: Path = RESOURCE_CATALOG_PATH) -> Tuple[Path, int]:
    catalog = generate_resource_catalog()
    fieldnames = ["mood", "title", "url", "source", "type", "offline_fallback"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(catalog)
    return path, len(catalog)


def make_local_user_id(profile_name: str, security_key: str) -> str:
    payload = f"{profile_name.strip().lower()}::{security_key.strip()}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def tokenize_text(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def analyze_text(text: str, method_name: str = "text-lexicon") -> Optional[Dict[str, Any]]:
    if not text or not text.strip():
        return None
    tokens = tokenize_text(text)
    if not tokens:
        return emotion_result("neutral", 0.42, method_name, note="No strong lexical signals detected.")

    joined = " ".join(tokens)
    scores: Dict[str, float] = defaultdict(float)
    for emotion, keywords in TEXT_EMOTION_LEXICON.items():
        for keyword in keywords:
            if " " in keyword:
                if keyword in joined:
                    scores[emotion] += 2.0
            else:
                count = tokens.count(keyword)
                if count:
                    scores[emotion] += count * 1.0

    if "!" in text:
        scores["joy"] += 0.3
        scores["surprise"] += 0.2
    if "?" in text:
        scores["fear"] += 0.1
    if any(word in joined for word in ["deadline", "exam", "interview", "uncertain"]):
        scores["fear"] += 0.8
    if any(word in joined for word in ["bored", "blank", "empty"]):
        scores["neutral"] += 0.6
        scores["sadness"] += 0.4

    if not scores:
        return emotion_result("neutral", 0.46, method_name, note="Defaulted to neutral due to limited text cues.")

    distribution = normalize_scores(scores)
    label = max(distribution, key=distribution.get)
    confidence = 0.45 + min(0.45, distribution[label] * 0.7) + min(0.05, len(tokens) / 200)
    return emotion_result(label, confidence, method_name, distribution=distribution)


def analyze_emoji(emoji_value: str) -> Optional[Dict[str, Any]]:
    if not emoji_value or emoji_value == "None":
        return None
    label = EMOJI_MAP.get(emoji_value, "neutral")
    return emotion_result(label, 0.78, "emoji-map")


def heuristic_image_emotion(pil_image: Any) -> Dict[str, Any]:
    from PIL import ImageStat

    image = pil_image.convert("RGB").resize((128, 128))
    stat = ImageStat.Stat(image)
    means = [channel / 255.0 for channel in stat.mean]
    avg_std = sum(stat.stddev) / 3.0 / 255.0
    brightness = sum(means) / 3.0
    warmth = means[0] - means[2]

    scores = {emotion: 0.12 for emotion in EMOTION_LABELS}
    if brightness > 0.65 and avg_std > 0.12:
        scores["joy"] += 1.1
    if 0.45 <= brightness <= 0.70 and avg_std < 0.11:
        scores["calm"] += 1.0
    if brightness < 0.38 and avg_std < 0.13:
        scores["sadness"] += 1.1
    if warmth > 0.08 and avg_std > 0.15:
        scores["anger"] += 1.0
    if brightness < 0.35 and avg_std > 0.14:
        scores["fear"] += 0.8
    if avg_std > 0.22:
        scores["surprise"] += 0.7
    if abs(warmth) < 0.03:
        scores["neutral"] += 0.6

    distribution = normalize_scores(scores)
    label = max(distribution, key=distribution.get)
    confidence = 0.48 + distribution[label] * 0.25
    return emotion_result(
        label,
        confidence,
        "heuristic-image",
        distribution=distribution,
        note="Heuristic fallback was used because no trained model is saved.",
    )


def import_tensorflow():
    try:
        import tensorflow as tf
    except Exception as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "TensorFlow is not available. Install it from requirements.txt or run the app in heuristic mode."
        ) from exc
    return tf


def load_trained_model():
    cache_key = str(MODEL_PATH)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    if not MODEL_PATH.exists():
        return None, {}, None
    tf = import_tensorflow()
    model = tf.keras.models.load_model(MODEL_PATH)
    meta = load_json(
        MODEL_META_PATH,
        {"class_names": EMOTION_LABELS, "image_size": [224, 224], "backbone": "EfficientNetV2-B0"},
    )
    _MODEL_CACHE[cache_key] = (model, meta, tf)
    return _MODEL_CACHE[cache_key]


def predict_image_emotion(pil_image: Any) -> Dict[str, Any]:
    try:
        model, meta, tf = load_trained_model()
    except RuntimeError:
        return heuristic_image_emotion(pil_image)

    if model is None:
        return heuristic_image_emotion(pil_image)

    import numpy as np

    image_size = tuple(meta.get("image_size", [224, 224]))
    image = pil_image.convert("RGB").resize(image_size)
    array = tf.keras.utils.img_to_array(image)
    array = tf.keras.applications.efficientnet_v2.preprocess_input(array)
    array = np.expand_dims(array, axis=0)
    probabilities = model.predict(array, verbose=0)[0]
    class_names = meta.get("class_names", EMOTION_LABELS)
    scores = {emotion: 0.0 for emotion in EMOTION_LABELS}
    for class_name, probability in zip(class_names, probabilities):
        scores[canonical_emotion(class_name)] += float(probability)
    distribution = normalize_scores(scores)
    label = max(distribution, key=distribution.get)
    return emotion_result(label, distribution[label], meta.get("backbone", "EfficientNetV2-B0"), distribution)


def find_last_conv_layer(model: Any) -> str:
    for layer in reversed(model.layers):
        output = getattr(layer, "output", None)
        shape = getattr(output, "shape", None)
        if shape is not None and len(shape) == 4:
            return layer.name
    raise ValueError("No convolutional layer was found for Grad-CAM.")


def generate_gradcam_overlay(pil_image: Any) -> Tuple[Optional[Any], str]:
    try:
        model, meta, tf = load_trained_model()
    except RuntimeError as exc:
        return None, str(exc)
    if model is None:
        return None, "Train and save the EfficientNetV2 model first to enable Grad-CAM."

    import matplotlib.cm as cm
    import numpy as np
    from PIL import Image

    image_size = tuple(meta.get("image_size", [224, 224]))
    image = pil_image.convert("RGB").resize(image_size)
    array = tf.keras.utils.img_to_array(image)
    array = tf.keras.applications.efficientnet_v2.preprocess_input(array)
    array = np.expand_dims(array, axis=0)
    predicted_index = int(tf.argmax(model(array, training=False)[0]))
    last_conv_name = find_last_conv_layer(model)
    grad_model = tf.keras.models.Model(model.inputs, [model.get_layer(last_conv_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(array)
        loss = predictions[:, predicted_index]
    gradients = tape.gradient(loss, conv_outputs)
    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_gradients, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    jet = cm.get_cmap("jet")
    colored_heatmap = jet(heatmap)[..., :3]
    colored_heatmap = Image.fromarray((colored_heatmap * 255).astype("uint8")).resize(pil_image.size)
    overlay = Image.blend(pil_image.convert("RGB"), colored_heatmap, alpha=0.35)
    return overlay, f"Grad-CAM generated from {last_conv_name}."


def transcribe_audio_bytes(file_bytes: bytes, suffix: str = ".wav") -> Tuple[str, str]:
    try:
        import speech_recognition as sr
    except Exception:
        return "", "SpeechRecognition is not installed. Use the transcript text box as fallback."

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            handle.write(file_bytes)
            tmp_path = handle.name
        recognizer = sr.Recognizer()
        with sr.AudioFile(tmp_path) as source:
            audio = recognizer.record(source)
        transcript = recognizer.recognize_google(audio)
        return transcript, "Free Google Web Speech transcription was used."
    except Exception as exc:  # pragma: no cover - depends on network/audio
        return "", f"Automatic transcription was unavailable: {exc}"
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def compute_modality_agreement(modality_results: Dict[str, Optional[Dict[str, Any]]]) -> float:
    labels = [result["label"] for result in modality_results.values() if result]
    if not labels:
        return 0.0
    top_count = Counter(labels).most_common(1)[0][1]
    return round(top_count / len(labels), 2)


def fuse_emotions(modality_results: Dict[str, Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    scores = {emotion: 0.0 for emotion in EMOTION_LABELS}
    used = []
    for modality, result in modality_results.items():
        if not result:
            continue
        used.append(modality)
        weight = MODALITY_WEIGHTS.get(modality, 0.10)
        for emotion, value in result["distribution"].items():
            scores[emotion] += weight * float(value)
    distribution = normalize_scores(scores)
    label = max(distribution, key=distribution.get)
    ranking = sorted(distribution.items(), key=lambda item: item[1], reverse=True)
    return {
        "label": label,
        "confidence": round(distribution[label], 4),
        "distribution": distribution,
        "runner_up": ranking[1][0] if len(ranking) > 1 else label,
        "modalities_used": used,
        "agreement": compute_modality_agreement(modality_results),
    }


def ethical_ai_monitor(modality_results: Dict[str, Optional[Dict[str, Any]]], fused_result: Dict[str, Any]) -> Dict[str, Any]:
    flags: List[str] = []
    used_results = {name: result for name, result in modality_results.items() if result}
    labels = [result["label"] for result in used_results.values()]
    if len(used_results) < 2:
        flags.append("Single-modality reading detected. Add text, voice, or emoji context for a more stable result.")
    if fused_result["confidence"] < 0.55:
        flags.append("Low-confidence fusion detected. Treat the output as a suggestion, not a fact.")
    if len(set(labels)) > 2:
        flags.append("Modalities disagree strongly. Conservative recommendations were preferred.")
    image_result = used_results.get("image")
    if image_result and image_result["method"] == "heuristic-image":
        flags.append("The face image result came from heuristic fallback because no trained model is currently saved.")
    risk = "low"
    if len(flags) >= 3 or fused_result["confidence"] < 0.40:
        risk = "high"
    elif flags:
        risk = "moderate"
    return {
        "risk_level": risk,
        "agreement": fused_result.get("agreement", 0.0),
        "flags": flags,
        "safeguards": [
            "The system is a wellbeing support tool, not a diagnostic tool.",
            "Only short excerpts should be logged if the user is handling sensitive information.",
            "Feedback updates only the local recommender state inside this project folder.",
        ],
    }


def fallback_questions(emotion: str) -> List[str]:
    banks = {
        "joy": [
            "What is going well right now that you can intentionally continue?",
            "How can you use this energy for one meaningful task today?",
            "Who could benefit from the positive momentum you have right now?",
        ],
        "calm": [
            "What task is easiest to finish while your mind feels steady?",
            "How can you protect this calm state for the next hour?",
            "What small routine helps you stay centered when work becomes busy?",
        ],
        "neutral": [
            "What would make the next hour feel productive and manageable?",
            "Which tiny action could create momentum without pressure?",
            "What kind of background environment helps you focus best today?",
        ],
        "sadness": [
            "What is one gentle action that would make today feel a little lighter?",
            "Is there one person, place, or habit that usually helps you feel supported?",
            "What important task can be reduced into a kinder, smaller version right now?",
        ],
        "anger": [
            "What part of the situation is truly in your control right now?",
            "What would help your body settle before you respond?",
            "What problem statement can replace the emotional trigger in one sentence?",
        ],
        "fear": [
            "What is the smallest safe next step you can complete today?",
            "Which part of the situation is uncertain, and which part is already clear?",
            "What would help you feel more grounded in the next ten minutes?",
        ],
        "surprise": [
            "Is this new information exciting, stressful, or a mix of both?",
            "What should be captured immediately before the moment passes?",
            "What one action matters most after this unexpected event?",
        ],
    }
    return banks.get(emotion, banks["neutral"])


def parse_questions(text: str) -> List[str]:
    lines = []
    for raw_line in text.splitlines():
        cleaned = raw_line.strip().lstrip("-").strip()
        cleaned = re.sub(r"^\d+[\).\s-]*", "", cleaned).strip()
        if cleaned:
            lines.append(cleaned)
    unique = []
    for line in lines:
        if line not in unique:
            unique.append(line)
    return unique[:3]


def request_hf_questions(hf_token: str, emotion: str, context_text: str) -> Tuple[List[str], str]:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a supportive emotional intelligence assistant for a student project. "
                "Generate exactly 3 short reflective questions. Avoid diagnosis, therapy claims, or crisis language. "
                "Keep each question under 20 words."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Detected emotion: {emotion}. Context: {context_text or 'No additional context provided.'} "
                "Return only 3 numbered reflective questions that help the user understand mood and choose one healthy action."
            ),
        },
    ]
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {hf_token}",
    }

    for model_id in [item for item in HF_CHAT_MODELS if item]:
        request_body = json.dumps(
            {
                "model": model_id,
                "messages": messages,
                "max_tokens": 220,
                "temperature": 0.6,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            "https://router.huggingface.co/v1/chat/completions",
            data=request_body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                data = json.loads(response.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            questions = parse_questions(content)
            if questions:
                return questions, f"Hugging Face Inference ({model_id})"
        except urllib.error.URLError:
            continue
        except Exception:
            continue
    return fallback_questions(emotion), "Local fallback template"


def generate_reflective_questions(emotion: str, context_text: str, hf_token: str) -> Dict[str, Any]:
    if hf_token and hf_token.strip():
        questions, source = request_hf_questions(hf_token.strip(), emotion, context_text)
    else:
        questions, source = fallback_questions(emotion), "Local fallback template"
    return {"questions": questions, "source": source}


def build_adaptive_plan(emotion: str) -> List[str]:
    return LIFESTYLE_ACTIONS.get(emotion, LIFESTYLE_ACTIONS["neutral"])


def load_bandit_state() -> Dict[str, Any]:
    return load_json(BANDIT_STATE_PATH, {})


def save_bandit_state(state: Dict[str, Any]) -> None:
    save_json(BANDIT_STATE_PATH, state)


def recommend_resources(
    emotion: str,
    user_id: str,
    recent_urls: Optional[List[str]] = None,
    top_n: int = 6,
) -> List[Dict[str, Any]]:
    catalog = [item for item in generate_resource_catalog() if item["mood"] == canonical_emotion(emotion)]
    state = load_bandit_state()
    recent_urls = recent_urls or []
    emotion_prefix = f"{user_id}::{emotion}::"
    total_visits = sum(entry.get("n", 0) for key, entry in state.items() if key.startswith(emotion_prefix))
    scored: List[Tuple[float, Dict[str, Any]]] = []

    for item in catalog:
        key = f"{user_id}::{emotion}::{item['url']}"
        bandit_entry = state.get(key, {"q": 0.0, "n": 0})
        q_value = float(bandit_entry.get("q", 0.0))
        visits = int(bandit_entry.get("n", 0))
        exploration = math.sqrt((2.0 * math.log(total_visits + 2.0)) / (visits + 1.0))
        novelty_bonus = 0.55 if item["url"] not in recent_urls else -1.40
        source_bonus = {
            "youtube_direct": 0.18,
            "youtube_search": 0.10,
            "spotify_search": 0.08,
            "youtube_music_search": 0.08,
        }.get(item["source"], 0.05)
        score = q_value + 0.25 * exploration + novelty_bonus + source_bonus + random.random() * 0.01
        scored.append((score, item))

    scored.sort(key=lambda entry: entry[0], reverse=True)
    return [item for _, item in scored[:top_n]]


def record_feedback(
    user_id: str,
    emotion: str,
    recommendations: List[Dict[str, Any]],
    reward: int,
    feedback_note: str,
) -> None:
    state = load_bandit_state()
    for item in recommendations:
        key = f"{user_id}::{emotion}::{item['url']}"
        entry = state.get(key, {"q": 0.0, "n": 0, "title": item["title"], "source": item["source"]})
        entry["n"] += 1
        alpha = 1.0 / entry["n"]
        entry["q"] = round(entry["q"] + alpha * (reward - entry["q"]), 4)
        state[key] = entry
        write_csv_row(
            RECOMMENDER_STATS_PATH,
            {
                "timestamp": now_iso(),
                "user_id": user_id,
                "emotion": emotion,
                "title": item["title"],
                "url": item["url"],
                "source": item["source"],
                "reward": reward,
                "feedback_note": feedback_note,
            },
            RECOMMENDER_STATS_FIELDS,
        )
    save_bandit_state(state)


def log_twin_event(
    user_id: str,
    fused_result: Dict[str, Any],
    modality_results: Dict[str, Optional[Dict[str, Any]]],
    ethical_report: Dict[str, Any],
    question_source: str,
    recommendations: List[Dict[str, Any]],
    context_text: str,
) -> None:
    write_csv_row(
        CEI_LOG_PATH,
        {
            "timestamp": now_iso(),
            "user_id": user_id,
            "fused_emotion": fused_result["label"],
            "fused_confidence": fused_result["confidence"],
            "image_emotion": (modality_results.get("image") or {}).get("label", ""),
            "text_emotion": (modality_results.get("text") or {}).get("label", ""),
            "voice_emotion": (modality_results.get("voice") or {}).get("label", ""),
            "emoji_emotion": (modality_results.get("emoji") or {}).get("label", ""),
            "modalities_used": ",".join(fused_result["modalities_used"]),
            "ethical_risk": ethical_report["risk_level"],
            "ethical_flags": " | ".join(ethical_report["flags"]),
            "question_source": question_source,
            "top_recommendation": recommendations[0]["title"] if recommendations else "",
            "context_excerpt": (context_text or "").strip()[:160],
        },
        CEI_LOG_FIELDS,
    )


def clear_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def can_stratify(labels: List[str]) -> bool:
    counts = Counter(labels)
    return len(counts) > 1 and all(count >= 2 for count in counts.values())


def to_pil_image(item: Any):
    from PIL import Image

    if hasattr(item, "convert"):
        return item.convert("RGB")
    if isinstance(item, dict):
        if item.get("bytes"):
            return Image.open(io.BytesIO(item["bytes"])).convert("RGB")
        if item.get("path"):
            return Image.open(item["path"]).convert("RGB")
    raise ValueError("Unsupported image object in dataset.")


def detect_image_and_label_columns(dataset: Any) -> Tuple[str, str]:
    from datasets import ClassLabel, Image

    image_col = ""
    label_col = ""
    for name, feature in dataset.features.items():
        if isinstance(feature, Image) and not image_col:
            image_col = name
        if isinstance(feature, ClassLabel) and not label_col:
            label_col = name
    if not image_col:
        for name in dataset.features:
            if any(token in name.lower() for token in ["image", "img", "face", "pixel"]):
                image_col = name
                break
    if not label_col:
        for name in dataset.features:
            if any(token in name.lower() for token in ["label", "emotion", "sentiment", "class", "target"]):
                label_col = name
                break
    if not image_col or not label_col:
        raise ValueError("Could not detect both image and label columns automatically.")
    return image_col, label_col


def normalize_dataset_label(raw_label: Any, class_names: Optional[List[str]] = None) -> str:
    if class_names and isinstance(raw_label, int) and 0 <= raw_label < len(class_names):
        return canonical_emotion(class_names[raw_label])
    return canonical_emotion(raw_label)


def search_hf_datasets(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    from huggingface_hub import HfApi

    api = HfApi()
    results = []
    for item in api.list_datasets(search=query, limit=limit):
        results.append(
            {
                "id": item.id,
                "downloads": getattr(item, "downloads", ""),
                "likes": getattr(item, "likes", ""),
                "gated": getattr(item, "gated", False),
                "private": getattr(item, "private", False),
            }
        )
    return results


def prepare_sampled_hf_dataset(dataset_id: str, sample_size: int = 800, image_size: int = 224) -> Dict[str, Any]:
    from datasets import concatenate_datasets, load_dataset
    from sklearn.model_selection import train_test_split

    ensure_runtime_dirs()
    clear_directory(TRAIN_DIR)
    clear_directory(VAL_DIR)
    clear_directory(TEST_DIR)

    loaded = load_dataset(dataset_id)
    if hasattr(loaded, "keys"):
        split_names = list(loaded.keys())
        dataset = concatenate_datasets([loaded[name] for name in split_names])
    else:
        split_names = ["train"]
        dataset = loaded

    image_col, label_col = detect_image_and_label_columns(dataset)
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(range(min(sample_size, len(dataset))))
    label_feature = dataset.features.get(label_col)
    class_names = getattr(label_feature, "names", None)

    prepared_rows = []
    for item in dataset:
        try:
            image = to_pil_image(item[image_col]).resize((image_size, image_size))
            label = normalize_dataset_label(item[label_col], class_names)
            prepared_rows.append({"image": image, "label": label})
        except Exception:
            continue

    if len(prepared_rows) < 12:
        raise ValueError("Too few valid image rows were extracted from the dataset.")

    labels = [row["label"] for row in prepared_rows]
    indices = list(range(len(prepared_rows)))
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.30,
        random_state=42,
        stratify=labels if can_stratify(labels) else None,
    )
    temp_labels = [labels[index] for index in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=42,
        stratify=temp_labels if can_stratify(temp_labels) else None,
    )

    split_map = {"train": train_idx, "val": val_idx, "test": test_idx}
    split_counts: Dict[str, Dict[str, int]] = {}
    for split_name, split_indices in split_map.items():
        split_counter = Counter()
        for local_index, data_index in enumerate(split_indices):
            row = prepared_rows[data_index]
            split_counter[row["label"]] += 1
            output_dir = DATASET_DIR / split_name / row["label"]
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{split_name}_{local_index:05d}.jpg"
            row["image"].save(output_path, format="JPEG", quality=95)
        split_counts[split_name] = dict(sorted(split_counter.items()))

    manifest = {
        "dataset_id": dataset_id,
        "source_splits": split_names,
        "sample_size_requested": sample_size,
        "sample_size_prepared": len(prepared_rows),
        "image_column": image_col,
        "label_column": label_col,
        "image_size": image_size,
        "class_distribution": dict(sorted(Counter(labels).items())),
        "split_distribution": split_counts,
        "prepared_at": now_iso(),
    }
    save_json(DATASET_MANIFEST_PATH, manifest)
    return manifest


def extract_manual_zip(file_bytes: bytes) -> Dict[str, Any]:
    ensure_runtime_dirs()
    clear_directory(MANUAL_IMPORT_DIR)
    with zipfile.ZipFile(io.BytesIO(file_bytes), "r") as archive:
        archive.extractall(MANUAL_IMPORT_DIR)
        file_count = len(archive.namelist())
    return {"target": str(MANUAL_IMPORT_DIR), "file_count": file_count}


def directory_has_images(path: Path) -> bool:
    for pattern in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        if any(path.rglob(pattern)):
            return True
    return False


def load_tf_dataset(split_dir: Path, image_size: int, batch_size: int, shuffle: bool):
    if not split_dir.exists() or not directory_has_images(split_dir):
        return None, []
    tf = import_tensorflow()
    dataset = tf.keras.utils.image_dataset_from_directory(
        split_dir,
        labels="inferred",
        label_mode="int",
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=shuffle,
    )
    class_names = [canonical_emotion(name) for name in dataset.class_names]
    preprocess = tf.keras.applications.efficientnet_v2.preprocess_input
    dataset = dataset.map(
        lambda x, y: (preprocess(tf.cast(x, tf.float32)), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)
    return dataset, class_names


def build_emotion_model(num_classes: int, image_size: int, learning_rate: float, unfrozen_layers: int = 40):
    tf = import_tensorflow()
    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="image_input")
    base_model = tf.keras.applications.EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
    )
    unfrozen_layers = max(0, min(unfrozen_layers, len(base_model.layers)))
    for layer in base_model.layers[:-unfrozen_layers]:
        layer.trainable = False
    for layer in base_model.layers[-unfrozen_layers:]:
        layer.trainable = True

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dropout(0.30, name="dropout_1")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="dense_projection")(x)
    x = tf.keras.layers.Dropout(0.20, name="dropout_2")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="emotion_head")(x)
    model = tf.keras.Model(inputs, outputs, name="cei_efficientnetv2_gap")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def evaluate_model(model: Any, dataset: Any, class_names: List[str]) -> Dict[str, Any]:
    import numpy as np
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
    from sklearn.preprocessing import label_binarize

    probabilities = model.predict(dataset, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    truth: List[int] = []
    for _, labels in dataset:
        truth.extend(labels.numpy().tolist())

    accuracy = accuracy_score(truth, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth,
        predictions,
        average="weighted",
        zero_division=0,
    )
    confusion = confusion_matrix(truth, predictions).tolist()
    roc_auc = None
    try:
        classes = list(range(len(class_names)))
        binarized = label_binarize(truth, classes=classes)
        if binarized.shape[1] == 1:
            roc_auc = None
        elif binarized.shape[1] == 2:
            roc_auc = float(roc_auc_score(binarized, probabilities[:, 1]))
        else:
            roc_auc = float(
                roc_auc_score(binarized, probabilities, average="weighted", multi_class="ovr")
            )
    except Exception:
        roc_auc = None

    return {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4) if roc_auc is not None else None,
        "confusion_matrix": confusion,
    }


def train_emotion_model(
    epochs: int = 2,
    batch_size: int = 8,
    image_size: int = 224,
    learning_rate: float = 1e-4,
    unfrozen_layers: int = 40,
) -> Dict[str, Any]:
    ensure_runtime_dirs()
    train_ds, class_names = load_tf_dataset(TRAIN_DIR, image_size, batch_size, shuffle=True)
    val_ds, _ = load_tf_dataset(VAL_DIR, image_size, batch_size, shuffle=False)
    test_ds, _ = load_tf_dataset(TEST_DIR, image_size, batch_size, shuffle=False)
    if train_ds is None or val_ds is None:
        raise ValueError("Training requires dataset/train and dataset/val folders with class subdirectories.")

    tf = import_tensorflow()
    model = build_emotion_model(
        num_classes=len(class_names),
        image_size=image_size,
        learning_rate=learning_rate,
        unfrozen_layers=unfrozen_layers,
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
        )
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=1)
    model.save(MODEL_PATH)
    metadata = {
        "class_names": class_names,
        "image_size": [image_size, image_size],
        "backbone": "EfficientNetV2-B0",
        "gap_layer": "global_average_pooling",
        "saved_at": now_iso(),
        "dataset_manifest": str(DATASET_MANIFEST_PATH.name) if DATASET_MANIFEST_PATH.exists() else "",
    }
    save_json(MODEL_META_PATH, metadata)
    _MODEL_CACHE.clear()
    evaluation_ds = test_ds if test_ds is not None else val_ds
    metrics = evaluate_model(model, evaluation_ds, class_names)
    metrics["evaluation_split"] = "test" if test_ds is not None else "val"
    return {"history": history.history, "metrics": metrics, "class_names": class_names, "metadata": metadata}


def probability_rows(distribution: Dict[str, float]) -> List[Dict[str, Any]]:
    ranked = sorted(distribution.items(), key=lambda item: item[1], reverse=True)
    return [{"emotion": emotion, "score": round(value, 4)} for emotion, value in ranked]


def plot_history(history: Dict[str, List[float]]):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.get("loss", []), label="train_loss", marker="o")
    if "val_loss" in history:
        axes[0].plot(history["val_loss"], label="val_loss", marker="o")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.get("accuracy", []), label="train_accuracy", marker="o")
    if "val_accuracy" in history:
        axes[1].plot(history["val_accuracy"], label="val_accuracy", marker="o")
    axes[1].set_title("Training Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    figure.tight_layout()
    return figure


def plot_confusion(confusion_matrix_data: List[List[int]], class_names: List[str]):
    import matplotlib.pyplot as plt
    import numpy as np

    matrix = np.array(confusion_matrix_data)
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(class_names)))
    axis.set_yticks(range(len(class_names)))
    axis.set_xticklabels(class_names, rotation=45, ha="right")
    axis.set_yticklabels(class_names)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Confusion Matrix")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            axis.text(col, row, int(matrix[row, col]), ha="center", va="center", color="black")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    return figure


def app_help_text() -> str:
    return textwrap.dedent(
        """
        Quick start
        1. Create a virtual environment: python -m venv .venv
        2. Activate it in PowerShell: .\\.venv\\Scripts\\Activate.ps1
        3. Install packages: pip install -r requirements.txt
        4. Optional free Hugging Face token: set HF_TOKEN in your environment
        5. Initialize runtime files: python app.py --bootstrap-project
        6. Launch the UI: streamlit run app.py

        Why Hugging Face is used instead of password-based API login
        - Hugging Face supports token-based access, which is safer than putting account passwords in code.
        - The app accepts a local profile name and security key only for local personalization and logging.
        - For text generation, the app uses a free Hugging Face token if you provide one, otherwise it falls back to offline templates.

        Free Android / APK shortcut
        - Deploy the Streamlit app to a reachable URL.
        - Open it in Chrome on Android and use Add to Home Screen for a free install-like experience.
        - If an APK is mandatory, wrap the deployed URL in a simple Android WebView project.
        """
    ).strip()


def run_streamlit_app() -> None:  # pragma: no cover - UI runtime
    import streamlit as st
    from PIL import Image

    ensure_runtime_dirs()
    ensure_csv_header(CEI_LOG_PATH, CEI_LOG_FIELDS)
    ensure_csv_header(RECOMMENDER_STATS_PATH, RECOMMENDER_STATS_FIELDS)

    st.set_page_config(
        page_title="Cognitive Emotion Intelligence & Adaptive Lifestyle System",
        page_icon=":sparkles:",
        layout="wide",
    )
    st.title("Cognitive Emotion Intelligence & Adaptive Lifestyle System")
    st.caption(
        "Single-file Streamlit app with multimodal emotion fusion, EfficientNetV2 + GAP training, "
        "Grad-CAM explainability, ethical AI monitoring, and a history-aware recommender."
    )

    st.session_state.setdefault("recent_urls", [])

    with st.sidebar:
        st.subheader("Profile and free API setup")
        profile_name = st.text_input("Local profile name", value=st.session_state.get("profile_name", "student_demo"))
        security_key = st.text_input("Local security key", type="password", help="Used only to create a local hashed user ID.")
        st.session_state["profile_name"] = profile_name
        user_id = make_local_user_id(profile_name or "student_demo", security_key or "demo")
        st.code(f"Local user id: {user_id}", language="text")

        hf_username = st.text_input("Hugging Face username (optional)", value=os.getenv("HF_USERNAME", ""))
        hf_token = st.text_input("Hugging Face token (optional)", value=os.getenv("HF_TOKEN", ""), type="password")
        st.info("Use a Hugging Face token. Do not enter your Hugging Face account password here.")

        if MODEL_PATH.exists():
            meta = load_json(MODEL_META_PATH, {})
            st.success(
                f"Saved model detected: {meta.get('backbone', 'EfficientNetV2-B0')} | "
                f"classes: {len(meta.get('class_names', []))}"
            )
        else:
            st.warning("No trained emotion model is saved yet. Image analysis will use heuristic fallback.")

        st.caption("Free default mode: Hugging Face token + local templates + search links. No paid OpenAI or Spotify key is required.")

    tab_assess, tab_training, tab_catalog, tab_logs, tab_help = st.tabs(
        ["Live Assessment", "Dataset and Training", "Catalog", "Twin Logs", "Project Guide"]
    )

    with tab_assess:
        st.subheader("Multimodal mood fusion")
        with st.form("assessment_form"):
            col1, col2 = st.columns(2)
            with col1:
                uploaded_image = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
                emoji_value = st.selectbox("Emoji snapshot", ["None", "😀", "😊", "😌", "😐", "😔", "😡", "😰", "😮"])
                uploaded_audio = st.file_uploader(
                    "Optional voice note for free speech-to-text",
                    type=["wav", "aiff", "aif", "flac"],
                )
                auto_transcribe = st.checkbox("Try free speech-to-text when audio is uploaded", value=True)
            with col2:
                context_text = st.text_area("Free-text context", height=180, placeholder="Describe how you feel and what is happening.")
                manual_voice_text = st.text_area(
                    "Voice transcript fallback",
                    height=120,
                    placeholder="Paste or type the transcript here if automatic speech-to-text is unavailable.",
                )
            submitted = st.form_submit_button("Analyze mood and recommend")

        if submitted:
            transcription_note = ""
            transcribed_text = manual_voice_text.strip()
            if uploaded_audio is not None and auto_transcribe and not transcribed_text:
                transcribed_text, transcription_note = transcribe_audio_bytes(
                    uploaded_audio.getvalue(),
                    suffix=Path(uploaded_audio.name).suffix or ".wav",
                )

            image_result = None
            image_bytes = None
            if uploaded_image is not None:
                image_bytes = uploaded_image.getvalue()
                image_result = predict_image_emotion(Image.open(io.BytesIO(image_bytes)))

            text_result = analyze_text(context_text, method_name="text-lexicon")
            voice_result = analyze_text(transcribed_text, method_name="voice-transcript") if transcribed_text else None
            emoji_result = analyze_emoji(emoji_value)
            modality_results = {
                "image": image_result,
                "text": text_result,
                "voice": voice_result,
                "emoji": emoji_result,
            }
            fused = fuse_emotions(modality_results)
            ethical_report = ethical_ai_monitor(modality_results, fused)
            combined_context = " ".join([part for part in [context_text, transcribed_text] if part]).strip()
            questions = generate_reflective_questions(fused["label"], combined_context, hf_token)
            recommendations = recommend_resources(
                fused["label"],
                user_id=user_id,
                recent_urls=st.session_state.get("recent_urls", []),
                top_n=6,
            )
            support_plan = build_adaptive_plan(fused["label"])
            st.session_state["last_assessment"] = {
                "image_bytes": image_bytes,
                "modality_results": modality_results,
                "fused": fused,
                "ethical_report": ethical_report,
                "questions": questions,
                "recommendations": recommendations,
                "support_plan": support_plan,
                "user_id": user_id,
                "combined_context": combined_context,
                "transcription_note": transcription_note,
                "hf_username": hf_username,
            }
            st.session_state["recent_urls"] = (
                [item["url"] for item in recommendations] + st.session_state.get("recent_urls", [])
            )[:12]
            log_twin_event(
                user_id=user_id,
                fused_result=fused,
                modality_results=modality_results,
                ethical_report=ethical_report,
                question_source=questions["source"],
                recommendations=recommendations,
                context_text=combined_context,
            )

        last = st.session_state.get("last_assessment")
        if last:
            fused = last["fused"]
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            col1.metric("Detected emotion", fused["label"].title())
            col2.metric("Fusion confidence", f"{fused['confidence']:.2f}")
            col3.metric("Modality agreement", f"{fused['agreement']:.2f}")

            st.write("Fused probability distribution")
            st.dataframe(probability_rows(fused["distribution"]), use_container_width=True, hide_index=True)

            modality_rows = []
            for modality, result in last["modality_results"].items():
                if result:
                    modality_rows.append(
                        {
                            "modality": modality,
                            "emotion": result["label"],
                            "confidence": result["confidence"],
                            "method": result["method"],
                            "note": result["note"],
                        }
                    )
            st.write("Modality breakdown")
            st.dataframe(modality_rows, use_container_width=True, hide_index=True)

            if last["transcription_note"]:
                st.caption(last["transcription_note"])

            if last["image_bytes"]:
                st.image(last["image_bytes"], caption="Uploaded face image", width=260)
                image_result = last["modality_results"].get("image")
                if image_result and image_result["method"] != "heuristic-image":
                    if st.button("Generate Grad-CAM explanation"):
                        overlay, message = generate_gradcam_overlay(Image.open(io.BytesIO(last["image_bytes"])))
                        if overlay is not None:
                            st.image(overlay, caption=message)
                        else:
                            st.warning(message)
                else:
                    st.info("Grad-CAM becomes available after training and saving the EfficientNetV2-B0 emotion model.")

            st.subheader("Adaptive lifestyle plan")
            for item in last["support_plan"]:
                st.markdown(f"- {item}")

            st.subheader("Reflective questions")
            for index, question in enumerate(last["questions"]["questions"], start=1):
                st.markdown(f"{index}. {question}")
            st.caption(f"Question source: {last['questions']['source']}")

            st.subheader("History-aware recommendations")
            for item in last["recommendations"]:
                st.markdown(
                    f"- [{item['title']}]({item['url']})  \n"
                    f"  Source: {item['source']} | Offline fallback: {item['offline_fallback']}"
                )

            st.subheader("Ethical AI monitor")
            if last["ethical_report"]["flags"]:
                for flag in last["ethical_report"]["flags"]:
                    st.markdown(f"- {flag}")
            else:
                st.markdown("- No major warning flags were triggered for this run.")
            for safeguard in last["ethical_report"]["safeguards"]:
                st.caption(safeguard)

            with st.form("feedback_form"):
                reward = st.slider("Rate the usefulness of the recommendations", min_value=1, max_value=5, value=3)
                feedback_note = st.text_input("Optional feedback note")
                feedback_submitted = st.form_submit_button("Save recommender feedback")
            if feedback_submitted:
                record_feedback(
                    user_id=last["user_id"],
                    emotion=last["fused"]["label"],
                    recommendations=last["recommendations"],
                    reward=reward,
                    feedback_note=feedback_note,
                )
                st.success("Feedback saved to recommender_stats.csv and local bandit state.")

    with tab_training:
        st.subheader("Automatic dataset preparation from Hugging Face")
        search_query = st.text_input("Search query", value=st.session_state.get("dataset_search_query", "facial emotion"))
        st.session_state["dataset_search_query"] = search_query
        result_limit = st.selectbox("Result limit", [5, 10, 15], index=1)
        if st.button("Search Hugging Face datasets"):
            try:
                st.session_state["dataset_results"] = search_hf_datasets(search_query, limit=result_limit)
            except Exception as exc:
                st.error(f"Dataset search failed: {exc}")

        if st.session_state.get("dataset_results"):
            st.dataframe(st.session_state["dataset_results"], use_container_width=True, hide_index=True)

        dataset_id = st.text_input("Dataset ID", value=st.session_state.get("dataset_id", ""))
        sample_size = st.slider("Sample size for low-RAM demo training", min_value=600, max_value=1200, value=800, step=100)
        image_size = st.selectbox("Training image size", [160, 192, 224], index=2)

        if st.button("Prepare sampled dataset"):
            if not dataset_id.strip():
                st.warning("Enter a Hugging Face dataset ID first.")
            else:
                with st.spinner("Downloading, sampling, and restructuring dataset..."):
                    try:
                        manifest = prepare_sampled_hf_dataset(dataset_id.strip(), sample_size=sample_size, image_size=image_size)
                        st.session_state["dataset_id"] = dataset_id.strip()
                        st.success("Dataset prepared successfully under dataset/train, dataset/val, and dataset/test.")
                        st.json(manifest)
                    except Exception as exc:
                        st.error(f"Dataset preparation failed: {exc}")

        manual_zip = st.file_uploader("Manual ZIP import for approved local datasets", type=["zip"], key="manual_zip_uploader")
        if manual_zip is not None and st.button("Extract uploaded ZIP"):
            try:
                result = extract_manual_zip(manual_zip.getvalue())
                st.success(f"ZIP extracted to {result['target']} with {result['file_count']} files.")
            except Exception as exc:
                st.error(f"ZIP extraction failed: {exc}")

        if DATASET_MANIFEST_PATH.exists():
            st.write("Latest dataset manifest")
            st.json(load_json(DATASET_MANIFEST_PATH, {}))

        st.markdown("---")
        st.subheader("Train EfficientNetV2-B0 + GlobalAveragePooling + Grad-CAM")
        epochs = st.selectbox("Epochs", [1, 2, 3], index=1)
        batch_size = st.selectbox("Batch size", [4, 8], index=1)
        learning_rate = st.selectbox("Learning rate", [1e-3, 5e-4, 1e-4], index=2, format_func=lambda value: f"{value:.0e}")
        unfrozen_layers = st.selectbox("Unfrozen EfficientNetV2 layers", [20, 40, 60], index=1)
        if st.button("Train and evaluate model"):
            with st.spinner("Training EfficientNetV2-B0 on the prepared dataset..."):
                try:
                    training_results = train_emotion_model(
                        epochs=epochs,
                        batch_size=batch_size,
                        image_size=image_size,
                        learning_rate=learning_rate,
                        unfrozen_layers=unfrozen_layers,
                    )
                    st.session_state["training_results"] = training_results
                    st.success("Model trained and saved to models/efficientnetv2_emotion.keras")
                except Exception as exc:
                    st.error(f"Training failed: {exc}")

        if st.session_state.get("training_results"):
            training_results = st.session_state["training_results"]
            metrics = training_results["metrics"]
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
            col2.metric("Precision", f"{metrics['precision']:.3f}")
            col3.metric("Recall", f"{metrics['recall']:.3f}")
            col4.metric("F1", f"{metrics['f1']:.3f}")
            col5.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}" if metrics["roc_auc"] is not None else "N/A")
            st.caption(f"Evaluation split: {metrics['evaluation_split']}")

            history_figure = plot_history(training_results["history"])
            st.pyplot(history_figure)
            confusion_figure = plot_confusion(metrics["confusion_matrix"], training_results["class_names"])
            st.pyplot(confusion_figure)

    with tab_catalog:
        st.subheader("Structured resource catalog")
        catalog = generate_resource_catalog()
        mood_filter = st.selectbox("Filter by mood", ["all"] + EMOTION_LABELS, index=0)
        if mood_filter == "all":
            filtered_catalog = catalog
        else:
            filtered_catalog = [item for item in catalog if item["mood"] == mood_filter]
        st.caption(f"Catalog size: {len(catalog)} total entries.")
        st.dataframe(filtered_catalog[:80], use_container_width=True, hide_index=True)

        if st.button("Export resource catalog CSV"):
            path, count = export_resource_catalog()
            st.success(f"Exported {count} catalog entries to {path.name}")

        if RESOURCE_CATALOG_PATH.exists():
            with RESOURCE_CATALOG_PATH.open("rb") as handle:
                st.download_button(
                    "Download exported catalog",
                    data=handle.read(),
                    file_name=RESOURCE_CATALOG_PATH.name,
                    mime="text/csv",
                )
        st.code("python app.py --export-catalog", language="bash")

    with tab_logs:
        st.subheader("Digital Emotional Twin log")
        twin_rows = read_csv_rows(CEI_LOG_PATH, limit=25)
        if twin_rows:
            st.dataframe(twin_rows, use_container_width=True, hide_index=True)
        else:
            st.info("No twin log entries yet. Run an assessment first.")

        st.subheader("Recommender feedback log")
        recommender_rows = read_csv_rows(RECOMMENDER_STATS_PATH, limit=25)
        if recommender_rows:
            st.dataframe(recommender_rows, use_container_width=True, hide_index=True)
        else:
            st.info("No recommender feedback has been saved yet.")

        st.subheader("Local reinforcement-learning state")
        bandit_state = load_bandit_state()
        if bandit_state:
            preview_rows = []
            for key, value in list(bandit_state.items())[:20]:
                preview_rows.append({"key": key, "q": value.get("q"), "n": value.get("n"), "title": value.get("title")})
            st.dataframe(preview_rows, use_container_width=True, hide_index=True)
        else:
            st.info("Bandit state will appear after saving recommendation feedback.")

    with tab_help:
        st.subheader("Step-by-step execution guide")
        st.code(app_help_text(), language="text")
        st.markdown(
            """
            Key design choices
            - EfficientNetV2-B0 replaces MobileNetV2 for a more modern CNN backbone.
            - GlobalAveragePooling2D keeps the classifier head lightweight and Grad-CAM friendly.
            - Paid OpenAI and Spotify APIs are optional but not required. The free default path is Hugging Face plus platform search links.
            - The app does not generate Hugging Face tokens from username/password because secure token-based authentication is the correct approach.
            """
        )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cognitive Emotion Intelligence & Adaptive Lifestyle System")
    parser.add_argument("--export-catalog", action="store_true", help="Export the generated resource catalog to CSV.")
    parser.add_argument(
        "--bootstrap-project",
        action="store_true",
        help="Create runtime folders and starter CSV files for the project.",
    )
    args, _ = parser.parse_known_args(argv)
    return args


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    if args.bootstrap_project:
        result = bootstrap_project()
        print(json.dumps(result, indent=2))
        return
    if args.export_catalog:
        path, count = export_resource_catalog()
        print(f"Exported {count} catalog items to {path}")
        return
    try:
        run_streamlit_app()
    except ImportError as exc:
        print("Install the dependencies from requirements.txt and launch the app with: streamlit run app.py")
        print(f"Missing dependency: {exc}")


if __name__ == "__main__":
    main()
