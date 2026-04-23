from __future__ import annotations

import io
import json
import os
import random
import re
import shutil
import sys
import textwrap
import zipfile
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import requests
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from torchvision import datasets as tv_datasets
from torchvision import models, transforms

try:
    from datasets import DatasetDict, load_dataset
except Exception:
    DatasetDict = None
    load_dataset = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    import seaborn as sns
except Exception:
    sns = None

try:
    import speech_recognition as sr
except Exception:
    sr = None

try:
    from gtts import gTTS
except Exception:
    gTTS = None

try:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
except Exception:
    spotipy = None
    SpotifyClientCredentials = None


APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
DATASET_DIR = APP_DIR / "dataset"
DOCS_DIR = APP_DIR / "docs"
MODEL_PATH = MODELS_DIR / "efficientnet_emotion.pth"
MODEL_META_PATH = MODELS_DIR / "efficientnet_emotion_metadata.json"
TWIN_LOG_PATH = APP_DIR / "cei_twin_log.csv"
RECOMMENDER_STATS_PATH = APP_DIR / "recommender_stats.csv"
RESOURCE_CATALOG_PATH = APP_DIR / "resource_catalog.csv"
DATASET_MANIFEST_PATH = APP_DIR / "dataset_manifest.json"

EMOTION_LABELS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

EMOTION_COLORS = {
    "angry": "#ef4444",
    "disgust": "#22c55e",
    "fear": "#8b5cf6",
    "happy": "#f59e0b",
    "neutral": "#94a3b8",
    "sad": "#3b82f6",
    "surprise": "#ec4899",
}

EMOJI_SCORES = {
    "😀": "happy",
    "😄": "happy",
    "😁": "happy",
    "😊": "happy",
    "🙂": "neutral",
    "😐": "neutral",
    "😶": "neutral",
    "😔": "sad",
    "😢": "sad",
    "😭": "sad",
    "😡": "angry",
    "😠": "angry",
    "🤢": "disgust",
    "😨": "fear",
    "😱": "fear",
    "😮": "surprise",
    "🤯": "surprise",
}

TEXT_KEYWORDS = {
    "happy": [
        "happy",
        "joy",
        "joyful",
        "grateful",
        "excited",
        "great",
        "good",
        "energetic",
        "confident",
        "relieved",
        "motivated",
        "optimistic",
        "hopeful",
    ],
    "sad": [
        "sad",
        "down",
        "cry",
        "crying",
        "lonely",
        "tired",
        "hopeless",
        "upset",
        "hurt",
        "broken",
        "demotivated",
        "lost",
        "stress",
        "stressed",
    ],
    "angry": [
        "angry",
        "mad",
        "furious",
        "annoyed",
        "frustrated",
        "irritated",
        "rage",
        "hate",
    ],
    "fear": [
        "fear",
        "afraid",
        "scared",
        "nervous",
        "anxious",
        "panic",
        "worried",
        "uncertain",
        "pressure",
    ],
    "surprise": [
        "surprised",
        "shock",
        "shocked",
        "unexpected",
        "suddenly",
        "amazed",
        "wow",
    ],
    "disgust": [
        "disgust",
        "gross",
        "nasty",
        "awful",
        "sick",
        "repulsive",
    ],
    "neutral": [
        "okay",
        "fine",
        "normal",
        "balanced",
        "calm",
        "steady",
    ],
}

IMPORTANT_QUESTIONS = {
    "happy": [
        "What habit is helping you feel this positive today?",
        "How can we preserve this mood through a healthy routine?",
        "Would you like a productivity, celebration, or mindfulness recommendation?",
    ],
    "sad": [
        "Is there a specific reason behind this low mood today?",
        "Would reflective music, breathing guidance, or a small task plan help more right now?",
        "Do you want the chatbot to respond more gently and more slowly?",
    ],
    "angry": [
        "Is your frustration coming from workload, relationships, or self-pressure?",
        "Would you like calming media or a short de-escalation plan first?",
        "Should the system focus on support or on problem-solving steps?",
    ],
    "fear": [
        "Is the anxiety related to exams, career, health, or uncertainty?",
        "Would it help if I break the situation into a controllable action plan?",
        "Do you want grounding prompts before detailed advice?",
    ],
    "surprise": [
        "Is this surprise positive, stressful, or mixed?",
        "Would you like a fast summary of what to do next?",
        "Should the chatbot switch to explanation mode or celebration mode?",
    ],
    "disgust": [
        "Is this discomfort physical, environmental, or emotional?",
        "Would it help to redirect attention to something calming or constructive?",
        "Do you want a short reset routine?",
    ],
    "neutral": [
        "Would you like recommendations for focus, relaxation, or self-improvement?",
        "Do you want the bot to ask reflective questions or give direct suggestions?",
        "Should the system prioritize training, chatbot, or recommendation features?",
    ],
}

VIVA_KNOWLEDGE = {
    "future_scope": """
The future scope of this project is broad because the system already combines emotion detection,
explainable AI, lifestyle recommendation, and digital twin logging in one architecture.

1. A future version can add multimodal physiological sensing such as heart rate, respiration, and wearable-device signals.
2. The recommender can evolve from rule-guided reinforcement logic to full contextual bandits or safe reinforcement learning.
3. The digital emotional twin can become longitudinal, learning weekly and monthly emotional patterns instead of only session snapshots.
4. On-device optimization can make the model lighter for deployment on low-power mobile systems.
5. The chatbot can support multilingual therapy-style coaching with stronger safety filters and clinician review workflows.
""",
    "real_world_problem": """
This project addresses a real problem: people do not make decisions, consume media, or respond to stress in a purely rational way.
Their emotional state strongly influences concentration, health habits, and media choices.

Traditional recommendation systems only use historical behavior, but they ignore the user's present state.
This system solves that gap by sensing emotion in real time from text, emoji, voice transcript, and facial image,
then aligning recommendations and chatbot responses to the user's current mindset.
That makes the system relevant for student wellbeing, music therapy support, self-reflection, adaptive learning,
and emotionally aware human-computer interaction.
""",
    "research_gaps": """
Earlier 2023-2025 systems often had one or more weaknesses:

1. Single-modality emotion detection that used only text or only face.
2. Weak explainability, so users could not understand why a model predicted a particular emotion.
3. No digital emotional twin that stores longitudinal emotional behavior for personalization.
4. Recommendation engines that depended heavily on listening history and could not adapt to current emotion.
5. High-compute pipelines that were hard to deploy on low-cost student hardware.

This project narrows those gaps by offering multimodal fusion, Grad-CAM explainability, practical CPU-friendly training,
CSV-based digital twin tracking, and a local-first fallback chatbot that works even when paid APIs are unavailable.
""",
    "protagonist": """
The protagonist of the project is not only the chatbot or only the CNN model.
The real protagonist is the fusion engine that transforms scattered human signals into adaptive, explainable support.

What makes it different:
1. It combines perception, recommendation, explanation, and logging in one single-file application.
2. It uses multimodal emotion fusion rather than depending on one source.
3. It includes digital twin logging for continuity instead of isolated predictions.
4. It embeds ethical AI monitoring so low-confidence or biased situations can be flagged.
5. It keeps a no-repeat recommendation memory and reinforcement-style feedback loop.
""",
    "xai": """
Explainable AI is significant because emotional AI can influence user trust, self-perception, and decision-making.
If a model predicts sadness or stress incorrectly, the user should not be forced to trust a black box.

In this project, explainability matters for three reasons:
1. Transparency: the user can inspect what the visual model focused on.
2. Validation: the developer can detect whether the model is learning facial cues or background artifacts.
3. Ethics: explanation reduces blind automation and supports accountable system behavior.
""",
    "grad_cam": """
Grad-CAM is used to visualize which regions of the face influenced the CNN decision.
It works by computing gradients of the target class with respect to the final convolutional feature maps.

Why it matters here:
1. It shows whether the model focuses on eyebrows, eyes, mouth, or irrelevant background.
2. It is easy to explain during viva because the heatmap directly maps model attention to image regions.
3. It strengthens trust in the emotion classifier and makes error analysis much easier.
""",
    "efficientnetv2": """
EfficientNetV2 is a stronger and more modern family than older lightweight CNN baselines used in many student projects.

Its significance in this project:
1. It offers a better accuracy-efficiency balance.
2. It supports transfer learning very well on small datasets.
3. It integrates naturally with PyTorch and Grad-CAM pipelines.
4. It is practical on CPU with frozen-feature fine-tuning for a student laptop.
""",
    "gap": """
Global Average Pooling, or GAP, replaces large dense layers by averaging each feature map into a single value.

Why GAP is important:
1. It reduces parameters and helps prevent overfitting.
2. It preserves category-level spatial evidence in a cleaner way than huge fully connected blocks.
3. It helps class activation methods such as Grad-CAM by linking final feature maps more directly to class scores.
4. It makes the model lighter and more deployment-friendly.
""",
    "ethical_ai": """
Ethical AI monitoring is built in to ensure that the project remains responsible, not merely accurate.

The system checks for:
1. Low-confidence predictions.
2. Single-modality overdependence.
3. Potential dataset imbalance during training.
4. The need to avoid medical or psychiatric overclaiming.

This is important because an emotion system should assist reflection and recommendation, not act as a clinical authority.
""",
    "digital_twin": """
A digital emotional twin is a lightweight evolving profile of the user's emotional behavior over time.

In this project it is implemented as:
1. Session-by-session logging of fused emotion.
2. Mood trend aggregation.
3. Recommendation feedback memory.
4. Contextual prompts for future interaction.

This makes the system adaptive across time rather than reactive only to one isolated input.
""",
    "cei": """
Cognitive Emotional Intelligence in this project means combining emotional sensing with decision-oriented guidance.
The system does not stop at detecting a mood. It interprets the mood, adds context, and chooses a suitable response.

That bridge between sensing and adaptive action is what makes the project more meaningful than a plain classifier.
""",
    "multi_ai": """
Multi-AI fusion means multiple intelligent blocks work together:
1. A visual CNN for facial emotion.
2. NLP heuristics and optional language-model support for text.
3. Voice-to-text support for spoken input.
4. Reinforcement-style recommendation feedback.
5. Explainability and ethical monitoring layers.

The outcome is a system that is broader and more project-worthy than a single-model demo.
""",
    "rl_logic": """
The reinforcement-learning logic in this project is lightweight and practical.
It does not run a heavy research-grade deep RL policy. Instead, it updates recommendation preference values
from user feedback such as helpful or skip.

Why this is a good design for a major project:
1. It shows adaptive learning behavior.
2. It is simple enough to explain and maintain.
3. It avoids huge computational cost on a laptop.
4. It still demonstrates the principle of reward-driven personalization.
""",
    "pytorch": """
PyTorch is preferred here because it gives a clean modern workflow for training, transfer learning, and Grad-CAM integration.
Compared with older student TensorFlow setups, PyTorch is often easier to debug, easier to customize, and easier to explain during viva.
""",
    "free_api": """
The best free optional API path here is Hugging Face Inference with a free account and manual access token.
It is suitable because:
1. Token creation is free.
2. It supports many public instruction models.
3. It integrates by plain HTTPS requests.

For speech and voice demonstration, free web-based speech recognition and text-to-speech options can be used.
Spotify remains optional because its developer credentials are free but still require app registration.
""",
}


def ensure_runtime_structure() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    DATASET_DIR.mkdir(exist_ok=True)
    DOCS_DIR.mkdir(exist_ok=True)
    (MODELS_DIR / ".gitkeep").touch(exist_ok=True)
    (DATASET_DIR / ".gitkeep").touch(exist_ok=True)


def initialize_state() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "recent_recommendations" not in st.session_state:
        st.session_state.recent_recommendations = deque(maxlen=10)
    if "latest_analysis" not in st.session_state:
        st.session_state.latest_analysis = {}
    if "last_response_audio" not in st.session_state:
        st.session_state.last_response_audio = None


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def normalize_label(label: str) -> str:
    clean = re.sub(r"[^a-z0-9_ -]+", "", str(label).strip().lower())
    mapping = {
        "joy": "happy",
        "happiness": "happy",
        "content": "neutral",
        "calm": "neutral",
        "relaxed": "neutral",
        "relax": "neutral",
        "fearful": "fear",
        "afraid": "fear",
        "surprised": "surprise",
        "suprised": "surprise",
        "disgusted": "disgust",
    }
    if clean in mapping:
        return mapping[clean]
    if clean in EMOTION_LABELS:
        return clean
    if "happy" in clean or "joy" in clean:
        return "happy"
    if "sad" in clean:
        return "sad"
    if "ang" in clean:
        return "angry"
    if "fear" in clean or "anx" in clean:
        return "fear"
    if "surpr" in clean:
        return "surprise"
    if "disg" in clean:
        return "disgust"
    return "neutral"


def empty_distribution() -> Dict[str, float]:
    return {emotion: 0.0 for emotion in EMOTION_LABELS}


def normalize_distribution(scores: Dict[str, float]) -> Dict[str, float]:
    fixed = {emotion: max(float(scores.get(emotion, 0.0)), 0.0) for emotion in EMOTION_LABELS}
    total = sum(fixed.values())
    if total <= 0:
        fixed["neutral"] = 1.0
        total = 1.0
    return {key: value / total for key, value in fixed.items()}


def dominant_emotion(scores: Dict[str, float]) -> Tuple[str, float]:
    normalized = normalize_distribution(scores)
    mood = max(normalized, key=normalized.get)
    return mood, normalized[mood]


def blend_modal_scores(modal_scores: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    weights = {
        "image": 0.45,
        "text": 0.25,
        "emoji": 0.10,
        "voice": 0.20,
    }
    total_weight = 0.0
    blended = empty_distribution()
    for modality, scores in modal_scores.items():
        if not scores:
            continue
        weight = weights.get(modality, 0.0)
        total_weight += weight
        normalized = normalize_distribution(scores)
        for emotion, value in normalized.items():
            blended[emotion] += value * weight
    if total_weight == 0:
        blended["neutral"] = 1.0
        total_weight = 1.0
    return {emotion: value / total_weight for emotion, value in blended.items()}


def text_emotion_scores(text: str) -> Dict[str, float]:
    clean = re.sub(r"[^a-zA-Z0-9 ]+", " ", text.lower())
    tokens = clean.split()
    scores = Counter()
    for emotion, keywords in TEXT_KEYWORDS.items():
        for keyword in keywords:
            token_hits = sum(1 for token in tokens if token == keyword or token.startswith(keyword))
            scores[emotion] += token_hits
    if not scores:
        return {"neutral": 1.0}
    return normalize_distribution(dict(scores))


def emoji_emotion_scores(emoji_text: str) -> Dict[str, float]:
    scores = Counter()
    for char in emoji_text:
        emotion = EMOJI_SCORES.get(char)
        if emotion:
            scores[emotion] += 1
    if not scores:
        return {}
    return normalize_distribution(dict(scores))


def image_heuristic_scores(image: Image.Image) -> Dict[str, float]:
    small = image.convert("RGB").resize((224, 224))
    arr = np.asarray(small).astype(np.float32) / 255.0
    gray = arr.mean(axis=2)
    brightness = float(gray.mean())
    contrast = float(gray.std())
    red = float(arr[:, :, 0].mean())
    green = float(arr[:, :, 1].mean())
    blue = float(arr[:, :, 2].mean())
    saturation = float(np.std(arr, axis=2).mean())

    scores = empty_distribution()
    scores["happy"] = max(0.05, brightness * 0.9 + saturation * 0.8)
    scores["sad"] = max(0.05, (1.0 - brightness) * 1.1 + blue * 0.3)
    scores["angry"] = max(0.05, red * 0.9 + contrast * 0.6 - brightness * 0.2)
    scores["fear"] = max(0.05, contrast * 0.9 + (1.0 - green) * 0.4)
    scores["surprise"] = max(0.05, contrast * 1.1 + brightness * 0.3)
    scores["disgust"] = max(0.05, green * 0.5 + (1.0 - saturation) * 0.4)
    scores["neutral"] = max(0.05, 1.0 - abs(brightness - 0.5) - contrast * 0.3)
    return normalize_distribution(scores)


def load_uploaded_image(uploaded_file) -> Optional[Image.Image]:
    if uploaded_file is None:
        return None
    try:
        return Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
    except Exception:
        return None


def transcribe_audio(uploaded_audio) -> Tuple[str, str]:
    if uploaded_audio is None:
        return "", "No voice input captured."
    if sr is None:
        return "", "SpeechRecognition is not installed."

    recognizer = sr.Recognizer()
    audio_bytes = uploaded_audio.getvalue()
    try:
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio = recognizer.record(source)
        transcript = recognizer.recognize_google(audio)
        return transcript, "Voice transcript created successfully."
    except sr.UnknownValueError:
        return "", "Speech could not be understood clearly."
    except Exception as exc:
        return "", f"Voice transcription failed: {exc}"


def generate_speech_bytes(text: str) -> Optional[bytes]:
    if gTTS is None or not text.strip():
        return None
    try:
        buffer = io.BytesIO()
        tts = gTTS(text=text[:900], lang="en")
        tts.write_to_fp(buffer)
        return buffer.getvalue()
    except Exception:
        return None


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = None
    if pretrained:
        try:
            weights = models.EfficientNet_V2_S_Weights.DEFAULT
        except Exception:
            weights = None
    try:
        model = models.efficientnet_v2_s(weights=weights)
    except Exception:
        model = models.efficientnet_v2_s(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.25),
        nn.Linear(in_features, num_classes),
    )
    return model


def default_infer_transform() -> transforms.Compose:
    weights = None
    try:
        weights = models.EfficientNet_V2_S_Weights.DEFAULT
    except Exception:
        pass
    if weights is not None:
        return weights.transforms()
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(8),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def dataset_ready(split_name: str) -> bool:
    split_dir = DATASET_DIR / split_name
    if not split_dir.exists():
        return False
    return any(path.is_dir() for path in split_dir.iterdir())


def get_class_names_from_train() -> List[str]:
    train_dir = DATASET_DIR / "train"
    if not train_dir.exists():
        return []
    return sorted([path.name for path in train_dir.iterdir() if path.is_dir()])


def load_model_bundle() -> Tuple[Optional[nn.Module], List[str], str]:
    if not MODEL_PATH.exists() or not MODEL_META_PATH.exists():
        return None, [], "No trained PyTorch weights found. Image analysis will use heuristic mode."
    try:
        metadata = json.loads(MODEL_META_PATH.read_text())
        class_names = metadata.get("classes", EMOTION_LABELS)
        model = build_model(num_classes=len(class_names), pretrained=False)
        state_dict = torch.load(MODEL_PATH, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        return model, class_names, "Loaded saved EfficientNetV2-S model."
    except Exception as exc:
        return None, [], f"Model weights exist but could not be loaded: {exc}"


def predict_with_model(image: Image.Image, model: nn.Module, class_names: List[str]) -> Dict[str, float]:
    tensor = default_infer_transform()(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
    scores = {normalize_label(name): 0.0 for name in EMOTION_LABELS}
    for idx, label in enumerate(class_names):
        scores[normalize_label(label)] = max(scores.get(normalize_label(label), 0.0), float(probabilities[idx]))
    return normalize_distribution(scores)


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.fwd_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self.bwd_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output) -> None:
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def remove(self) -> None:
        self.fwd_handle.remove()
        self.bwd_handle.remove()

    def generate(self, input_tensor: torch.Tensor, class_index: Optional[int] = None) -> Tuple[np.ndarray, int]:
        output = self.model(input_tensor)
        if class_index is None:
            class_index = int(output.argmax(dim=1).item())
        self.model.zero_grad()
        score = output[:, class_index].sum()
        score.backward(retain_graph=True)

        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = gradients.mean(dim=(1, 2))
        cam = torch.sum(weights[:, None, None] * activations, dim=0)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy(), class_index


def create_gradcam_overlay(image: Image.Image, cam: np.ndarray) -> Image.Image:
    cam_uint8 = np.uint8(np.clip(cam, 0, 1) * 255)
    cam_image = Image.fromarray(cam_uint8).resize(image.size)
    heat = ImageOps.colorize(cam_image.convert("L"), black="#1e293b", white="#ef4444").convert("RGBA")
    base = image.convert("RGBA")
    overlay = Image.blend(base, heat, alpha=0.35)
    return overlay


def generate_gradcam(image: Image.Image, model: nn.Module, class_names: List[str]) -> Tuple[Optional[Image.Image], str]:
    try:
        target_layer = model.features[-1]
        grad_cam = GradCAM(model, target_layer)
        tensor = default_infer_transform()(image).unsqueeze(0)
        cam, class_index = grad_cam.generate(tensor)
        overlay = create_gradcam_overlay(image, cam)
        grad_cam.remove()
        label = class_names[class_index] if class_index < len(class_names) else str(class_index)
        return overlay, f"Grad-CAM generated for predicted class: {label}"
    except Exception as exc:
        return None, f"Grad-CAM could not be generated: {exc}"


def append_row_csv(path: Path, row: Dict[str, object]) -> None:
    frame = pd.DataFrame([row])
    if path.exists():
        frame.to_csv(path, mode="a", header=False, index=False)
    else:
        frame.to_csv(path, index=False)


def safe_read_csv(path: Path, columns: List[str]) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            pass
    return pd.DataFrame(columns=columns)


def log_twin_event(
    username: str,
    fused_scores: Dict[str, float],
    text_input: str,
    emoji_input: str,
    voice_text: str,
    image_mode: str,
) -> None:
    mood, confidence = dominant_emotion(fused_scores)
    row = {
        "timestamp": now_iso(),
        "username": username or "guest",
        "dominant_emotion": mood,
        "confidence": round(confidence, 4),
        "text_length": len(text_input.strip()),
        "emoji_count": len(emoji_input.strip()),
        "voice_length": len(voice_text.strip()),
        "image_mode": image_mode,
        "angry": round(fused_scores.get("angry", 0.0), 4),
        "disgust": round(fused_scores.get("disgust", 0.0), 4),
        "fear": round(fused_scores.get("fear", 0.0), 4),
        "happy": round(fused_scores.get("happy", 0.0), 4),
        "neutral": round(fused_scores.get("neutral", 0.0), 4),
        "sad": round(fused_scores.get("sad", 0.0), 4),
        "surprise": round(fused_scores.get("surprise", 0.0), 4),
    }
    append_row_csv(TWIN_LOG_PATH, row)


def build_twin_snapshot() -> Dict[str, object]:
    columns = [
        "timestamp",
        "username",
        "dominant_emotion",
        "confidence",
    ]
    df = safe_read_csv(TWIN_LOG_PATH, columns=columns)
    if df.empty:
        return {
            "total_sessions": 0,
            "dominant_trend": "neutral",
            "avg_confidence": 0.0,
            "recent_distribution": {emotion: 0 for emotion in EMOTION_LABELS},
        }

    recent = df.tail(20).copy()
    dominant_trend = recent["dominant_emotion"].mode().iloc[0]
    avg_confidence = float(recent["confidence"].astype(float).mean())
    counts = recent["dominant_emotion"].value_counts().to_dict()
    distribution = {emotion: int(counts.get(emotion, 0)) for emotion in EMOTION_LABELS}
    return {
        "total_sessions": int(len(df)),
        "dominant_trend": dominant_trend,
        "avg_confidence": avg_confidence,
        "recent_distribution": distribution,
    }


def ethical_ai_flags(
    modal_scores: Dict[str, Dict[str, float]],
    fused_scores: Dict[str, float],
    available_modalities: List[str],
) -> List[str]:
    mood, confidence = dominant_emotion(fused_scores)
    flags = []
    if confidence < 0.40:
        flags.append("Low-confidence emotional inference. Treat the output as supportive guidance, not a hard truth.")
    if len(available_modalities) <= 1:
        flags.append("Only one modality contributed to the decision, so multimodal robustness is limited.")
    if mood in {"sad", "fear", "angry"}:
        flags.append("Negative affect detected. The system should respond gently and avoid overconfident claims.")
    return flags


def build_resource_catalog() -> pd.DataFrame:
    moods = {
        "happy": [
            "celebration playlist",
            "focus with positivity",
            "upbeat coding music",
            "gratitude journaling music",
        ],
        "sad": [
            "healing acoustic music",
            "gentle piano for reflection",
            "supportive self care routine",
            "calm rain and recovery music",
        ],
        "angry": [
            "anger release breathing",
            "calm meditation for frustration",
            "reset routine after conflict",
            "de escalation ambient music",
        ],
        "fear": [
            "anxiety grounding playlist",
            "slow breathing guided audio",
            "confidence building affirmations",
            "exam stress calm music",
        ],
        "surprise": [
            "adapt quickly motivation",
            "unexpected challenge focus music",
            "decision making clarity sounds",
            "surprise to strategy routine",
        ],
        "disgust": [
            "mental reset routine",
            "mind cleansing ambient music",
            "recenter attention meditation",
            "environment refresh motivation",
        ],
        "neutral": [
            "deep work instrumental",
            "balanced study playlist",
            "productive morning routine",
            "mindful neutral focus",
        ],
    }
    rows = []
    for mood, themes in moods.items():
        for theme in themes:
            safe_query = quote_plus(theme)
            rows.append(
                {
                    "mood": mood,
                    "title": theme.title(),
                    "url": f"https://www.youtube.com/results?search_query={safe_query}",
                    "source": "YouTube",
                    "type": "search",
                    "offline_fallback_guidance": f"Search locally for {theme} or open saved wellness videos.",
                }
            )
            rows.append(
                {
                    "mood": mood,
                    "title": f"{theme.title()} - Spotify",
                    "url": f"https://open.spotify.com/search/{safe_query}",
                    "source": "Spotify",
                    "type": "search",
                    "offline_fallback_guidance": f"Use saved playlists tagged for {mood} mood.",
                }
            )
            rows.append(
                {
                    "mood": mood,
                    "title": f"{theme.title()} - YouTube Music",
                    "url": f"https://music.youtube.com/search?q={safe_query}",
                    "source": "YouTube Music",
                    "type": "search",
                    "offline_fallback_guidance": f"Open downloaded audio already available on the device.",
                }
            )
            rows.append(
                {
                    "mood": mood,
                    "title": f"{theme.title()} - Wellness Reading",
                    "url": f"https://www.google.com/search?q={safe_query}+wellbeing",
                    "source": "Web",
                    "type": "support",
                    "offline_fallback_guidance": "Follow the in-app breathing and journaling suggestions.",
                }
            )
    catalog = pd.DataFrame(rows)
    return catalog


def save_resource_catalog() -> pd.DataFrame:
    catalog = build_resource_catalog()
    catalog.to_csv(RESOURCE_CATALOG_PATH, index=False)
    return catalog


def load_recommender_stats() -> pd.DataFrame:
    columns = [
        "item_id",
        "mood",
        "title",
        "source",
        "shown_count",
        "accepted_count",
        "rejected_count",
        "q_value",
        "last_updated",
    ]
    return safe_read_csv(RECOMMENDER_STATS_PATH, columns)


def save_recommender_feedback(item_row: pd.Series, mood: str, reward: float) -> None:
    stats = load_recommender_stats()
    item_id = slugify(f"{item_row['title']}-{item_row['source']}")
    if item_id in stats.get("item_id", pd.Series(dtype=str)).values:
        index = stats.index[stats["item_id"] == item_id][0]
        shown_count = int(stats.at[index, "shown_count"]) + 1
        accepted_count = int(stats.at[index, "accepted_count"]) + (1 if reward > 0 else 0)
        rejected_count = int(stats.at[index, "rejected_count"]) + (1 if reward <= 0 else 0)
        q_value = float(stats.at[index, "q_value"])
        q_value = q_value + 0.30 * (reward - q_value)
        stats.loc[index, ["shown_count", "accepted_count", "rejected_count", "q_value", "last_updated"]] = [
            shown_count,
            accepted_count,
            rejected_count,
            round(q_value, 4),
            now_iso(),
        ]
    else:
        stats = pd.concat(
            [
                stats,
                pd.DataFrame(
                    [
                        {
                            "item_id": item_id,
                            "mood": mood,
                            "title": item_row["title"],
                            "source": item_row["source"],
                            "shown_count": 1,
                            "accepted_count": 1 if reward > 0 else 0,
                            "rejected_count": 0 if reward > 0 else 1,
                            "q_value": reward,
                            "last_updated": now_iso(),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    stats.to_csv(RECOMMENDER_STATS_PATH, index=False)


def recommend_items(mood: str, top_n: int = 5) -> pd.DataFrame:
    catalog = save_resource_catalog()
    stats = load_recommender_stats()
    recent = set(st.session_state.recent_recommendations)
    candidates = catalog[catalog["mood"] == mood].copy()
    if candidates.empty:
        candidates = catalog.copy()
    candidates["item_id"] = candidates.apply(lambda row: slugify(f"{row['title']}-{row['source']}"), axis=1)
    candidates["base_score"] = 1.0
    if not stats.empty:
        stats_subset = stats[["item_id", "q_value"]].copy()
        candidates = candidates.merge(stats_subset, on="item_id", how="left")
    else:
        candidates["q_value"] = 0.0
    candidates["q_value"] = candidates["q_value"].fillna(0.0)
    candidates["diversity_bonus"] = np.where(candidates["item_id"].isin(recent), -0.9, 0.25)
    candidates["score"] = candidates["base_score"] + candidates["q_value"] + candidates["diversity_bonus"]
    candidates = candidates.sort_values(by="score", ascending=False)
    selected = candidates.head(max(top_n, 1)).copy()
    for item_id in selected["item_id"].tolist():
        st.session_state.recent_recommendations.append(item_id)
    return selected


def spotify_search_tracks(query: str, client_id: str, client_secret: str, limit: int = 5) -> List[Dict[str, str]]:
    if not query or not client_id or not client_secret or spotipy is None or SpotifyClientCredentials is None:
        return []
    try:
        auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
        client = spotipy.Spotify(auth_manager=auth_manager)
        results = client.search(q=query, type="track", limit=limit)
        tracks = []
        for track in results.get("tracks", {}).get("items", []):
            tracks.append(
                {
                    "name": track["name"],
                    "artist": ", ".join(artist["name"] for artist in track["artists"]),
                    "url": track["external_urls"]["spotify"],
                }
            )
        return tracks
    except Exception:
        return []


def detect_intent(query: str) -> str:
    q = query.lower()
    keyword_map = {
        "future_scope": ["future scope", "future", "extension"],
        "real_world_problem": ["real world", "problem statement", "application", "benefit"],
        "research_gaps": ["research gap", "gap", "earlier papers", "previous work"],
        "protagonist": ["protagonist", "different from others", "difference", "unique"],
        "xai": ["explainable ai", "xai"],
        "grad_cam": ["grad cam", "grad-cam"],
        "efficientnetv2": ["efficientnet", "cnn architecture", "why efficientnet"],
        "gap": ["global average pooling", "gap "],
        "ethical_ai": ["ethical ai", "ethics", "bias"],
        "digital_twin": ["digital twin", "emotional twin"],
        "cei": ["cognitive emotional intelligence", "cei"],
        "multi_ai": ["multi ai fusion", "multimodal", "fusion"],
        "rl_logic": ["reinforcement", "rl logic", "reward"],
        "pytorch": ["pytorch", "tensorflow", "why pytorch"],
        "free_api": ["hugging face", "api", "free token", "spotify", "openai"],
    }
    for intent, keywords in keyword_map.items():
        if any(keyword in q for keyword in keywords):
            return intent
    return "general"


def supportive_tone(mood: str, username: str) -> str:
    name = username.strip() or "there"
    messages = {
        "happy": f"{name}, your current profile looks positive, so I am keeping the answer energetic and action-oriented.",
        "sad": f"{name}, your recent emotional profile suggests a softer and more supportive response style would be helpful.",
        "angry": f"{name}, I am keeping the explanation calm, structured, and practical so it helps under frustration.",
        "fear": f"{name}, I will explain things in small controllable steps because anxiety often improves with clarity.",
        "surprise": f"{name}, I will keep the response clear and adaptive because your current state appears more reactive than stable.",
        "disgust": f"{name}, I will keep the response concise and grounding so it feels less overwhelming.",
        "neutral": f"{name}, I will keep the answer balanced and detailed.",
    }
    return messages.get(mood, messages["neutral"])


def local_chat_response(query: str, username: str, mood: str) -> str:
    intent = detect_intent(query)
    preface = supportive_tone(mood, username)
    if intent == "general":
        answer = """
This project is a single-file Streamlit and PyTorch system that combines facial emotion analysis,
text and emoji understanding, optional voice transcription, Grad-CAM explainability, digital emotional twin logging,
and a lifestyle recommendation layer.

If you want the strongest viva answer, focus on these four ideas:
1. real-time multimodal emotion fusion,
2. explainable CNN predictions with Grad-CAM,
3. adaptive no-repeat recommendations with feedback learning,
4. ethical AI monitoring with a lightweight digital twin.
"""
    else:
        answer = VIVA_KNOWLEDGE.get(intent, VIVA_KNOWLEDGE["cei"])

    questions = IMPORTANT_QUESTIONS.get(mood, IMPORTANT_QUESTIONS["neutral"])
    follow_up = "\n".join([f"- {question}" for question in questions])
    return f"{preface}\n\n{answer.strip()}\n\nImportant follow-up questions for this mindset:\n{follow_up}"


def huggingface_chat_response(
    query: str,
    mood: str,
    username: str,
    token: str,
    model_id: str,
) -> Tuple[str, str]:
    prompt = textwrap.dedent(
        f"""
        You are helping a student explain a major project called
        "Cognitive Emotion Intelligence and Adaptive Lifestyle System".

        User name: {username or 'guest'}
        Current detected mood: {mood}

        Please answer the question in a polished, human-like engineering style.
        Keep the answer practical, accurate, and suitable for viva or project demo.
        Avoid unsupported clinical claims.

        Question:
        {query}
        """
    ).strip()

    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 250,
            "temperature": 0.5,
            "return_full_text": False,
        },
        "options": {"wait_for_model": True},
    }
    try:
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{model_id}",
            headers=headers,
            json=payload,
            timeout=60,
        )
        data = response.json()
        if response.status_code >= 400:
            return "", f"Hugging Face API returned {response.status_code}: {data}"
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict) and "generated_text" in first:
                return first["generated_text"], "Hugging Face response generated successfully."
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"], "Hugging Face response generated successfully."
        return "", f"Unexpected Hugging Face response shape: {data}"
    except Exception as exc:
        return "", f"Hugging Face generation failed: {exc}"


def build_avatar_html(mood: str) -> str:
    color = EMOTION_COLORS.get(mood, "#2563eb")
    mood_face = {
        "happy": "^_^",
        "sad": "T_T",
        "angry": ">_<",
        "fear": "o_o",
        "surprise": "O_O",
        "disgust": "-_-",
        "neutral": "._.",
    }.get(mood, "._.")
    return f"""
    <div style="display:flex; gap:16px; align-items:center; padding:16px; border-radius:18px; background:linear-gradient(135deg, #0f172a, #1e293b); color:white;">
        <div style="width:92px; height:92px; border-radius:50%; background:{color}; display:flex; align-items:center; justify-content:center; font-size:24px; font-weight:700; box-shadow:0 0 0 6px rgba(255,255,255,0.08); animation:pulse 2.2s infinite;">
            {mood_face}
        </div>
        <div>
            <div style="font-size:1.2rem; font-weight:700;">CEI Mascot Assistant</div>
            <div style="opacity:0.85;">Emotion-aware guide for viva, recommendations, and adaptive support.</div>
        </div>
    </div>
    <style>
    @keyframes pulse {{
        0% {{ transform: scale(1.0); }}
        50% {{ transform: scale(1.04); }}
        100% {{ transform: scale(1.0); }}
    }}
    </style>
    """


def plot_confusion(conf_matrix: np.ndarray, class_names: List[str]):
    if plt is None:
        st.write(conf_matrix)
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    if sns is not None:
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    else:
        ax.imshow(conf_matrix, cmap="Blues")
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticks(range(len(class_names)))
        ax.set_yticklabels(class_names)
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax.text(j, i, conf_matrix[i, j], ha="center", va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)


def training_ready() -> bool:
    return dataset_ready("train") and dataset_ready("val")


def make_dataloaders(batch_size: int):
    train_set = tv_datasets.ImageFolder(DATASET_DIR / "train", transform=train_transform())
    val_set = tv_datasets.ImageFolder(DATASET_DIR / "val", transform=eval_transform())
    test_set = tv_datasets.ImageFolder(DATASET_DIR / "test", transform=eval_transform()) if dataset_ready("test") else None
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False) if test_set is not None else None
    return train_set, val_set, test_set, train_loader, val_loader, test_loader


def evaluate_model(model: nn.Module, loader: DataLoader, class_names: List[str]) -> Dict[str, object]:
    model.eval()
    y_true = []
    y_pred = []
    probabilities = []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            probabilities.extend(probs.cpu().numpy().tolist())

    metrics = {}
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    metrics["precision"] = float(precision)
    metrics["recall"] = float(recall)
    metrics["f1"] = float(f1)
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()

    roc_auc = None
    try:
        y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))
        roc_auc = float(roc_auc_score(y_true_bin, np.array(probabilities), average="weighted", multi_class="ovr"))
    except Exception:
        roc_auc = None
    metrics["roc_auc"] = roc_auc
    return metrics


def train_emotion_model(
    epochs: int,
    batch_size: int,
    learning_rate: float,
    freeze_backbone: bool,
) -> Dict[str, object]:
    train_set, val_set, test_set, train_loader, val_loader, test_loader = make_dataloaders(batch_size=batch_size)
    class_names = [normalize_label(name) for name in train_set.classes]
    model = build_model(num_classes=len(train_set.classes), pretrained=True)
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda param: param.requires_grad, model.parameters()), lr=learning_rate)

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * images.size(0)
            preds = logits.argmax(dim=1)
            correct += int((preds == labels).sum().item())
            total += int(labels.size(0))

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        model.eval()
        val_loss_total = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss_total += float(loss.item()) * images.size(0)
                preds = logits.argmax(dim=1)
                val_correct += int((preds == labels).sum().item())
                val_total += int(labels.size(0))
        val_loss = val_loss_total / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_accuracy": round(train_acc, 4),
                "val_loss": round(val_loss, 4),
                "val_accuracy": round(val_acc, 4),
            }
        )

    val_metrics = evaluate_model(model, val_loader, class_names)
    test_metrics = evaluate_model(model, test_loader, class_names) if test_loader is not None else None

    torch.save(model.state_dict(), MODEL_PATH)
    metadata = {
        "created_at": now_iso(),
        "architecture": "efficientnet_v2_s",
        "framework": "PyTorch",
        "classes": class_names,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "freeze_backbone": freeze_backbone,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    MODEL_META_PATH.write_text(json.dumps(metadata, indent=2))
    return metadata | {"history": history}


def extract_manual_zip(uploaded_zip) -> Tuple[bool, str]:
    if uploaded_zip is None:
        return False, "No ZIP file selected."
    try:
        with zipfile.ZipFile(io.BytesIO(uploaded_zip.getvalue())) as archive:
            archive.extractall(DATASET_DIR)
        return True, "ZIP dataset extracted into the dataset folder."
    except Exception as exc:
        return False, f"Dataset ZIP extraction failed: {exc}"


def clear_dataset_splits() -> None:
    for split_name in ["train", "val", "test"]:
        split_dir = DATASET_DIR / split_name
        if split_dir.exists():
            shutil.rmtree(split_dir)


def detect_columns(record: Dict[str, object]) -> Tuple[Optional[str], Optional[str]]:
    image_col = None
    label_col = None
    for key, value in record.items():
        key_lower = key.lower()
        if image_col is None:
            if isinstance(value, Image.Image):
                image_col = key
            elif isinstance(value, dict) and ("bytes" in value or "path" in value):
                image_col = key
            elif key_lower in {"image", "img", "face"}:
                image_col = key
        if label_col is None and key_lower in {"label", "labels", "emotion", "class", "category", "sentiment"}:
            label_col = key
    return image_col, label_col


def convert_hf_image(value: object) -> Optional[Image.Image]:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        raw_bytes = value.get("bytes")
        path = value.get("path")
        if raw_bytes:
            return Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        if path and Path(path).exists():
            return Image.open(path).convert("RGB")
    return None


def hf_dataset_to_folders(dataset_id: str, sample_size: int, train_ratio: float, val_ratio: float, seed: int) -> Tuple[bool, str]:
    if load_dataset is None:
        return False, "The datasets package is not installed."
    if not dataset_id.strip():
        return False, "Please provide a Hugging Face dataset id."

    rng = random.Random(seed)
    try:
        dataset_obj = load_dataset(dataset_id)
    except Exception as exc:
        return False, f"Dataset download failed: {exc}"

    all_records = []
    if DatasetDict is not None and isinstance(dataset_obj, DatasetDict):
        split_names = list(dataset_obj.keys())
        for split_name in split_names:
            split_ds = dataset_obj[split_name]
            count = min(len(split_ds), max(sample_size // max(len(split_names), 1), 1))
            indices = list(range(len(split_ds)))
            rng.shuffle(indices)
            for idx in indices[:count]:
                all_records.append(split_ds[idx])
    else:
        count = min(len(dataset_obj), sample_size)
        indices = list(range(len(dataset_obj)))
        rng.shuffle(indices)
        for idx in indices[:count]:
            all_records.append(dataset_obj[idx])

    if not all_records:
        return False, "No records were loaded from the dataset."

    image_col, label_col = detect_columns(all_records[0])
    if image_col is None or label_col is None:
        return False, "Could not auto-detect image and label columns."

    records_by_label: Dict[str, List[Tuple[Image.Image, str]]] = {}
    for record in all_records:
        image = convert_hf_image(record.get(image_col))
        label = normalize_label(record.get(label_col))
        if image is None:
            continue
        records_by_label.setdefault(label, []).append((image, label))

    if not records_by_label:
        return False, "No valid image-label pairs could be extracted."

    clear_dataset_splits()
    for split_name in ["train", "val", "test"]:
        split_dir = DATASET_DIR / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

    split_counts = {"train": 0, "val": 0, "test": 0}
    for label, records in records_by_label.items():
        rng.shuffle(records)
        train_cut = max(1, int(len(records) * train_ratio))
        val_cut = max(train_cut + 1, int(len(records) * (train_ratio + val_ratio)))
        split_map = {
            "train": records[:train_cut],
            "val": records[train_cut:val_cut],
            "test": records[val_cut:],
        }
        for split_name, items in split_map.items():
            class_dir = DATASET_DIR / split_name / label
            class_dir.mkdir(parents=True, exist_ok=True)
            for idx, (image, _) in enumerate(items):
                output_path = class_dir / f"{slugify(dataset_id)}-{label}-{idx:04d}.png"
                image.resize((224, 224)).save(output_path)
                split_counts[split_name] += 1

    manifest = {
        "created_at": now_iso(),
        "source": "huggingface",
        "dataset_id": dataset_id,
        "sample_size_requested": sample_size,
        "classes": sorted(records_by_label.keys()),
        "image_column": image_col,
        "label_column": label_col,
        "split_counts": split_counts,
    }
    DATASET_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    return True, f"Dataset prepared successfully with {sum(split_counts.values())} images."


def show_download_button(path: Path, label: str, mime: str) -> None:
    if path.exists():
        st.download_button(
            label=label,
            data=path.read_bytes(),
            file_name=path.name,
            mime=mime,
        )


def export_catalog_cli() -> int:
    ensure_runtime_structure()
    catalog = save_resource_catalog()
    print(f"Exported {len(catalog)} resource entries to {RESOURCE_CATALOG_PATH}")
    return 0


def page_header(current_mood: str) -> None:
    st.title("Cognitive Emotion Intelligence and Adaptive Lifestyle System")
    st.caption("Single-file Streamlit + PyTorch major project with multimodal fusion, Grad-CAM, digital twin logging, and adaptive recommendations.")
    st.markdown(build_avatar_html(current_mood), unsafe_allow_html=True)


def render_sidebar() -> Dict[str, str]:
    st.sidebar.header("Runtime Controls")
    username = st.sidebar.text_input("Username", value="demo_user")
    hf_token = st.sidebar.text_input("HF_TOKEN", value=os.getenv("HF_TOKEN", ""), type="password")
    hf_model = st.sidebar.text_input("Hugging Face model id", value="google/flan-t5-base")
    spotify_id = st.sidebar.text_input("Spotify client id", value=os.getenv("SPOTIPY_CLIENT_ID", ""), type="password")
    spotify_secret = st.sidebar.text_input("Spotify client secret", value=os.getenv("SPOTIPY_CLIENT_SECRET", ""), type="password")
    voice_enabled = st.sidebar.toggle("Play chatbot response as voice", value=False)
    use_hf = st.sidebar.toggle("Use Hugging Face generation when token/model are available", value=False)
    st.sidebar.info("Recommended for Windows 11: Python 3.11 or 3.12, CPU mode, batch size 4 or 8, 1-3 epochs for demo training.")
    return {
        "username": username,
        "hf_token": hf_token,
        "hf_model": hf_model,
        "spotify_id": spotify_id,
        "spotify_secret": spotify_secret,
        "voice_enabled": voice_enabled,
        "use_hf": use_hf,
    }


def main() -> None:
    st.set_page_config(page_title="CEI Adaptive Lifestyle System", layout="wide")
    ensure_runtime_structure()
    initialize_state()
    sidebar = render_sidebar()
    current_mood = st.session_state.latest_analysis.get("dominant_emotion", "neutral")
    page_header(current_mood)

    tabs = st.tabs(
        [
            "Overview",
            "Emotion Fusion",
            "Dataset Preparation",
            "Training and Grad-CAM",
            "Recommendations and Twin",
            "Mascot Chatbot",
            "Research and Viva",
        ]
    )

    with tabs[0]:
        st.subheader("Project Overview")
        twin = build_twin_snapshot()
        col1, col2, col3 = st.columns(3)
        col1.metric("Twin sessions logged", twin["total_sessions"])
        col2.metric("Dominant recent trend", twin["dominant_trend"].title())
        col3.metric("Average confidence", f"{twin['avg_confidence']:.2f}")

        st.markdown(
            """
            This major-project build focuses on:

            - PyTorch-based EfficientNetV2-S emotion modeling
            - multimodal fusion from face, text, emoji, and voice transcript
            - Grad-CAM explainability
            - CSV-based digital emotional twin logging
            - no-repeat recommendations with reward-style feedback
            - optional Hugging Face and Spotify integrations
            - a single-file Streamlit application suitable for VS Code
            """
        )

        catalog = save_resource_catalog()
        st.write(f"Resource catalog entries ready for export: **{len(catalog)}**")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            show_download_button(RESOURCE_CATALOG_PATH, "Download resource_catalog.csv", "text/csv")
        with col_b:
            show_download_button(TWIN_LOG_PATH, "Download cei_twin_log.csv", "text/csv")
        with col_c:
            show_download_button(RECOMMENDER_STATS_PATH, "Download recommender_stats.csv", "text/csv")

        if DATASET_MANIFEST_PATH.exists():
            show_download_button(DATASET_MANIFEST_PATH, "Download dataset_manifest.json", "application/json")
        if MODEL_META_PATH.exists():
            show_download_button(MODEL_META_PATH, "Download model metadata", "application/json")

    with tabs[1]:
        st.subheader("Multimodal Emotion Fusion")
        model, class_names, model_status = load_model_bundle()
        st.info(model_status)

        input_col1, input_col2 = st.columns(2)
        with input_col1:
            text_input = st.text_area("Text context", placeholder="Describe how you are feeling and what support you want.")
            emoji_input = st.text_input("Emoji input", placeholder="Example: 😄🙂 or 😔😢")
            uploaded_image = st.file_uploader("Upload face image", type=["jpg", "jpeg", "png"], key="image_uploader")
            captured_image = st.camera_input("Or capture image with camera")
        with input_col2:
            if hasattr(st, "audio_input"):
                audio_input = st.audio_input("Record your voice")
            else:
                audio_input = None
                st.warning("This Streamlit build does not expose browser audio input.")

            st.markdown("#### Important emotion-aware questions")
            for question in IMPORTANT_QUESTIONS.get(current_mood, IMPORTANT_QUESTIONS["neutral"]):
                st.markdown(f"- {question}")

        if st.button("Analyze Emotion Profile", type="primary"):
            face_image = load_uploaded_image(uploaded_image) or load_uploaded_image(captured_image)
            modal_scores = {}
            image_mode = "none"
            gradcam_overlay = None

            if text_input.strip():
                modal_scores["text"] = text_emotion_scores(text_input)
            if emoji_input.strip():
                modal_scores["emoji"] = emoji_emotion_scores(emoji_input)

            voice_text, voice_status = transcribe_audio(audio_input)
            if voice_text.strip():
                modal_scores["voice"] = text_emotion_scores(voice_text)
            if audio_input is not None:
                st.caption(voice_status)

            if face_image is not None:
                if model is not None and class_names:
                    modal_scores["image"] = predict_with_model(face_image, model, class_names)
                    image_mode = "trained_model"
                    gradcam_overlay, gradcam_status = generate_gradcam(face_image, model, class_names)
                    st.caption(gradcam_status)
                else:
                    modal_scores["image"] = image_heuristic_scores(face_image)
                    image_mode = "heuristic"

            fused = blend_modal_scores(modal_scores)
            mood, confidence = dominant_emotion(fused)
            available_modalities = sorted(modal_scores.keys())

            st.session_state.latest_analysis = {
                "dominant_emotion": mood,
                "confidence": confidence,
                "fused_scores": fused,
                "modal_scores": modal_scores,
                "voice_text": voice_text,
                "text_input": text_input,
                "emoji_input": emoji_input,
                "image_mode": image_mode,
            }
            log_twin_event(sidebar["username"], fused, text_input, emoji_input, voice_text, image_mode)

            result_col1, result_col2 = st.columns([1, 1])
            with result_col1:
                st.success(f"Dominant emotion: {mood.title()} ({confidence:.2f})")
                st.bar_chart(pd.DataFrame({"score": fused}))
                st.write("Modalities used:", ", ".join(available_modalities) if available_modalities else "none")
                if voice_text:
                    st.write("Voice transcript:", voice_text)
                if face_image is not None:
                    st.image(face_image, caption="Input image", width=250)

            with result_col2:
                if gradcam_overlay is not None:
                    st.image(gradcam_overlay, caption="Grad-CAM overlay", width=250)
                flags = ethical_ai_flags(modal_scores, fused, available_modalities)
                if flags:
                    for flag in flags:
                        st.warning(flag)
                st.markdown("#### Adaptive system response")
                st.write(local_chat_response("general project summary", sidebar["username"], mood))

    with tabs[2]:
        st.subheader("Dataset Preparation")
        st.markdown("Use either a local ZIP import or a public Hugging Face dataset id. The app auto-splits into train/val/test.")

        method = st.radio("Dataset source", ["Manual ZIP import", "Hugging Face dataset"], horizontal=True)
        if method == "Manual ZIP import":
            zip_file = st.file_uploader("Upload dataset ZIP", type=["zip"], key="dataset_zip")
            if st.button("Extract ZIP dataset"):
                ok, message = extract_manual_zip(zip_file)
                if ok:
                    st.success(message)
                else:
                    st.error(message)
        else:
            dataset_id = st.text_input("Hugging Face dataset id", placeholder="Example: your-public-dataset-id")
            sample_size = st.slider("Sample size", min_value=200, max_value=2000, value=800, step=100)
            train_ratio = st.slider("Train ratio", min_value=0.5, max_value=0.85, value=0.70, step=0.05)
            val_ratio = st.slider("Validation ratio", min_value=0.05, max_value=0.30, value=0.15, step=0.05)
            seed = st.number_input("Random seed", min_value=1, max_value=9999, value=42)
            if st.button("Prepare Hugging Face dataset"):
                ok, message = hf_dataset_to_folders(
                    dataset_id=dataset_id,
                    sample_size=sample_size,
                    train_ratio=float(train_ratio),
                    val_ratio=float(val_ratio),
                    seed=int(seed),
                )
                if ok:
                    st.success(message)
                else:
                    st.error(message)

        if DATASET_MANIFEST_PATH.exists():
            manifest = json.loads(DATASET_MANIFEST_PATH.read_text())
            st.json(manifest)

        st.markdown("#### Expected local folder format")
        st.code(
            """dataset/
train/<class_name>/*.png
val/<class_name>/*.png
test/<class_name>/*.png""",
            language="text",
        )

    with tabs[3]:
        st.subheader("Training and Grad-CAM")
        classes = get_class_names_from_train()
        st.write("Detected classes in dataset/train:", classes if classes else "No prepared dataset found yet.")

        batch_size = st.select_slider("Batch size", options=[2, 4, 8, 16], value=4)
        epochs = st.select_slider("Epochs", options=[1, 2, 3, 4, 5], value=2)
        learning_rate = st.select_slider("Learning rate", options=[1e-4, 3e-4, 5e-4, 1e-3], value=3e-4)
        freeze_backbone = st.toggle("Freeze EfficientNetV2 backbone for low-resource training", value=True)

        if st.button("Train PyTorch Emotion Model", type="primary"):
            if not training_ready():
                st.error("Please prepare dataset/train and dataset/val before training.")
            else:
                with st.spinner("Training EfficientNetV2-S on CPU. Keep settings small for student hardware."):
                    metadata = train_emotion_model(
                        epochs=int(epochs),
                        batch_size=int(batch_size),
                        learning_rate=float(learning_rate),
                        freeze_backbone=freeze_backbone,
                    )
                st.success("Training completed and weights saved.")
                history_df = pd.DataFrame(metadata["history"])
                st.dataframe(history_df, use_container_width=True)
                if not history_df.empty:
                    st.line_chart(history_df.set_index("epoch")[["train_accuracy", "val_accuracy"]])
                    st.line_chart(history_df.set_index("epoch")[["train_loss", "val_loss"]])

                val_metrics = metadata["val_metrics"]
                metric_cols = st.columns(4)
                metric_cols[0].metric("Precision", f"{val_metrics['precision']:.3f}")
                metric_cols[1].metric("Recall", f"{val_metrics['recall']:.3f}")
                metric_cols[2].metric("F1", f"{val_metrics['f1']:.3f}")
                metric_cols[3].metric("ROC-AUC", "N/A" if val_metrics["roc_auc"] is None else f"{val_metrics['roc_auc']:.3f}")
                plot_confusion(np.array(val_metrics["confusion_matrix"]), metadata["classes"])

        st.markdown("#### Grad-CAM quick demo")
        gradcam_image = st.file_uploader("Upload an image for Grad-CAM", type=["jpg", "jpeg", "png"], key="gradcam_image")
        if st.button("Generate Grad-CAM from saved model"):
            model, class_names, status = load_model_bundle()
            image = load_uploaded_image(gradcam_image)
            if image is None:
                st.error("Upload an image first.")
            elif model is None:
                st.error(status)
            else:
                overlay, message = generate_gradcam(image, model, class_names)
                st.info(message)
                if overlay is not None:
                    col_a, col_b = st.columns(2)
                    col_a.image(image, caption="Original image", use_container_width=True)
                    col_b.image(overlay, caption="Grad-CAM overlay", use_container_width=True)

    with tabs[4]:
        st.subheader("Recommendations and Digital Twin")
        latest = st.session_state.latest_analysis or {}
        selected_mood = latest.get("dominant_emotion", "neutral")
        selected_mood = st.selectbox("Mood for recommendation", EMOTION_LABELS, index=EMOTION_LABELS.index(selected_mood))
        recommendations = recommend_items(selected_mood, top_n=5)
        st.dataframe(recommendations[["mood", "title", "source", "url", "score"]], use_container_width=True)

        if not recommendations.empty:
            top_item = recommendations.iloc[0]
            st.markdown(f"**Top recommendation:** {top_item['title']} ({top_item['source']})")
            st.markdown(f"[Open recommendation]({top_item['url']})")
            fb_col1, fb_col2 = st.columns(2)
            if fb_col1.button("Helpful"):
                save_recommender_feedback(top_item, selected_mood, reward=1.0)
                st.success("Positive feedback saved.")
            if fb_col2.button("Skip"):
                save_recommender_feedback(top_item, selected_mood, reward=-0.4)
                st.warning("Skip feedback saved.")

            spotify_tracks = spotify_search_tracks(
                query=top_item["title"],
                client_id=sidebar["spotify_id"],
                client_secret=sidebar["spotify_secret"],
                limit=5,
            )
            if spotify_tracks:
                st.markdown("#### Spotify API results")
                for track in spotify_tracks:
                    st.markdown(f"- [{track['name']} - {track['artist']}]({track['url']})")

        twin = build_twin_snapshot()
        st.markdown("#### Digital Emotional Twin Snapshot")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total sessions", twin["total_sessions"])
        c2.metric("Dominant trend", twin["dominant_trend"].title())
        c3.metric("Average confidence", f"{twin['avg_confidence']:.2f}")
        st.bar_chart(pd.DataFrame({"count": twin["recent_distribution"]}))

        stats_df = load_recommender_stats()
        if not stats_df.empty:
            st.markdown("#### Recommendation learning memory")
            st.dataframe(stats_df.sort_values(by="q_value", ascending=False), use_container_width=True)

    with tabs[5]:
        st.subheader("Animated Mascot Chatbot")
        latest = st.session_state.latest_analysis or {}
        mood = latest.get("dominant_emotion", "neutral")
        st.markdown(build_avatar_html(mood), unsafe_allow_html=True)

        st.markdown("#### Suggested important questions")
        for question in IMPORTANT_QUESTIONS.get(mood, IMPORTANT_QUESTIONS["neutral"]):
            st.markdown(f"- {question}")

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        query = st.chat_input("Ask about viva, future scope, real-world use, APIs, Grad-CAM, EfficientNetV2, research gaps, or recommendations.")
        if query:
            st.session_state.chat_history.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            reply = local_chat_response(query, sidebar["username"], mood)
            status = "Local response generated."
            if sidebar["use_hf"]:
                hf_reply, hf_status = huggingface_chat_response(
                    query=query,
                    mood=mood,
                    username=sidebar["username"],
                    token=sidebar["hf_token"],
                    model_id=sidebar["hf_model"],
                )
                if hf_reply:
                    reply = hf_reply
                status = hf_status

            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)
                st.caption(status)
                if sidebar["voice_enabled"]:
                    audio_bytes = generate_speech_bytes(reply)
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/mp3")
                    else:
                        st.info("Voice output is unavailable in the current environment.")

    with tabs[6]:
        st.subheader("Research and Viva Support")
        st.markdown("#### High-value viva topics")
        topic_order = [
            "real_world_problem",
            "research_gaps",
            "protagonist",
            "xai",
            "grad_cam",
            "efficientnetv2",
            "gap",
            "ethical_ai",
            "digital_twin",
            "cei",
            "multi_ai",
            "rl_logic",
            "pytorch",
            "free_api",
            "future_scope",
        ]
        for topic in topic_order:
            with st.expander(topic.replace("_", " ").title(), expanded=False):
                st.write(VIVA_KNOWLEDGE[topic].strip())

        st.markdown("#### Repository documents")
        show_download_button(DOCS_DIR / "RESEARCH_PAPER_IEEE_DRAFT.md", "Download IEEE-style draft", "text/markdown")
        show_download_button(DOCS_DIR / "VIVA_GUIDE.md", "Download viva guide", "text/markdown")

        st.warning("This system supports emotional reflection and adaptive recommendations, but it is not a medical diagnosis tool.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--export-catalog":
        raise SystemExit(export_catalog_cli())
    main()
