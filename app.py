"""
Single-file Streamlit major project:
Cognitive Emotion Intelligence and Adaptive Lifestyle System

Key features
- Text emotion detection with Hugging Face transformers + NLTK enrichment
- Voice input using Streamlit audio capture + SpeechRecognition transcript
- Facial emotion analysis using OpenCV + PyTorch EfficientNetV2-S
- Grad-CAM explainability for the facial model
- Cognitive Emotional Intelligence fusion engine
- Adaptive therapeutic chatbot with optional free Hugging Face Inference API
- Animated mascot, digital emotional twin logging, ethical AI notes
- Lightweight recommendation engine with reinforcement-style feedback
- Training, evaluation, and export utilities inside the same file

Run:
    streamlit run app.py

Optional CLI:
    python app.py --export-catalog
"""

from __future__ import annotations

import csv
import io
import json
import os
import random
import re
import sys
import tempfile
import textwrap
import urllib.error
import urllib.parse
import urllib.request
import wave
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import nltk
import speech_recognition as sr
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.corpus import wordnet
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import EfficientNet_V2_S_Weights
from transformers import pipeline


APP_TITLE = "Cognitive Emotion Intelligence and Adaptive Lifestyle System"
APP_SUBTITLE = (
    "Single-file PyTorch + Streamlit project for multimodal emotion fusion, "
    "therapeutic dialogue, explainable AI, and adaptive lifestyle guidance."
)

ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"
DATASET_DIR = ROOT_DIR / "dataset"
MODEL_PATH = MODELS_DIR / "efficientnet_emotion.pth"
MODEL_META_PATH = MODELS_DIR / "efficientnet_emotion_metadata.json"
DATASET_MANIFEST_PATH = ROOT_DIR / "dataset_manifest.json"
TWIN_LOG_PATH = ROOT_DIR / "cei_twin_log.csv"
RECOMMENDER_LOG_PATH = ROOT_DIR / "recommender_stats.csv"
RESOURCE_CATALOG_PATH = ROOT_DIR / "resource_catalog.csv"

DEFAULT_FACE_CLASSES = ["happy", "sad", "angry", "neutral"]
FUSION_EMOTIONS = ["happy", "sad", "angry", "anxious", "neutral"]

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

INTENT_KEYWORDS = {
    "study_support": {
        "study",
        "exam",
        "focus",
        "revision",
        "assignment",
        "college",
        "semester",
        "defence",
        "project",
    },
    "stress_relief": {
        "stress",
        "pressure",
        "panic",
        "tense",
        "overwhelmed",
        "anxiety",
        "worry",
        "fear",
    },
    "motivation": {
        "motivation",
        "energy",
        "discipline",
        "confidence",
        "goal",
        "achieve",
        "future",
        "career",
    },
    "wellness": {
        "sleep",
        "diet",
        "health",
        "exercise",
        "routine",
        "lifestyle",
        "walk",
        "rest",
    },
    "social_support": {
        "friend",
        "family",
        "relationship",
        "lonely",
        "talk",
        "support",
        "together",
        "trust",
    },
    "general_support": set(),
}

QUESTION_BANK = {
    "study_support": {
        "anxious": [
            "Which subject feels heaviest right now, and what is the smallest task you can finish in the next 20 minutes?",
            "What part of the project is unclear: coding, explanation, documentation, or confidence before viva?",
            "If you had to explain your project to a classmate in 3 lines, what would you say first?",
        ],
        "sad": [
            "What result or setback affected your confidence the most today?",
            "Would it help to split the work into coding, documentation, and presentation practice?",
            "What is one completed task from this week that proves you are still making progress?",
        ],
        "angry": [
            "What is frustrating you most: bugs, deadlines, or repeated rework?",
            "Which part can be simplified so the project feels under control again?",
            "Would a short reset help before you return to the hardest task?",
        ],
        "happy": [
            "What part of the project are you most proud of right now?",
            "Which demo feature should you highlight first in front of the examiners?",
            "How can you turn this momentum into a clean final presentation?",
        ],
        "neutral": [
            "Which module do you want to improve next: emotion detection, chatbot logic, or explainability?",
            "What single milestone would make today feel productive?",
            "Do you want concise guidance, technical detail, or interview-style preparation?",
        ],
    },
    "stress_relief": {
        "anxious": [
            "What is your biggest source of pressure at this moment?",
            "Would a calming routine, focused plan, or supportive conversation help you most right now?",
            "What would make the next one hour feel manageable instead of overwhelming?",
        ],
        "sad": [
            "Is your low mood linked more to exhaustion, disappointment, or feeling isolated?",
            "What kind of support usually helps you recover gently?",
            "Would you like a softer plan with less pressure and more recovery time?",
        ],
        "angry": [
            "Did something specific trigger your irritation, or has stress been building gradually?",
            "Would channeling energy into movement, music, or problem-solving help first?",
            "What outcome are you trying to protect by reacting this strongly?",
        ],
        "happy": [
            "What activity is helping you stay light and balanced today?",
            "Would you like upbeat music, a productivity boost, or a reflective check-in?",
            "How can you preserve this positive mood through the rest of the day?",
        ],
        "neutral": [
            "Do you want a short emotional check-in or a practical action plan?",
            "How has your stress level changed compared with earlier today?",
            "Would you prefer support for focus, calm, sleep, or confidence?",
        ],
    },
    "motivation": {
        "anxious": [
            "What goal matters most right now, and what is the smallest reliable step toward it?",
            "What fear is reducing your momentum: failure, judgment, or uncertainty?",
            "Would a progress tracker help you feel more in control?",
        ],
        "sad": [
            "What used to motivate you that feels distant today?",
            "Which achievement would give you a quick sense of regained control?",
            "Would positive reinforcement or a simpler target help more?",
        ],
        "angry": [
            "Can this frustration be redirected into a concrete achievement today?",
            "Which obstacle are you ready to remove immediately?",
            "What standard are you holding yourself to that may be too harsh?",
        ],
        "happy": [
            "What ambitious target feels realistic while your energy is high?",
            "Would you like a bold action list or a polished execution plan?",
            "Which strength should you use most today: creativity, persistence, or clarity?",
        ],
        "neutral": [
            "What kind of motivation do you need most: emotional encouragement or a task structure?",
            "Which action would create visible progress quickly?",
            "Do you want an accountability question or a confidence boost first?",
        ],
    },
    "wellness": {
        "anxious": [
            "How are sleep, food, and rest affecting your emotional balance today?",
            "Would a short breathing exercise or a light activity help your body settle first?",
            "What habit most urgently needs consistency right now?",
        ],
        "sad": [
            "Have you eaten, rested, and moved enough today to support your mood?",
            "What low-effort wellness action feels possible right now?",
            "Would a kinder daily routine help more than a strict one?",
        ],
        "angry": [
            "Is physical tiredness making your reactions sharper today?",
            "Would movement or quiet recovery help release some intensity?",
            "What body signal are you ignoring because of workload pressure?",
        ],
        "happy": [
            "Which healthy habit is helping you feel steady right now?",
            "How can you keep this balance through study or work demands?",
            "Would you like suggestions for maintaining mood with simple routines?",
        ],
        "neutral": [
            "Would you like help with sleep, daily routine, movement, or nutrition habits?",
            "Which wellness habit is easiest for you to maintain consistently?",
            "Do you want a minimal routine or a more structured plan?",
        ],
    },
    "social_support": {
        "anxious": [
            "Is there someone you trust enough to talk to today, even briefly?",
            "What makes asking for support difficult right now?",
            "Would preparing what to say reduce some of the emotional load?",
        ],
        "sad": [
            "Do you feel unheard, disconnected, or emotionally drained by recent interactions?",
            "Who usually helps you feel safe and understood?",
            "Would a message draft help you reconnect with someone supportive?",
        ],
        "angry": [
            "Is this conflict about being misunderstood, disrespected, or overloaded?",
            "Would you prefer to calm down first or plan a clear response?",
            "What boundary needs to be communicated more clearly?",
        ],
        "happy": [
            "Who would you like to share your good energy or recent progress with?",
            "How can you strengthen the relationships that support your growth?",
            "Would you like ideas for meaningful conversation rather than small talk?",
        ],
        "neutral": [
            "Would talking to someone help, or do you need private reflection first?",
            "What kind of support feels most useful: emotional, practical, or motivational?",
            "Who in your circle is most likely to respond calmly and helpfully?",
        ],
    },
    "general_support": {
        "anxious": [
            "What is the main thought repeating in your mind right now?",
            "Would you like help understanding the emotion, making a plan, or both?",
            "What would feeling 10 percent better look like in the next hour?",
        ],
        "sad": [
            "What feels emotionally heavy for you today?",
            "Do you want space to reflect, or would practical guidance help more?",
            "What small act of self-support is realistic right now?",
        ],
        "angry": [
            "What are you trying to protect or fix underneath this frustration?",
            "Would you like a calm response strategy or a way to release tension first?",
            "What outcome matters more: being heard, being right, or finding relief?",
        ],
        "happy": [
            "What would you like to build on while your mood is positive?",
            "Should I help you channel this energy into work, wellness, or creativity?",
            "What success would make today memorable for you?",
        ],
        "neutral": [
            "What kind of help feels most useful right now?",
            "Would you like an emotional check-in, a practical plan, or a creative idea?",
            "What topic is most important for you at this moment?",
        ],
    },
}

EMOTION_STYLE = {
    "happy": {"tone": "warm, encouraging, and upbeat", "mode": "friendly"},
    "sad": {"tone": "gentle, validating, and slow-paced", "mode": "therapist"},
    "angry": {"tone": "calm, grounded, and de-escalating", "mode": "motivational"},
    "anxious": {"tone": "steady, reassuring, and structured", "mode": "therapist"},
    "neutral": {"tone": "balanced, clear, and supportive", "mode": "friendly"},
}

VALENCE_MAP = {
    "happy": 0.88,
    "sad": 0.18,
    "angry": 0.15,
    "anxious": 0.28,
    "neutral": 0.56,
}

AROUSAL_MAP = {
    "happy": 0.72,
    "sad": 0.24,
    "angry": 0.86,
    "anxious": 0.78,
    "neutral": 0.42,
}


@dataclass
class ModalityEmotion:
    source: str
    label: str
    confidence: float
    scores: Dict[str, float]
    details: Dict[str, str | float]


@dataclass
class FusionState:
    label: str
    confidence: float
    valence: float
    arousal: float
    wellbeing_score: int
    mode: str
    notes: List[str]


def ensure_directories() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    DATASET_DIR.mkdir(exist_ok=True)


def append_csv_row(path: Path, headers: Sequence[str], row: Dict[str, object]) -> None:
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(headers))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def save_json(path: Path, payload: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_json(path: Path, default: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    if not path.exists():
        return default or {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_resource(show_spinner=False)
def ensure_nltk_resources() -> bool:
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for locator, package in resources:
        try:
            nltk.data.find(locator)
        except LookupError:
            nltk.download(package, quiet=True)
    return True


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z']+", (text or "").lower())


def expand_synonyms(tokens: Sequence[str], max_tokens: int = 8) -> set[str]:
    ensure_nltk_resources()
    expanded = set(token.lower() for token in tokens)
    for token in list(expanded)[:max_tokens]:
        for synset in wordnet.synsets(token)[:3]:
            for lemma in synset.lemma_names()[:3]:
                expanded.add(lemma.lower().replace("_", " "))
    return expanded


def infer_intent(text: str) -> Tuple[str, Dict[str, int], List[str]]:
    tokens = tokenize(text)
    expanded = expand_synonyms(tokens)
    scores = {
        intent: len(expanded.intersection(keywords))
        for intent, keywords in INTENT_KEYWORDS.items()
    }
    best_intent = max(scores, key=scores.get)
    if scores.get(best_intent, 0) == 0:
        best_intent = "general_support"
    context_terms = sorted(expanded.intersection(set().union(*INTENT_KEYWORDS.values())))
    return best_intent, scores, context_terms[:12]


def normalize_emotion_label(label: str) -> str:
    raw = (label or "").lower().strip()
    mapping = {
        "joy": "happy",
        "happiness": "happy",
        "positive": "happy",
        "love": "happy",
        "sadness": "sad",
        "negative": "sad",
        "anger": "angry",
        "disgust": "angry",
        "fear": "anxious",
        "surprise": "happy",
        "neutral": "neutral",
        "calm": "neutral",
    }
    return mapping.get(raw, raw if raw in FUSION_EMOTIONS else "neutral")


@st.cache_resource(show_spinner=False)
def load_text_emotion_classifier(hf_token: str = ""):
    try:
        return pipeline(
            task="text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,
            token=hf_token or None,
        )
    except Exception:
        return None


def lexicon_emotion_scores(text: str) -> Dict[str, float]:
    words = set(tokenize(text))
    lexicon = {
        "happy": {
            "happy",
            "excited",
            "grateful",
            "love",
            "great",
            "good",
            "proud",
            "relaxed",
            "calm",
        },
        "sad": {
            "sad",
            "down",
            "tired",
            "hopeless",
            "lonely",
            "upset",
            "hurt",
            "cry",
            "empty",
        },
        "angry": {
            "angry",
            "irritated",
            "frustrated",
            "annoyed",
            "rage",
            "mad",
            "hate",
            "fed",
        },
        "anxious": {
            "anxious",
            "stress",
            "panic",
            "worried",
            "overwhelmed",
            "nervous",
            "fear",
            "tense",
        },
        "neutral": {"okay", "fine", "normal", "average", "stable"},
    }
    scores = {}
    total = 0.0
    for emotion, keywords in lexicon.items():
        score = float(len(words.intersection(keywords))) + 0.1
        scores[emotion] = score
        total += score
    return {key: value / total for key, value in scores.items()}


def detect_text_emotion(text: str, hf_token: str = "") -> ModalityEmotion:
    content = clean_text(text)
    if not content:
        return ModalityEmotion(
            source="text",
            label="neutral",
            confidence=0.0,
            scores={"neutral": 1.0},
            details={"note": "No text supplied."},
        )

    classifier = load_text_emotion_classifier(hf_token)
    if classifier is not None:
        try:
            raw_scores = classifier(content)
            if raw_scores and isinstance(raw_scores[0], list):
                raw_scores = raw_scores[0]
            normalized: Dict[str, float] = {}
            for item in raw_scores:
                label = normalize_emotion_label(str(item["label"]))
                normalized[label] = normalized.get(label, 0.0) + float(item["score"])
            for emotion in FUSION_EMOTIONS:
                normalized.setdefault(emotion, 0.0)
            label = max(normalized, key=normalized.get)
            return ModalityEmotion(
                source="text",
                label=label,
                confidence=float(normalized[label]),
                scores=normalized,
                details={"text_length": float(len(content))},
            )
        except Exception:
            pass

    fallback_scores = lexicon_emotion_scores(content)
    label = max(fallback_scores, key=fallback_scores.get)
    return ModalityEmotion(
        source="text",
        label=label,
        confidence=float(fallback_scores[label]),
        scores=fallback_scores,
        details={"note": "Lexicon fallback used."},
    )


def parse_audio_bytes(audio_bytes: bytes) -> Dict[str, float]:
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as audio_file:
            channels = audio_file.getnchannels()
            frame_rate = audio_file.getframerate()
            sample_width = audio_file.getsampwidth()
            frames = audio_file.readframes(audio_file.getnframes())
    except wave.Error:
        return {"duration": 0.0, "rms": 0.0, "zcr": 0.0}

    dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
    dtype = dtype_map.get(sample_width)
    if dtype is None or not frames:
        return {"duration": 0.0, "rms": 0.0, "zcr": 0.0}

    signal = np.frombuffer(frames, dtype=dtype).astype(np.float32)
    if channels > 1:
        signal = signal.reshape(-1, channels).mean(axis=1)
    if signal.size == 0:
        return {"duration": 0.0, "rms": 0.0, "zcr": 0.0}

    peak = max(1.0, float(np.max(np.abs(signal))))
    signal = signal / peak
    duration = float(signal.size / max(frame_rate, 1))
    rms = float(np.sqrt(np.mean(signal**2)))
    zcr = float(np.mean(np.abs(np.diff(np.sign(signal)))) / 2.0) if signal.size > 1 else 0.0
    return {"duration": duration, "rms": rms, "zcr": zcr}


def transcribe_audio_to_text(audio_bytes: bytes) -> str:
    recognizer = sr.Recognizer()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_handle:
        temp_handle.write(audio_bytes)
        temp_path = temp_handle.name

    try:
        with sr.AudioFile(temp_path) as source:
            audio = recognizer.record(source)
        try:
            return clean_text(recognizer.recognize_google(audio))
        except sr.UnknownValueError:
            return ""
        except sr.RequestError:
            return ""
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass


def detect_voice_emotion(audio_bytes: bytes, hf_token: str = "") -> ModalityEmotion:
    signal_features = parse_audio_bytes(audio_bytes)
    transcript = transcribe_audio_to_text(audio_bytes)
    transcript_result = detect_text_emotion(transcript, hf_token) if transcript else None

    rms = signal_features["rms"]
    zcr = signal_features["zcr"]
    duration = signal_features["duration"]

    signal_scores = {
        "happy": 0.1,
        "sad": 0.1,
        "angry": 0.1,
        "anxious": 0.1,
        "neutral": 0.2,
    }

    if rms > 0.28 and zcr > 0.18:
        signal_scores["angry"] += 0.45
        signal_scores["happy"] += 0.15
    elif rms > 0.18 and zcr > 0.10:
        signal_scores["happy"] += 0.35
        signal_scores["anxious"] += 0.15
    elif rms < 0.08 and duration > 2.5:
        signal_scores["sad"] += 0.40
        signal_scores["neutral"] += 0.10
    elif rms < 0.10:
        signal_scores["neutral"] += 0.30
    else:
        signal_scores["neutral"] += 0.20

    if transcript_result is not None:
        for emotion in FUSION_EMOTIONS:
            signal_scores[emotion] = signal_scores.get(emotion, 0.0) + 0.55 * transcript_result.scores.get(emotion, 0.0)

    total = sum(signal_scores.values()) or 1.0
    normalized = {key: value / total for key, value in signal_scores.items()}
    label = max(normalized, key=normalized.get)

    return ModalityEmotion(
        source="voice",
        label=label,
        confidence=float(normalized[label]),
        scores=normalized,
        details={
            "transcript": transcript or "Speech could not be transcribed.",
            "duration": duration,
            "rms": rms,
            "zcr": zcr,
        },
    )


class EmotionEfficientNet(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        try:
            base_model = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
        except Exception:
            # Fall back to randomly initialized weights if pretrained weights
            # are unavailable in an offline or restricted environment.
            base_model = models.efficientnet_v2_s(weights=None)
        in_features = base_model.classifier[1].in_features
        base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(in_features, num_classes),
        )
        self.backbone = base_model

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.backbone(inputs)


def available_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_face_bgr(face_bgr: np.ndarray) -> torch.Tensor:
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(face_rgb, (224, 224))
    tensor = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return tensor.unsqueeze(0)


def heuristic_face_emotion(face_bgr: np.ndarray) -> Tuple[str, float]:
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    mouth_region = gray[int(height * 0.58) : int(height * 0.90), int(width * 0.20) : int(width * 0.80)]
    eye_region = gray[int(height * 0.18) : int(height * 0.45), int(width * 0.18) : int(width * 0.82)]
    mouth_edges = cv2.Canny(mouth_region, 50, 150).mean() / 255.0 if mouth_region.size else 0.0
    eye_variance = float(np.var(eye_region)) / (255.0**2) if eye_region.size else 0.0
    brightness = float(np.mean(gray)) / 255.0

    if mouth_edges > 0.15 and brightness > 0.42:
        return "happy", 0.56
    if brightness < 0.32:
        return "sad", 0.54
    if eye_variance > 0.09 and mouth_edges < 0.10:
        return "angry", 0.53
    return "neutral", 0.50


@st.cache_resource(show_spinner=False)
def load_face_model():
    metadata = load_json(MODEL_META_PATH, default={})
    class_names = metadata.get("class_names", DEFAULT_FACE_CLASSES)
    num_classes = len(class_names)
    model = EmotionEfficientNet(num_classes=num_classes)
    trained = False

    if MODEL_PATH.exists():
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
            if "class_names" in checkpoint:
                class_names = checkpoint["class_names"]
        else:
            model.load_state_dict(checkpoint)
        trained = True

    model.eval()
    return model, class_names, trained


class GradCAM:
    def __init__(self, model: nn.Module, target_module: nn.Module) -> None:
        self.model = model
        self.target_module = target_module
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.forward_handle = target_module.register_forward_hook(self._forward_hook)
        self.backward_handle = target_module.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module: nn.Module, inputs: Tuple[torch.Tensor], output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _backward_hook(
        self,
        module: nn.Module,
        grad_inputs: Tuple[torch.Tensor],
        grad_output: Tuple[torch.Tensor],
    ) -> None:
        self.gradients = grad_output[0].detach()

    def generate(self, inputs: torch.Tensor, target_index: Optional[int] = None) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        outputs = self.model(inputs)
        if target_index is None:
            target_index = int(outputs.argmax(dim=1).item())
        score = outputs[:, target_index].sum()
        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            return np.zeros((224, 224), dtype=np.float32)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        return cam.astype(np.float32)

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()


def overlay_gradcam(image_bgr: np.ndarray, cam: np.ndarray) -> np.ndarray:
    resized_cam = cv2.resize(cam, (image_bgr.shape[1], image_bgr.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(np.clip(resized_cam, 0.0, 1.0) * 255), cv2.COLORMAP_JET)
    blended = cv2.addWeighted(image_bgr, 0.55, heatmap, 0.45, 0)
    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)


def detect_faces(frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))
    return list(faces)


def detect_face_emotion(frame_bgr: np.ndarray) -> Tuple[ModalityEmotion, np.ndarray, Optional[np.ndarray], bool]:
    faces = detect_faces(frame_bgr)
    display_frame = frame_bgr.copy()
    if not faces:
        result = ModalityEmotion(
            source="face",
            label="neutral",
            confidence=0.0,
            scores={"neutral": 1.0},
            details={"note": "No face detected."},
        )
        return result, cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), None, False

    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (33, 150, 243), 2)
    face_crop = frame_bgr[y : y + h, x : x + w]

    model, class_names, trained = load_face_model()
    input_tensor = preprocess_face_bgr(face_crop)

    if trained:
        with torch.no_grad():
            logits = model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
        label_index = int(np.argmax(probabilities))
        raw_label = class_names[label_index]
        label = normalize_emotion_label(str(raw_label))
        scores = {
            normalize_emotion_label(str(class_names[idx])): float(probabilities[idx])
            for idx in range(len(class_names))
        }
        for emotion in FUSION_EMOTIONS:
            scores.setdefault(emotion, 0.0)
        confidence = float(max(probabilities))

        gradcam = GradCAM(model, model.backbone.features[-1])
        cam = gradcam.generate(input_tensor, target_index=label_index)
        gradcam.close()
        overlay = overlay_gradcam(face_crop, cam)
    else:
        label, confidence = heuristic_face_emotion(face_crop)
        scores = {emotion: 0.0 for emotion in FUSION_EMOTIONS}
        scores[label] = confidence
        overlay = None

    result = ModalityEmotion(
        source="face",
        label=label,
        confidence=confidence,
        scores=scores,
        details={"bbox": f"{x},{y},{w},{h}", "trained_model": float(trained)},
    )
    display_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
    return result, display_rgb, overlay, trained


def fuse_emotions(modalities: Sequence[ModalityEmotion]) -> FusionState:
    weights = {"text": 0.40, "voice": 0.35, "face": 0.25}
    weighted_scores = {emotion: 0.0 for emotion in FUSION_EMOTIONS}
    total_weight = 0.0
    notes: List[str] = []

    for modality in modalities:
        if modality.confidence <= 0.0:
            notes.append(f"{modality.source} input unavailable or low confidence.")
            continue
        modality_weight = weights.get(modality.source, 0.20) * max(0.10, modality.confidence)
        total_weight += modality_weight
        for emotion in FUSION_EMOTIONS:
            weighted_scores[emotion] += modality_weight * modality.scores.get(emotion, 0.0)

    if total_weight == 0.0:
        return FusionState(
            label="neutral",
            confidence=0.0,
            valence=VALENCE_MAP["neutral"],
            arousal=AROUSAL_MAP["neutral"],
            wellbeing_score=56,
            mode="friendly",
            notes=["No reliable modality was available."],
        )

    fused_scores = {key: value / total_weight for key, value in weighted_scores.items()}
    label = max(fused_scores, key=fused_scores.get)
    confidence = float(fused_scores[label])
    valence = sum(VALENCE_MAP[emotion] * score for emotion, score in fused_scores.items())
    arousal = sum(AROUSAL_MAP[emotion] * score for emotion, score in fused_scores.items())
    wellbeing_score = int(round(np.clip(valence, 0.0, 1.0) * 100))

    if label in {"sad", "anxious"} or wellbeing_score < 35:
        mode = "therapist"
    elif label == "angry" or arousal > 0.70:
        mode = "motivational"
    else:
        mode = "friendly"

    if confidence < 0.55:
        notes.append("Fusion confidence is moderate; treat guidance as supportive, not diagnostic.")

    return FusionState(
        label=label,
        confidence=confidence,
        valence=valence,
        arousal=arousal,
        wellbeing_score=wellbeing_score,
        mode=mode,
        notes=notes,
    )


def ethical_ai_monitoring(modalities: Sequence[ModalityEmotion], fusion: FusionState, face_trained: bool) -> List[str]:
    notes = [
        "This system is for wellbeing support, education, and demo use only. It is not a medical diagnosis tool.",
        "Only the current interaction is used for response generation unless the user keeps the browser session open.",
    ]
    if not face_trained:
        notes.append("Facial analysis is running in heuristic fallback mode until a trained .pth checkpoint is available.")
    if any(modality.confidence == 0.0 for modality in modalities):
        notes.append("Multimodal fusion used partial inputs because one or more modalities were missing.")
    if fusion.confidence < 0.55:
        notes.append("Low-confidence emotional inference detected. Human review and user self-report should be preferred.")
    return notes


def build_resource_catalog() -> List[Dict[str, str]]:
    moods = {
        "happy": [
            "upbeat focus",
            "celebration playlist",
            "feel good acoustic",
            "creative flow mix",
            "sunrise motivation",
            "positive study beats",
        ],
        "sad": [
            "gentle healing",
            "soft comfort music",
            "mindful recovery sounds",
            "self compassion ambient",
            "quiet journaling mix",
            "slow emotional reset",
        ],
        "angry": [
            "cool down music",
            "release tension instrumental",
            "reset focus beats",
            "calm after conflict",
            "grounding piano focus",
            "breathing reset soundtrack",
        ],
        "anxious": [
            "calming breathing music",
            "deep focus ambient",
            "anxiety relief instrumental",
            "steady heartbeat calm",
            "exam stress relief mix",
            "mind clearing nature audio",
        ],
        "neutral": [
            "balanced productivity",
            "study lounge",
            "light reflective mix",
            "gentle coding focus",
            "weekend reset session",
            "stable routine soundtrack",
        ],
    }
    sources = {
        "Spotify Search": "https://open.spotify.com/search/{query}",
        "YouTube Search": "https://www.youtube.com/results?search_query={query}",
        "YouTube Music Search": "https://music.youtube.com/search?q={query}",
        "Wellness Search": "https://www.youtube.com/results?search_query={query}",
    }
    catalog: List[Dict[str, str]] = []
    for mood, queries in moods.items():
        for query in queries:
            for source_name, source_url in sources.items():
                if source_name == "Wellness Search":
                    type_name = "breathing"
                    full_query = f"{mood} wellbeing exercise {query}"
                else:
                    type_name = "music"
                    full_query = f"{mood} {query}"
                catalog.append(
                    {
                        "mood": mood,
                        "title": f"{mood.title()} support - {query.title()}",
                        "url": source_url.format(query=urllib.parse.quote(full_query)),
                        "source": source_name,
                        "type": type_name,
                        "offline_fallback": "Use locally saved calming music, journaling, breathing, or a short walk.",
                    }
                )
    return catalog


def export_resource_catalog(path: Path = RESOURCE_CATALOG_PATH) -> int:
    rows = build_resource_catalog()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["mood", "title", "url", "source", "type", "offline_fallback"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def load_feedback_memory() -> Dict[str, float]:
    memory: Dict[str, float] = {}
    if not RECOMMENDER_LOG_PATH.exists():
        return memory
    with RECOMMENDER_LOG_PATH.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = f"{row.get('mood','neutral')}|{row.get('resource_title','')}"
            feedback = row.get("feedback", "")
            if feedback == "liked":
                memory[key] = memory.get(key, 0.0) + 1.0
            elif feedback == "skipped":
                memory[key] = memory.get(key, 0.0) - 0.4
    return memory


def recommend_resources(
    mood: str,
    history: set[str],
    top_n: int = 3,
) -> List[Dict[str, str]]:
    catalog = [row for row in build_resource_catalog() if row["mood"] == mood]
    if not catalog:
        catalog = build_resource_catalog()

    feedback_memory = load_feedback_memory()
    scored: List[Tuple[float, Dict[str, str]]] = []
    for item in catalog:
        key = f"{mood}|{item['title']}"
        score = random.random() * 0.2 + feedback_memory.get(key, 0.0)
        if item["url"] in history:
            score -= 1.5
        scored.append((score, item))

    ranked = [item for _, item in sorted(scored, key=lambda pair: pair[0], reverse=True)]
    unseen = [item for item in ranked if item["url"] not in history]
    if len(unseen) < top_n:
        history.clear()
        unseen = ranked
    chosen = unseen[:top_n]
    for item in chosen:
        history.add(item["url"])
    return chosen


def log_recommendation_feedback(username: str, mood: str, item: Dict[str, str], feedback: str) -> None:
    append_csv_row(
        RECOMMENDER_LOG_PATH,
        [
            "timestamp",
            "username",
            "mood",
            "resource_title",
            "resource_url",
            "source",
            "feedback",
        ],
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "username": username,
            "mood": mood,
            "resource_title": item["title"],
            "resource_url": item["url"],
            "source": item["source"],
            "feedback": feedback,
        },
    )


def build_supportive_template(
    username: str,
    user_text: str,
    intent: str,
    fusion: FusionState,
    questions: List[str],
    resources: List[Dict[str, str]],
) -> str:
    first_name = username.strip() or "there"
    style = EMOTION_STYLE.get(fusion.label, EMOTION_STYLE["neutral"])
    action_map = {
        "study_support": "Break the work into one tiny coded task, one written explanation, and one confidence-building rehearsal.",
        "stress_relief": "Slow the pace, exhale fully, and choose one action that reduces pressure instead of chasing everything at once.",
        "motivation": "Anchor your energy to one visible win so momentum becomes measurable, not just emotional.",
        "wellness": "Support the mind through the body: hydration, a short walk, posture reset, and a realistic rest plan.",
        "social_support": "Reach out to one safe person with one honest sentence instead of waiting until everything becomes heavier.",
        "general_support": "Name the feeling, reduce the next task, and focus only on the next controllable step.",
    }
    recommendation_text = ""
    if resources:
        recommendation_text = (
            f" A suitable lifestyle cue is '{resources[0]['title']}' via {resources[0]['source']}."
        )
    follow_up = questions[0] if questions else "What feels most important to you right now?"

    return (
        f"{first_name}, I am reading your state as mostly {fusion.label} with a wellbeing score around "
        f"{fusion.wellbeing_score}/100, so I will respond in a {style['tone']} way. "
        f"From what you shared, the key theme is {intent.replace('_', ' ')}. "
        f"{action_map.get(intent, action_map['general_support'])}{recommendation_text} "
        f"My next question is: {follow_up}"
    )


def call_hugging_face_inference_api(prompt: str, hf_token: str) -> str:
    if not hf_token:
        return ""

    endpoint = "https://api-inference.huggingface.co/models/google/flan-t5-small"
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 160,
            "temperature": 0.6,
            "return_full_text": False,
        },
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {hf_token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
        return ""

    if isinstance(data, list) and data and "generated_text" in data[0]:
        return clean_text(data[0]["generated_text"])
    if isinstance(data, dict) and "generated_text" in data:
        return clean_text(str(data["generated_text"]))
    return ""


def select_questions(intent: str, emotion: str, asked: set[str], count: int = 3) -> List[str]:
    bank = QUESTION_BANK.get(intent, QUESTION_BANK["general_support"])
    options = bank.get(emotion, bank.get("neutral", []))
    unseen = [question for question in options if question not in asked]
    if len(unseen) < count:
        asked.clear()
        unseen = options
    selected = unseen[:count]
    for question in selected:
        asked.add(question)
    return selected


def generate_chatbot_reply(
    username: str,
    user_text: str,
    fusion: FusionState,
    intent: str,
    context_terms: List[str],
    questions: List[str],
    resources: List[Dict[str, str]],
    history: List[Dict[str, str]],
    hf_token: str,
) -> str:
    template_reply = build_supportive_template(username, user_text, intent, fusion, questions, resources)
    history_lines = []
    for item in history[-4:]:
        history_lines.append(f"{item['role']}: {item['content']}")
    history_block = "\n".join(history_lines) if history_lines else "No earlier history."
    resource_hint = resources[0]["title"] if resources else "balanced music or breathing support"
    question_hint = questions[0] if questions else "What matters most to you right now?"
    prompt = textwrap.dedent(
        f"""
        You are an emotionally intelligent therapeutic mascot named AARA.
        Reply in less than 150 words.
        User name: {username or 'User'}
        Detected emotion: {fusion.label}
        Response mode: {fusion.mode}
        Wellbeing score: {fusion.wellbeing_score}/100
        Intent: {intent}
        Context terms: {", ".join(context_terms) if context_terms else "none"}
        Recent history:
        {history_block}

        User message:
        {user_text}

        Requirements:
        - Be empathetic, specific, and non-clinical.
        - Give one practical next step.
        - Ask one personalized follow-up question.
        - Avoid generic FAQ language and avoid repetition.
        - If suitable, mention this lifestyle cue: {resource_hint}.
        - Suggested follow-up question theme: {question_hint}
        """
    ).strip()

    api_reply = call_hugging_face_inference_api(prompt, hf_token)
    if api_reply:
        return api_reply
    return template_reply


def maybe_speak_text(text: str) -> Tuple[bool, str]:
    try:
        import pyttsx3
    except Exception:
        return False, "pyttsx3 is not available in this environment."

    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 175)
        engine.say(text)
        engine.runAndWait()
        return True, "Response spoken on the local machine."
    except Exception as exc:
        return False, f"Text-to-speech could not start: {exc}"


def log_digital_twin(
    username: str,
    user_text: str,
    text_result: ModalityEmotion,
    voice_result: ModalityEmotion,
    face_result: ModalityEmotion,
    fusion: FusionState,
    reply: str,
) -> None:
    append_csv_row(
        TWIN_LOG_PATH,
        [
            "timestamp",
            "username",
            "user_text",
            "text_emotion",
            "voice_emotion",
            "voice_transcript",
            "face_emotion",
            "fusion_emotion",
            "fusion_confidence",
            "wellbeing_score",
            "mode",
            "reply_excerpt",
        ],
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "username": username,
            "user_text": user_text,
            "text_emotion": text_result.label,
            "voice_emotion": voice_result.label,
            "voice_transcript": voice_result.details.get("transcript", ""),
            "face_emotion": face_result.label,
            "fusion_emotion": fusion.label,
            "fusion_confidence": round(fusion.confidence, 4),
            "wellbeing_score": fusion.wellbeing_score,
            "mode": fusion.mode,
            "reply_excerpt": clean_text(reply)[:240],
        },
    )


def summarize_dataset(root: Path = DATASET_DIR) -> Dict[str, object]:
    manifest = {"root": str(root), "splits": {}, "generated_at": datetime.now().isoformat(timespec="seconds")}
    for split in ["train", "val", "test"]:
        split_dir = root / split
        classes = {}
        if split_dir.exists():
            for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
                count = len(
                    [
                        file
                        for file in class_dir.iterdir()
                        if file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
                    ]
                )
                classes[class_dir.name] = count
        manifest["splits"][split] = classes
    save_json(DATASET_MANIFEST_PATH, manifest)
    return manifest


def build_data_transforms(train: bool = False):
    augmentation = []
    if train:
        augmentation = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8),
        ]
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            *augmentation,
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN.squeeze().tolist(), std=IMAGENET_STD.squeeze().tolist()),
        ]
    )


def confusion_matrix_np(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    for truth, pred in zip(y_true, y_pred):
        matrix[int(truth), int(pred)] += 1
    return matrix


def precision_recall_f1_from_cm(matrix: np.ndarray) -> Dict[str, float]:
    precisions = []
    recalls = []
    f1_scores = []
    for idx in range(matrix.shape[0]):
        tp = float(matrix[idx, idx])
        fp = float(matrix[:, idx].sum() - tp)
        fn = float(matrix[idx, :].sum() - tp)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = (2.0 * precision * recall) / (precision + recall + 1e-8)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    return {
        "precision_macro": float(np.mean(precisions)),
        "recall_macro": float(np.mean(recalls)),
        "f1_macro": float(np.mean(f1_scores)),
    }


def binary_auc(y_true_binary: np.ndarray, scores: np.ndarray) -> Optional[float]:
    positives = float(np.sum(y_true_binary == 1))
    negatives = float(np.sum(y_true_binary == 0))
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores)
    sorted_truth = y_true_binary[order]
    cum_neg = 0.0
    auc = 0.0
    for label in sorted_truth:
        if label == 0:
            cum_neg += 1.0
        else:
            auc += cum_neg
    return float(auc / (positives * negatives))


def multiclass_macro_auc(y_true: Sequence[int], probabilities: np.ndarray, num_classes: int) -> Optional[float]:
    aucs = []
    y_true_arr = np.asarray(y_true)
    for class_idx in range(num_classes):
        class_truth = (y_true_arr == class_idx).astype(np.int32)
        auc = binary_auc(class_truth, probabilities[:, class_idx])
        if auc is not None:
            aucs.append(auc)
    if not aucs:
        return None
    return float(np.mean(aucs))


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, object]:
    model.eval()
    losses = []
    y_true: List[int] = []
    y_pred: List[int] = []
    all_probabilities: List[np.ndarray] = []
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            losses.append(float(loss.item()))
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            predictions = np.argmax(probabilities, axis=1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(predictions.tolist())
            all_probabilities.append(probabilities)

    if not y_true:
        return {"loss": 0.0, "accuracy": 0.0, "metrics": {}, "confusion_matrix": [], "roc_auc": None}

    num_classes = len(set(y_true) | set(y_pred))
    probabilities = np.vstack(all_probabilities) if all_probabilities else np.zeros((0, num_classes))
    matrix = confusion_matrix_np(y_true, y_pred, num_classes=max(num_classes, probabilities.shape[1]))
    metrics = precision_recall_f1_from_cm(matrix)
    accuracy = float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))
    roc_auc = (
        multiclass_macro_auc(y_true, probabilities, num_classes=probabilities.shape[1])
        if probabilities.size > 0
        else None
    )
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": accuracy,
        "metrics": metrics,
        "confusion_matrix": matrix.tolist(),
        "roc_auc": roc_auc,
    }


def train_emotion_model(
    epochs: int,
    batch_size: int,
    learning_rate: float,
    freeze_backbone: bool,
    status_slot,
    progress_slot,
) -> Dict[str, object]:
    device = available_device()
    train_dir = DATASET_DIR / "train"
    val_dir = DATASET_DIR / "val"
    test_dir = DATASET_DIR / "test"

    if not train_dir.exists() or not any(train_dir.iterdir()):
        raise FileNotFoundError(
            "dataset/train was not found. Create dataset/train/<class_name>/image files before training."
        )

    train_dataset = datasets.ImageFolder(train_dir, transform=build_data_transforms(train=True))
    val_dataset = datasets.ImageFolder(val_dir, transform=build_data_transforms(train=False)) if val_dir.exists() else None
    test_dataset = datasets.ImageFolder(test_dir, transform=build_data_transforms(train=False)) if test_dir.exists() else None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0) if test_dataset else None

    class_names = train_dataset.classes
    model = EmotionEfficientNet(num_classes=len(class_names)).to(device)
    if freeze_backbone:
        for name, parameter in model.named_parameters():
            if "classifier" not in name:
                parameter.requires_grad = False

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    history = []
    best_state = None
    best_val_accuracy = -1.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            predictions = outputs.argmax(dim=1)
            running_loss += float(loss.item()) * images.size(0)
            running_correct += int((predictions == labels).sum().item())
            running_total += int(images.size(0))

            steps = len(train_loader)
            fractional_progress = ((epoch * steps) + batch_idx + 1) / max(1, epochs * steps)
            progress_slot.progress(min(1.0, float(fractional_progress)))
            status_slot.info(
                f"Epoch {epoch + 1}/{epochs} | Batch {batch_idx + 1}/{steps} | "
                f"Train Acc {running_correct / max(1, running_total):.3f}"
            )

        train_accuracy = running_correct / max(1, running_total)
        train_loss = running_loss / max(1, running_total)
        val_metrics = evaluate_model(model, val_loader, device) if val_loader else None
        val_accuracy = val_metrics["accuracy"] if val_metrics else train_accuracy

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_accuracy": val_accuracy,
            }
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}

    if best_state is None:
        best_state = {key: value.cpu() for key, value in model.state_dict().items()}

    torch.save({"state_dict": best_state, "class_names": class_names}, MODEL_PATH)
    metadata = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "class_names": class_names,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "freeze_backbone": freeze_backbone,
        "history": history,
    }
    save_json(MODEL_META_PATH, metadata)
    summarize_dataset()
    load_face_model.clear()

    final_model = EmotionEfficientNet(num_classes=len(class_names))
    final_model.load_state_dict(best_state)
    final_model.to(device)

    final_val = evaluate_model(final_model, val_loader, device) if val_loader else None
    final_test = evaluate_model(final_model, test_loader, device) if test_loader else None
    return {"metadata": metadata, "val": final_val, "test": final_test}


def render_metric_grid(fusion: FusionState) -> None:
    metric_cols = st.columns(4)
    metric_cols[0].metric("Fused emotion", fusion.label.title())
    metric_cols[1].metric("Confidence", f"{fusion.confidence:.2f}")
    metric_cols[2].metric("Wellbeing score", f"{fusion.wellbeing_score}/100")
    metric_cols[3].metric("Mode", fusion.mode.title())


def render_mascot(emotion: str, mode: str, speech_text: str = "") -> None:
    palette = {
        "happy": ("#fde68a", "#f59e0b"),
        "sad": ("#bfdbfe", "#2563eb"),
        "angry": ("#fecaca", "#dc2626"),
        "anxious": ("#ddd6fe", "#7c3aed"),
        "neutral": ("#d1fae5", "#059669"),
    }
    background, accent = palette.get(emotion, palette["neutral"])
    mouth = {
        "happy": "border-bottom: 10px solid #111827; border-radius: 0 0 60px 60px;",
        "sad": "border-top: 10px solid #111827; border-radius: 60px 60px 0 0;",
        "angry": "border-top: 6px solid #111827; transform: rotate(4deg);",
        "anxious": "border: 6px solid #111827; border-color: #111827 transparent transparent transparent; border-radius: 50%;",
        "neutral": "border-top: 6px solid #111827;",
    }.get(emotion, "border-top: 6px solid #111827;")
    animation = {
        "happy": "floaty 1.6s ease-in-out infinite",
        "sad": "sway 2.6s ease-in-out infinite",
        "angry": "shake 0.45s ease-in-out infinite",
        "anxious": "pulse 1.4s ease-in-out infinite",
        "neutral": "floaty 2.0s ease-in-out infinite",
    }.get(emotion, "floaty 2.0s ease-in-out infinite")
    speech = clean_text(speech_text)[:240]
    bubble = (
        f'<div class="mascot-bubble"><strong>AARA</strong><br>{speech}</div>'
        if speech
        else '<div class="mascot-bubble"><strong>AARA</strong><br>Ready to listen and respond.</div>'
    )

    html = f"""
    <style>
      .mascot-wrap {{
        display: flex;
        gap: 18px;
        align-items: center;
        margin: 8px 0 18px 0;
      }}
      .mascot {{
        width: 180px;
        height: 180px;
        border-radius: 50%;
        background: {background};
        border: 6px solid {accent};
        position: relative;
        animation: {animation};
        box-shadow: 0 16px 40px rgba(15, 23, 42, 0.16);
      }}
      .eye {{
        width: 18px;
        height: 18px;
        background: #111827;
        border-radius: 50%;
        position: absolute;
        top: 58px;
      }}
      .eye.left {{ left: 48px; }}
      .eye.right {{ right: 48px; }}
      .brow {{
        width: 36px;
        height: 8px;
        background: #111827;
        border-radius: 8px;
        position: absolute;
        top: 40px;
      }}
      .brow.left {{ left: 38px; transform: rotate(-12deg); }}
      .brow.right {{ right: 38px; transform: rotate(12deg); }}
      .mouth {{
        width: 72px;
        height: 34px;
        position: absolute;
        left: 54px;
        top: 112px;
        {mouth}
      }}
      .mascot-bubble {{
        flex: 1;
        padding: 16px 18px;
        border-radius: 18px;
        background: rgba(15, 23, 42, 0.05);
        border: 1px solid rgba(15, 23, 42, 0.10);
        min-height: 92px;
      }}
      @keyframes floaty {{
        0%, 100% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(-8px); }}
      }}
      @keyframes sway {{
        0%, 100% {{ transform: rotate(0deg); }}
        50% {{ transform: rotate(-2deg); }}
      }}
      @keyframes shake {{
        0%, 100% {{ transform: translateX(0); }}
        25% {{ transform: translateX(-3px); }}
        75% {{ transform: translateX(3px); }}
      }}
      @keyframes pulse {{
        0%, 100% {{ transform: scale(1.0); }}
        50% {{ transform: scale(1.03); }}
      }}
    </style>
    <div class="mascot-wrap">
      <div class="mascot">
        <div class="brow left"></div>
        <div class="brow right"></div>
        <div class="eye left"></div>
        <div class="eye right"></div>
        <div class="mouth"></div>
      </div>
      {bubble}
    </div>
    <div style="margin-top:-8px; color:#475569; font-size:0.95rem;">
      Animated emotional state: <strong>{emotion.title()}</strong> | Adaptive tone: <strong>{mode.title()}</strong>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def initialize_session() -> None:
    ensure_directories()
    defaults = {
        "messages": [],
        "asked_questions": set(),
        "recommended_urls": set(),
        "last_reply": "",
        "last_resources": [],
        "last_fusion": None,
        "last_text_result": None,
        "last_voice_result": None,
        "last_face_result": None,
        "last_questions": [],
        "last_ethics": [],
        "voice_identity_match": False,
        "face_preview_temp": None,
        "gradcam_preview_temp": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def matrix_to_table(matrix: List[List[int]], class_names: List[str]) -> List[Dict[str, int]]:
    rows = []
    for idx, row in enumerate(matrix):
        item = {"actual": class_names[idx] if idx < len(class_names) else f"class_{idx}"}
        for pred_idx, value in enumerate(row):
            column_name = class_names[pred_idx] if pred_idx < len(class_names) else f"class_{pred_idx}"
            item[f"pred_{column_name}"] = value
        rows.append(item)
    return rows


def render_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def render_sidebar() -> Tuple[str, str, bool]:
    with st.sidebar:
        st.header("Runtime settings")
        username = st.text_input("User name", value="Guneesh")
        hf_token = st.text_input(
            "Optional Hugging Face token",
            value=os.getenv("HF_TOKEN", ""),
            type="password",
            help="Use a free Hugging Face access token for cloud text generation. Public local models work without it.",
        )
        enable_tts = st.checkbox("Enable local text-to-speech", value=False)
        st.markdown(
            """
            Free API guidance:
            - Hugging Face is the recommended free option here.
            - Use a personal access token, not your account password.
            - Spotify is kept as search-link integration so the demo stays free.
            """
        )
        if st.button("Export resource catalog"):
            count = export_resource_catalog()
            st.success(f"Exported {count} catalog entries to resource_catalog.csv")
        summarize_dataset()
    return username, hf_token, enable_tts


def handle_feedback_buttons(username: str, mood: str) -> None:
    resources = st.session_state.get("last_resources", [])
    if not resources:
        return
    st.markdown("Recommendation feedback")
    feedback_cols = st.columns(len(resources))
    for idx, item in enumerate(resources):
        with feedback_cols[idx]:
            st.caption(item["title"])
            if st.button(f"Like {idx + 1}", key=f"like_{idx}"):
                log_recommendation_feedback(username, mood, item, "liked")
                st.success("Feedback saved")
            if st.button(f"Skip {idx + 1}", key=f"skip_{idx}"):
                log_recommendation_feedback(username, mood, item, "skipped")
                st.info("Feedback saved")


def render_streamlit_app() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon=":material/psychology:", layout="wide")
    initialize_session()
    username, hf_token, enable_tts = render_sidebar()

    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)

    tabs = st.tabs(["Interactive assistant", "Emotion dashboard", "Training and evaluation", "Project notes"])

    with tabs[0]:
        col_left, col_right = st.columns([1.2, 0.8], gap="large")
        with col_left:
            text_input = st.text_area(
                "Describe how you feel, what happened, or what support you need",
                height=120,
                placeholder="Example: I am stressed about my final defence and I need a focused plan that also calms me down.",
            )
            audio_file = st.audio_input("Voice input")
            camera_file = st.camera_input("Capture facial expression")
            submit = st.button("Analyze emotion and respond", type="primary")

            render_history()

        with col_right:
            fusion_for_display = st.session_state.last_fusion or FusionState(
                label="neutral",
                confidence=0.0,
                valence=0.56,
                arousal=0.42,
                wellbeing_score=56,
                mode="friendly",
                notes=["Awaiting user input."],
            )
            render_mascot(fusion_for_display.label, fusion_for_display.mode, st.session_state.last_reply)

        if submit:
            text_result = detect_text_emotion(text_input, hf_token)

            if audio_file is not None:
                audio_bytes = audio_file.read()
                voice_result = detect_voice_emotion(audio_bytes, hf_token)
                voice_identity_match = bool(
                    username.strip()
                    and isinstance(voice_result.details.get("transcript"), str)
                    and username.lower() in str(voice_result.details.get("transcript")).lower()
                )
            else:
                voice_result = ModalityEmotion(
                    source="voice",
                    label="neutral",
                    confidence=0.0,
                    scores={"neutral": 1.0},
                    details={"transcript": "No voice input provided."},
                )
                voice_identity_match = False

            if camera_file is not None:
                frame_bytes = np.frombuffer(camera_file.getvalue(), dtype=np.uint8)
                frame_bgr = cv2.imdecode(frame_bytes, cv2.IMREAD_COLOR)
                face_result, face_preview, gradcam_preview, face_trained = detect_face_emotion(frame_bgr)
                st.session_state["face_preview_temp"] = face_preview
                st.session_state["gradcam_preview_temp"] = gradcam_preview
            else:
                face_result = ModalityEmotion(
                    source="face",
                    label="neutral",
                    confidence=0.0,
                    scores={"neutral": 1.0},
                    details={"note": "No camera input provided."},
                )
                face_preview, gradcam_preview, face_trained = None, None, False
                st.session_state["face_preview_temp"] = None
                st.session_state["gradcam_preview_temp"] = None

            fusion = fuse_emotions([text_result, voice_result, face_result])
            intent, _, context_terms = infer_intent(text_input or str(voice_result.details.get("transcript", "")))
            questions = select_questions(intent, fusion.label, st.session_state.asked_questions)
            resources = recommend_resources(fusion.label, st.session_state.recommended_urls)
            final_user_text = clean_text(text_input) or clean_text(str(voice_result.details.get("transcript", "")))
            reply = generate_chatbot_reply(
                username=username,
                user_text=final_user_text or "The user opened the app and needs general support.",
                fusion=fusion,
                intent=intent,
                context_terms=context_terms,
                questions=questions,
                resources=resources,
                history=st.session_state.messages,
                hf_token=hf_token,
            )
            ethical_notes = ethical_ai_monitoring([text_result, voice_result, face_result], fusion, face_trained)

            if final_user_text:
                st.session_state.messages.append({"role": "user", "content": final_user_text})
            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.session_state.last_reply = reply
            st.session_state.last_resources = resources
            st.session_state.last_fusion = fusion
            st.session_state.last_text_result = text_result
            st.session_state.last_voice_result = voice_result
            st.session_state.last_face_result = face_result
            st.session_state.last_questions = questions
            st.session_state.last_ethics = ethical_notes
            st.session_state.voice_identity_match = voice_identity_match

            log_digital_twin(username, final_user_text, text_result, voice_result, face_result, fusion, reply)

            if enable_tts and reply:
                spoken, message = maybe_speak_text(reply)
                st.info(message if spoken else message)

            st.rerun()

    with tabs[1]:
        fusion = st.session_state.last_fusion
        if fusion is None:
            st.info("Submit a text, voice, or face input to populate the live dashboard.")
        else:
            render_metric_grid(fusion)
            text_result = st.session_state.last_text_result
            voice_result = st.session_state.last_voice_result
            face_result = st.session_state.last_face_result
            ethics = st.session_state.last_ethics

            detail_cols = st.columns(3)
            with detail_cols[0]:
                st.subheader("Text emotion")
                st.json(asdict(text_result))
            with detail_cols[1]:
                st.subheader("Voice emotion")
                st.json(asdict(voice_result))
                st.write(
                    f"Voice identity match with username: {'Yes' if st.session_state.voice_identity_match else 'No'}"
                )
            with detail_cols[2]:
                st.subheader("Face emotion")
                st.json(asdict(face_result))

            if st.session_state.get("face_preview_temp") is not None:
                st.image(st.session_state["face_preview_temp"], caption="Face detection preview", use_container_width=True)

            if st.session_state.get("gradcam_preview_temp") is not None:
                st.image(st.session_state["gradcam_preview_temp"], caption="Grad-CAM overlay", use_container_width=True)

            st.subheader("Ethical AI monitoring")
            for note in ethics:
                st.write(f"- {note}")

            st.subheader("Personalized follow-up questions")
            for question in st.session_state.get("last_questions", []):
                st.write(f"- {question}")

            st.subheader("Adaptive lifestyle suggestions")
            for item in st.session_state.last_resources:
                st.markdown(f"- [{item['title']}]({item['url']}) via {item['source']}")
            handle_feedback_buttons(username, fusion.label)

    with tabs[2]:
        st.subheader("Dataset structure")
        st.code(
            "dataset/\n"
            "  train/\n"
            "    happy/\n"
            "    sad/\n"
            "    angry/\n"
            "    neutral/\n"
            "  val/\n"
            "  test/\n",
            language="text",
        )
        manifest = summarize_dataset()
        st.json(manifest)

        st.subheader("Train EfficientNetV2-S")
        train_cols = st.columns(4)
        epochs = train_cols[0].number_input("Epochs", min_value=1, max_value=10, value=2, step=1)
        batch_size = train_cols[1].selectbox("Batch size", [4, 8, 16], index=1)
        learning_rate = train_cols[2].selectbox("Learning rate", [1e-4, 2e-4, 5e-4, 1e-3], index=1)
        freeze_backbone = train_cols[3].checkbox("Freeze backbone", value=True)
        status_slot = st.empty()
        progress_slot = st.progress(0.0)

        if st.button("Start training"):
            try:
                results = train_emotion_model(
                    epochs=int(epochs),
                    batch_size=int(batch_size),
                    learning_rate=float(learning_rate),
                    freeze_backbone=freeze_backbone,
                    status_slot=status_slot,
                    progress_slot=progress_slot,
                )
                status_slot.success("Training completed and the .pth checkpoint was saved.")
                st.json(results["metadata"])
                if results["val"]:
                    st.subheader("Validation metrics")
                    st.json(results["val"])
                if results["test"]:
                    st.subheader("Test metrics")
                    st.json(results["test"])
                    class_names = results["metadata"]["class_names"]
                    matrix = results["test"]["confusion_matrix"]
                    if matrix:
                        st.table(matrix_to_table(matrix, class_names))
            except Exception as exc:
                status_slot.error(str(exc))

        st.subheader("Grad-CAM note")
        st.write(
            "This project uses EfficientNetV2-S with built-in global average pooling before the classifier. "
            "Grad-CAM is generated from the last feature stage so viva examiners can inspect which facial region "
            "drove the emotional prediction."
        )

    with tabs[3]:
        st.subheader("Free API recommendation")
        st.write(
            "Hugging Face is the best free API option for this project because public models can run locally "
            "through transformers and optional cloud inference can be enabled with a free access token. "
            "Using passwords directly is not recommended; use an access token stored in the HF_TOKEN environment variable."
        )
        st.subheader("Windows 11 execution notes")
        st.write(
            "For an Intel i5 laptop with 8 GB RAM, stay in sample-size mode, keep batch size at 4 or 8, and begin with 1 to 3 epochs. "
            "The current app is optimized for local Streamlit demos rather than large-scale training."
        )
        st.subheader("APK conversion trick")
        st.write(
            "The free practical route is to run the Streamlit app locally or on a lightweight host, open it in Chrome or Edge on Android, "
            "and use Add to Home Screen or Install App. This gives an app-like experience without maintaining a separate Android codebase."
        )
def cli_main() -> bool:
    ensure_directories()
    if "--export-catalog" in sys.argv:
        count = export_resource_catalog()
        print(f"Exported {count} entries to {RESOURCE_CATALOG_PATH.name}")
        return True
    return False


if __name__ == "__main__":
    if not cli_main():
        render_streamlit_app()
