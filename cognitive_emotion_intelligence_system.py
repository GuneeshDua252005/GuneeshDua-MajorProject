#!/usr/bin/env python3
"""
Cognitive Emotion Intelligence and Adaptive Lifestyle System

Single-file academic project prototype that combines:
- text-based emotion understanding
- adaptive lifestyle planning
- free-tier Hugging Face integration for important question generation
- optional Spotify metadata search
- digital emotional twin tracking
- ethical AI monitoring
- reinforcement-learning style feedback adaptation
- training-ready EfficientNetV2-S + Grad-CAM utilities for facial emotion analysis

The core text workflow runs with standard-library Python plus requests.
Deep-learning features are optional and activate only when the required
packages are installed locally.
"""

from __future__ import annotations

import base64
import json
import os
import random
import textwrap
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests


APP_NAME = "Cognitive Emotion Intelligence and Adaptive Lifestyle System"
APP_VERSION = "1.0.0"
CANONICAL_TEXT_EMOTIONS = [
    "happy",
    "sad",
    "angry",
    "anxious",
    "calm",
    "overwhelmed",
    "motivated",
    "neutral",
]
IMAGE_EMOTION_TO_TEXT = {
    "angry": "angry",
    "disgust": "overwhelmed",
    "fear": "anxious",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "surprise": "motivated",
}
RL_ACTIONS = [
    "focused_study_block",
    "deep_breathing_break",
    "walk_and_reset",
    "reflective_journaling",
    "hydration_and_nutrition",
    "calm_music_session",
    "motivational_playlist",
    "support_contact_prompt",
]
DEFAULT_HF_MODEL = os.getenv("HF_LLM_MODEL", "google/flan-t5-base")


TRACK_LIBRARY = {
    "happy": [
        {"title": "Upbeat study mix", "artist": "offline library", "query": "happy uplifting study songs"},
        {"title": "Positive focus session", "artist": "offline library", "query": "energetic focus playlist"},
    ],
    "sad": [
        {"title": "Gentle recovery set", "artist": "offline library", "query": "comforting calm acoustic songs"},
        {"title": "Reflective piano set", "artist": "offline library", "query": "soft piano healing playlist"},
    ],
    "angry": [
        {"title": "Decompress and cool down", "artist": "offline library", "query": "calm down ambient playlist"},
        {"title": "Reset rhythm session", "artist": "offline library", "query": "slow lo fi stress relief"},
    ],
    "anxious": [
        {"title": "Exam calm playlist", "artist": "offline library", "query": "anxiety relief instrumental study"},
        {"title": "Breathing support mix", "artist": "offline library", "query": "meditation ambient focus"},
    ],
    "calm": [
        {"title": "Steady concentration loop", "artist": "offline library", "query": "deep work instrumental focus"},
        {"title": "Mindful coding playlist", "artist": "offline library", "query": "calm coding instrumental"},
    ],
    "overwhelmed": [
        {"title": "Slow down and recover", "artist": "offline library", "query": "soft relaxing no lyrics playlist"},
        {"title": "Mental declutter mix", "artist": "offline library", "query": "minimal piano stress relief"},
    ],
    "motivated": [
        {"title": "Momentum builder", "artist": "offline library", "query": "productive motivational playlist"},
        {"title": "Goal mode focus", "artist": "offline library", "query": "power focus coding mix"},
    ],
    "neutral": [
        {"title": "Balanced background focus", "artist": "offline library", "query": "balanced instrumental work playlist"},
        {"title": "Daily routine mix", "artist": "offline library", "query": "casual productive playlist"},
    ],
}

QUESTION_BANK = {
    "happy": [
        "Which activity helped you feel this positive today, and how can you repeat it tomorrow?",
        "How can you use your present emotional strength to complete one important academic task?",
        "Which person or environment is amplifying your good mood in a healthy way?",
    ],
    "sad": [
        "What specific event or thought seems to be lowering your mood right now?",
        "What is one small task you can finish today that may restore a sense of progress?",
        "Would talking to a trusted friend, mentor, or family member reduce the emotional load?",
    ],
    "angry": [
        "What exactly triggered this frustration: workload, people, delay, or self-pressure?",
        "Which part of the situation can you control within the next one hour?",
        "What response would protect your dignity without damaging relationships?",
    ],
    "anxious": [
        "Is your anxiety linked to uncertainty, lack of preparation, comparison, or fear of failure?",
        "What evidence shows that the situation is difficult but still manageable?",
        "What would be the most useful next step instead of trying to solve everything at once?",
    ],
    "calm": [
        "How can you protect this calm state while moving toward your next priority?",
        "Which focused task fits best with your current mental clarity?",
        "What habit helped you feel balanced today?",
    ],
    "overwhelmed": [
        "Which three tasks are real priorities, and which ones can wait?",
        "What sign is telling you that your system needs rest instead of more pressure?",
        "Can you divide your current problem into a 15-minute starting step?",
    ],
    "motivated": [
        "What is the most valuable target to attack while your momentum is high?",
        "How will you convert motivation into a routine instead of a one-time burst?",
        "Which distraction must be removed to preserve your current drive?",
    ],
    "neutral": [
        "What outcome would make today feel meaningful by the evening?",
        "Do you need stimulation, rest, clarity, or structure right now?",
        "What single task would move your project forward with the least resistance?",
    ],
}

LIFESTYLE_LIBRARY = {
    "happy": [
        "Use the positive state for one high-value learning task.",
        "Document what went well in a short gratitude note.",
        "Avoid overcommitting just because you currently feel energetic.",
    ],
    "sad": [
        "Lower the task barrier and start with a 10-minute warm-up task.",
        "Use natural light, hydration, and a brief walk to interrupt emotional inertia.",
        "Choose a supportive human conversation instead of isolation if the feeling persists.",
    ],
    "angry": [
        "Delay important replies until your physiological arousal is lower.",
        "Do one cycle of box breathing and step away from triggering inputs.",
        "Convert frustration into a bounded action list with a time limit.",
    ],
    "anxious": [
        "Use a 3-step grounding sequence: breathe, prioritize, begin.",
        "Break work into smaller chunks and finish one visible unit.",
        "Reduce doom-scrolling and high-noise notifications for the next hour.",
    ],
    "calm": [
        "Enter a deep work block while your cognitive load is balanced.",
        "Protect your routine by keeping the next steps simple and scheduled.",
        "Use this stable state to plan tomorrow's priorities.",
    ],
    "overwhelmed": [
        "Cut the current plan down to essentials only.",
        "Take a recovery break before attempting another complex decision.",
        "Shift from perfect planning to minimum viable progress.",
    ],
    "motivated": [
        "Use a timer and attack the most difficult task first.",
        "Track a measurable milestone while your drive is strong.",
        "Pair intense work with a realistic recovery point to prevent burnout.",
    ],
    "neutral": [
        "Use structure rather than waiting for strong emotion.",
        "Create a short task list with one easy start and one meaningful finish.",
        "Add either stimulation or rest depending on your energy score.",
    ],
}

TEXT_KEYWORDS = {
    "happy": ["happy", "joy", "grateful", "excited", "good", "great", "amazing", "relaxed", "smile"],
    "sad": ["sad", "down", "cry", "lonely", "empty", "hurt", "unhappy", "low", "upset"],
    "angry": ["angry", "frustrated", "irritated", "annoyed", "rage", "mad", "furious"],
    "anxious": ["anxious", "nervous", "worried", "stress", "panic", "fear", "scared", "tense"],
    "calm": ["calm", "peaceful", "balanced", "steady", "focused", "mindful"],
    "overwhelmed": ["overwhelmed", "burnout", "exhausted", "too much", "drained", "tired", "confused"],
    "motivated": ["motivated", "determined", "productive", "ready", "disciplined", "confident"],
}


@dataclass
class UserContext:
    name: str
    role: str
    study_pressure: int
    sleep_hours: float
    energy_level: int
    screen_time_hours: float
    current_goal: str
    journal_text: str
    consent: bool = True


@dataclass
class AnalysisResult:
    primary_emotion: str
    confidence: float
    emotion_scores: Dict[str, float]
    reasoning: List[str] = field(default_factory=list)
    source: str = "heuristic"


@dataclass
class AdaptivePlan:
    chosen_action: str
    questions: List[str]
    lifestyle_steps: List[str]
    music_suggestions: List[Dict[str, str]]
    mood_summary: str
    ethical_notice: str


def safe_int(raw: str, default: int, minimum: int = 0, maximum: int = 10) -> int:
    try:
        value = int(raw.strip())
        return max(minimum, min(maximum, value))
    except Exception:
        return default


def safe_float(raw: str, default: float, minimum: float = 0.0, maximum: float = 24.0) -> float:
    try:
        value = float(raw.strip())
        return max(minimum, min(maximum, value))
    except Exception:
        return default


def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(value, 0.0) for value in scores.values())
    if total <= 0:
        uniform = 1.0 / len(CANONICAL_TEXT_EMOTIONS)
        return {emotion: uniform for emotion in CANONICAL_TEXT_EMOTIONS}
    return {emotion: max(value, 0.0) / total for emotion, value in scores.items()}


def pretty_label(value: str) -> str:
    return value.replace("_", " ").title()


def prompt_input(message: str, default: str = "") -> str:
    try:
        return input(message)
    except EOFError:
        return default


class EthicalAIMonitor:
    def assess(self, context: UserContext, result: AnalysisResult) -> str:
        notes: List[str] = []
        if not context.consent:
            notes.append("consent not granted")
        if len(context.journal_text.strip()) < 15:
            notes.append("very short text may reduce confidence")
        if result.confidence < 0.42:
            notes.append("prediction confidence is limited, so human judgment should dominate")
        if context.study_pressure >= 8 and result.primary_emotion in {"sad", "overwhelmed", "anxious"}:
            notes.append("high stress detected; if distress persists, contact a trusted human support system")
        if not notes:
            notes.append("privacy-first local execution path is active and recommendations are advisory")
        return "; ".join(notes)


class TextEmotionAnalyzer:
    """
    Offline-first analyzer.

    The implementation uses a keyword-and-context scoring layer by default.
    If transformers is installed and a compatible model is available locally,
    the same class can be extended without changing the rest of the program.
    """

    def __init__(self) -> None:
        self.keyword_map = TEXT_KEYWORDS

    def analyze(self, text: str) -> AnalysisResult:
        lowered = text.lower()
        scores: Dict[str, float] = {emotion: 0.1 for emotion in CANONICAL_TEXT_EMOTIONS}
        reasoning: List[str] = []

        for emotion, keywords in self.keyword_map.items():
            hits = sum(lowered.count(keyword) for keyword in keywords)
            if hits:
                scores[emotion] += hits * 1.4
                reasoning.append(f"keyword evidence supports {emotion}")

        if any(term in lowered for term in ["exam", "deadline", "submission", "project", "viva"]):
            scores["anxious"] += 1.0
            scores["motivated"] += 0.4
            reasoning.append("academic pressure terms detected")

        if any(term in lowered for term in ["sleep", "late night", "insomnia", "restless"]):
            scores["overwhelmed"] += 0.8
            scores["sad"] += 0.3
            reasoning.append("sleep-related stress indicators detected")

        if any(term in lowered for term in ["plan", "goal", "improve", "finish", "complete"]):
            scores["motivated"] += 0.7
            reasoning.append("goal-oriented language detected")

        if "not" in lowered or "never" in lowered:
            scores["sad"] += 0.2
            scores["anxious"] += 0.2

        normalized = normalize_scores(scores)
        primary_emotion, confidence = max(normalized.items(), key=lambda item: item[1])
        confidence = round(confidence, 3)
        return AnalysisResult(
            primary_emotion=primary_emotion,
            confidence=confidence,
            emotion_scores=normalized,
            reasoning=reasoning[:5] or ["baseline heuristic analysis used"],
            source="heuristic-text",
        )


class HuggingFaceQuestionGenerator:
    """
    Uses the free Hugging Face Inference API when the user supplies an HF_TOKEN.
    Falls back to high-quality rule templates when no token is configured.
    """

    def __init__(self, token: Optional[str], model: Optional[str] = None, timeout: int = 60) -> None:
        self.token = (token or "").strip()
        self.model = (model or DEFAULT_HF_MODEL).strip()
        self.timeout = timeout

    def is_enabled(self) -> bool:
        return bool(self.token)

    def generate_questions(self, context: UserContext, analysis: AnalysisResult) -> List[str]:
        if self.is_enabled():
            generated = self._call_hf(context, analysis)
            if generated:
                return generated
        return self._fallback_questions(context, analysis)

    def _call_hf(self, context: UserContext, analysis: AnalysisResult) -> List[str]:
        prompt = textwrap.dedent(
            f"""
            Generate exactly 5 short but important counseling-style reflective questions.
            User role: {context.role}
            Current goal: {context.current_goal}
            Emotion: {analysis.primary_emotion}
            Confidence: {analysis.confidence}
            Journal: {context.journal_text}

            The questions should be practical, respectful, and focused on mindset,
            mood, study pressure, productivity, and emotional clarity.
            Return plain numbered lines only.
            """
        ).strip()

        headers = {"Authorization": f"Bearer {self.token}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 220,
                "temperature": 0.5,
                "return_full_text": False,
            },
        }
        url = f"https://api-inference.huggingface.co/models/{self.model}"
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            response.raise_for_status()
            body = response.json()
            text = self._extract_generated_text(body)
            if not text:
                return []
            lines = []
            for raw in text.splitlines():
                clean = raw.strip().lstrip("-").strip()
                if not clean:
                    continue
                if clean[0].isdigit():
                    clean = clean.split(".", 1)[-1].strip()
                lines.append(clean)
            unique_lines = []
            for line in lines:
                if line not in unique_lines:
                    unique_lines.append(line)
            return unique_lines[:5]
        except Exception:
            return []

    @staticmethod
    def _extract_generated_text(payload: Any) -> str:
        if isinstance(payload, list) and payload:
            first = payload[0]
            if isinstance(first, dict):
                return str(first.get("generated_text", "")).strip()
        if isinstance(payload, dict):
            if "generated_text" in payload:
                return str(payload.get("generated_text", "")).strip()
            if "error" in payload:
                return ""
        return ""

    @staticmethod
    def _fallback_questions(context: UserContext, analysis: AnalysisResult) -> List[str]:
        bank = QUESTION_BANK.get(analysis.primary_emotion, QUESTION_BANK["neutral"])
        custom = [
            f"What is the most realistic next step for your goal: {context.current_goal}?",
            f"How is your current emotional state affecting your role as a {context.role}?",
        ]
        merged = bank + custom
        deduplicated = []
        for question in merged:
            if question not in deduplicated:
                deduplicated.append(question)
        return deduplicated[:5]


class SpotifyMusicRecommender:
    def __init__(self, client_id: str, client_secret: str) -> None:
        self.client_id = (client_id or "").strip()
        self.client_secret = (client_secret or "").strip()
        self.cached_token: Optional[Tuple[str, float]] = None

    def recommend(self, emotion: str, limit: int = 5) -> List[Dict[str, str]]:
        offline = TRACK_LIBRARY.get(emotion, TRACK_LIBRARY["neutral"])
        if not self.client_id or not self.client_secret:
            return offline[:limit]

        token = self._get_token()
        if not token:
            return offline[:limit]

        query = TRACK_LIBRARY.get(emotion, TRACK_LIBRARY["neutral"])[0]["query"]
        headers = {"Authorization": f"Bearer {token}"}
        params = {"q": query, "type": "track", "limit": limit}
        try:
            response = requests.get(
                "https://api.spotify.com/v1/search",
                headers=headers,
                params=params,
                timeout=30,
            )
            response.raise_for_status()
            body = response.json()
            items = body.get("tracks", {}).get("items", [])
            output = []
            for item in items:
                artists = ", ".join(artist["name"] for artist in item.get("artists", []))
                output.append(
                    {
                        "title": item.get("name", "Unknown Track"),
                        "artist": artists or "Unknown Artist",
                        "url": item.get("external_urls", {}).get("spotify", ""),
                    }
                )
            return output or offline[:limit]
        except Exception:
            return offline[:limit]

    def _get_token(self) -> Optional[str]:
        now = time.time()
        if self.cached_token and self.cached_token[1] > now:
            return self.cached_token[0]

        auth_header = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode("utf-8")).decode("utf-8")
        headers = {"Authorization": f"Basic {auth_header}"}
        data = {"grant_type": "client_credentials"}
        try:
            response = requests.post(
                "https://accounts.spotify.com/api/token",
                headers=headers,
                data=data,
                timeout=30,
            )
            response.raise_for_status()
            body = response.json()
            token = body.get("access_token")
            expires_in = int(body.get("expires_in", 3600))
            if token:
                self.cached_token = (token, now + max(60, expires_in - 60))
            return token
        except Exception:
            return None


class AdaptivePolicyEngine:
    def __init__(self, policy_path: Path) -> None:
        self.policy_path = policy_path
        self.q_table: Dict[str, Dict[str, float]] = self._load()

    def choose_action(self, context: UserContext, emotion: str) -> str:
        state = self._build_state(context, emotion)
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in RL_ACTIONS}
        ranked = sorted(self.q_table[state].items(), key=lambda item: item[1], reverse=True)
        best_action = ranked[0][0]

        if emotion in {"sad", "overwhelmed"} and context.energy_level <= 4:
            return "walk_and_reset"
        if emotion == "anxious":
            return "deep_breathing_break"
        if emotion == "motivated":
            return "focused_study_block"
        return best_action

    def update_feedback(self, context: UserContext, emotion: str, action: str, reward: float) -> None:
        state = self._build_state(context, emotion)
        if state not in self.q_table:
            self.q_table[state] = {name: 0.0 for name in RL_ACTIONS}
        current = self.q_table[state].get(action, 0.0)
        self.q_table[state][action] = round(current + 0.3 * (reward - current), 4)
        self._save()

    def describe_action(self, action: str) -> str:
        descriptions = {
            "focused_study_block": "Start a 25-minute distraction-free study or coding sprint.",
            "deep_breathing_break": "Use box breathing for 3 minutes before restarting work.",
            "walk_and_reset": "Take a short walk, hydrate, and return with a smaller task.",
            "reflective_journaling": "Write a quick emotion-to-action journal entry.",
            "hydration_and_nutrition": "Refuel physically before making another difficult decision.",
            "calm_music_session": "Use low-arousal instrumental music to settle your state.",
            "motivational_playlist": "Use energetic background music while tackling a hard task.",
            "support_contact_prompt": "Reach out to a trusted friend, mentor, or family member.",
        }
        return descriptions.get(action, action.replace("_", " "))

    def _build_state(self, context: UserContext, emotion: str) -> str:
        pressure_band = "high" if context.study_pressure >= 7 else "mid" if context.study_pressure >= 4 else "low"
        sleep_band = "low_sleep" if context.sleep_hours < 6 else "rested"
        energy_band = "low_energy" if context.energy_level < 5 else "high_energy"
        return f"{emotion}|{pressure_band}|{sleep_band}|{energy_band}"

    def _load(self) -> Dict[str, Dict[str, float]]:
        if not self.policy_path.exists():
            return {}
        try:
            return json.loads(self.policy_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save(self) -> None:
        self.policy_path.write_text(json.dumps(self.q_table, indent=2), encoding="utf-8")


class DigitalEmotionalTwin:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.state = self._load()

    def record(self, context: UserContext, analysis: AnalysisResult, chosen_action: str) -> None:
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "name": context.name,
            "role": context.role,
            "goal": context.current_goal,
            "emotion": analysis.primary_emotion,
            "confidence": analysis.confidence,
            "action": chosen_action,
            "study_pressure": context.study_pressure,
            "sleep_hours": context.sleep_hours,
            "energy_level": context.energy_level,
        }
        history = self.state.setdefault("history", [])
        history.append(entry)
        self.state["history"] = history[-30:]
        self.state["emotion_counts"] = dict(Counter(item["emotion"] for item in self.state["history"]))
        self.path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")

    def summary(self) -> str:
        history = self.state.get("history", [])
        if not history:
            return "No emotional twin history is available yet."
        counts = self.state.get("emotion_counts", {})
        dominant = max(counts.items(), key=lambda item: item[1])[0]
        last = history[-1]
        return (
            f"Recent pattern: dominant emotion is {dominant}. "
            f"Latest session focused on '{last['goal']}' with action '{last['action']}'."
        )

    def _load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {"history": [], "emotion_counts": {}}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {"history": [], "emotion_counts": {}}


class MultiAIFusionEngine:
    def fuse(self, text_result: AnalysisResult, image_result: Optional[Dict[str, float]] = None) -> AnalysisResult:
        combined = defaultdict(float)
        for emotion, score in text_result.emotion_scores.items():
            combined[emotion] += score * 0.7

        if image_result:
            for image_emotion, score in image_result.items():
                mapped = IMAGE_EMOTION_TO_TEXT.get(image_emotion, "neutral")
                combined[mapped] += score * 0.3

        normalized = normalize_scores(dict(combined))
        primary_emotion, confidence = max(normalized.items(), key=lambda item: item[1])
        reasoning = list(text_result.reasoning)
        if image_result:
            reasoning.append("multimodal fusion combined text emotion and facial emotion evidence")
        return AnalysisResult(
            primary_emotion=primary_emotion,
            confidence=round(confidence, 3),
            emotion_scores=normalized,
            reasoning=reasoning[:6],
            source="multimodal-fusion" if image_result else text_result.source,
        )


class VisionEmotionModule:
    """
    Optional PyTorch module.
    Expected dataset layout:
        data/fer/train/<class_name>/*.png
        data/fer/val/<class_name>/*.png
    """

    class_names = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

    @staticmethod
    def _require_dependencies() -> Tuple[Any, Any, Any, Any]:
        try:
            import torch
            from PIL import Image
            from torchvision import datasets, models, transforms
        except ImportError as exc:
            raise RuntimeError(
                "Install torch, torchvision, and pillow to use training and Grad-CAM features."
            ) from exc
        return torch, Image, datasets, (models, transforms)

    def build_model(self, num_classes: int = 7, pretrained: bool = True) -> Any:
        torch, _, _, misc = self._require_dependencies()
        models, _ = misc
        weights = None
        if pretrained:
            try:
                weights = models.EfficientNet_V2_S_Weights.DEFAULT
            except Exception:
                weights = None
        model = models.efficientnet_v2_s(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, num_classes)
        return model

    def train(
        self,
        data_root: str,
        output_model: str = "artifacts/efficientnetv2_emotion.pt",
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 1e-4,
    ) -> Dict[str, Any]:
        torch, _, datasets, misc = self._require_dependencies()
        models, transforms = misc
        del models

        train_dir = Path(data_root) / "train"
        val_dir = Path(data_root) / "val"
        if not train_dir.exists() or not val_dir.exists():
            raise RuntimeError("Dataset folders were not found. Expected train and val subfolders.")

        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        train_data = datasets.ImageFolder(str(train_dir), transform=transform)
        val_data = datasets.ImageFolder(str(val_dir), transform=transform)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)

        model = self.build_model(num_classes=len(train_data.classes), pretrained=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        best_val_acc = 0.0
        history = []
        for epoch in range(epochs):
            model.train()
            train_correct = 0
            train_total = 0
            train_loss_sum = 0.0
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss_sum += float(loss.item()) * labels.size(0)
                predictions = outputs.argmax(dim=1)
                train_correct += int((predictions == labels).sum().item())
                train_total += int(labels.size(0))

            val_metrics = self._evaluate_loader(model, val_loader, device)
            train_acc = train_correct / max(1, train_total)
            epoch_result = {
                "epoch": epoch + 1,
                "train_loss": round(train_loss_sum / max(1, train_total), 4),
                "train_accuracy": round(train_acc, 4),
                "val_accuracy": round(val_metrics["accuracy"], 4),
            }
            history.append(epoch_result)
            if val_metrics["accuracy"] >= best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                Path(output_model).parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "classes": train_data.classes,
                    },
                    output_model,
                )

        return {"best_val_accuracy": round(best_val_acc, 4), "history": history, "saved_model": output_model}

    @staticmethod
    def _evaluate_loader(model: Any, loader: Any, device: str) -> Dict[str, float]:
        torch = __import__("torch")
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                correct += int((preds == labels).sum().item())
                total += int(labels.size(0))
        return {"accuracy": correct / max(1, total)}

    def predict_image(self, model_path: str, image_path: str) -> Dict[str, float]:
        torch, Image, _, misc = self._require_dependencies()
        _, transforms = misc
        checkpoint = torch.load(model_path, map_location="cpu")
        classes = checkpoint.get("classes", self.class_names)
        model = self.build_model(num_classes=len(classes), pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        image = Image.open(image_path).convert("RGB")
        tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0].tolist()
        return {label: round(float(prob), 4) for label, prob in zip(classes, probs)}

    def generate_gradcam(
        self,
        model_path: str,
        image_path: str,
        output_path: str = "artifacts/gradcam_overlay.png",
    ) -> Dict[str, Any]:
        torch, Image, _, misc = self._require_dependencies()
        _, transforms = misc

        checkpoint = torch.load(model_path, map_location="cpu")
        classes = checkpoint.get("classes", self.class_names)
        model = self.build_model(num_classes=len(classes), pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        image = Image.open(image_path).convert("RGB")
        tensor = transform(image).unsqueeze(0)

        activations: List[Any] = []
        gradients: List[Any] = []

        def forward_hook(_module: Any, _inputs: Any, output: Any) -> None:
            activations.append(output.detach())

        def backward_hook(_module: Any, grad_in: Any, grad_out: Any) -> None:
            del grad_in
            gradients.append(grad_out[0].detach())

        target_layer = model.features[-1]
        handle_forward = target_layer.register_forward_hook(forward_hook)
        handle_backward = target_layer.register_full_backward_hook(backward_hook)

        logits = model(tensor)
        class_index = int(logits.argmax(dim=1).item())
        score = logits[0, class_index]
        model.zero_grad()
        score.backward()

        handle_forward.remove()
        handle_backward.remove()

        feature_map = activations[0][0]
        grads = gradients[0][0]
        pooled_grads = grads.mean(dim=(1, 2))
        for channel_index in range(feature_map.shape[0]):
            feature_map[channel_index] *= pooled_grads[channel_index]

        heatmap = feature_map.mean(dim=0).clamp(min=0)
        heatmap /= heatmap.max() + 1e-8
        heatmap_np = heatmap.numpy()

        base = image.resize((224, 224))
        try:
            import numpy as np
            from PIL import ImageEnhance
        except ImportError as exc:
            raise RuntimeError("Install numpy and pillow for Grad-CAM export.") from exc

        red_map = (heatmap_np * 255).astype("uint8")
        overlay = Image.new("RGBA", (224, 224), (255, 0, 0, 0))
        alpha = Image.fromarray(red_map).convert("L")
        overlay.putalpha(alpha)
        combined = Image.blend(base.convert("RGBA"), overlay, alpha=0.35)
        combined = ImageEnhance.Contrast(combined).enhance(1.1)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        combined.save(output_path)
        return {
            "predicted_label": classes[class_index],
            "confidence": round(float(torch.softmax(logits, dim=1)[0, class_index].item()), 4),
            "gradcam_output": output_path,
        }


def build_adaptive_plan(
    context: UserContext,
    analysis: AnalysisResult,
    policy: AdaptivePolicyEngine,
    question_generator: HuggingFaceQuestionGenerator,
    music_recommender: SpotifyMusicRecommender,
    monitor: EthicalAIMonitor,
) -> AdaptivePlan:
    action = policy.choose_action(context, analysis.primary_emotion)
    questions = question_generator.generate_questions(context, analysis)
    lifestyle_steps = list(LIFESTYLE_LIBRARY.get(analysis.primary_emotion, LIFESTYLE_LIBRARY["neutral"]))
    lifestyle_steps.insert(0, policy.describe_action(action))
    music = music_recommender.recommend(analysis.primary_emotion, limit=5)
    mood_summary = (
        f"The system reads the current state as {analysis.primary_emotion} "
        f"with confidence {analysis.confidence:.2f} based on text evidence."
    )
    ethical_notice = monitor.assess(context, analysis)
    return AdaptivePlan(
        chosen_action=action,
        questions=questions,
        lifestyle_steps=lifestyle_steps,
        music_suggestions=music,
        mood_summary=mood_summary,
        ethical_notice=ethical_notice,
    )


def print_header() -> None:
    print("=" * 78)
    print(APP_NAME)
    print(f"Version: {APP_VERSION}")
    print("=" * 78)


def collect_user_context() -> UserContext:
    print("\nEnter the current user context. Press Enter to accept defaults where shown.")
    name = prompt_input("Name [Student]: ").strip() or "Student"
    role = prompt_input("Role [Final-year major project student]: ").strip() or "Final-year major project student"
    study_pressure = safe_int(prompt_input("Study pressure 1-10 [7]: "), 7, 1, 10)
    sleep_hours = safe_float(prompt_input("Sleep hours last night [6.0]: "), 6.0, 0.0, 16.0)
    energy_level = safe_int(prompt_input("Energy level 1-10 [5]: "), 5, 1, 10)
    screen_time_hours = safe_float(prompt_input("Daily screen time hours [7.5]: "), 7.5, 0.0, 16.0)
    current_goal = (
        prompt_input("Current goal [Prepare project report and viva]: ").strip()
        or "Prepare project report and viva"
    )
    print("\nWrite the current mood, thoughts, and situation in a few sentences.")
    journal_text = prompt_input("Journal text: ").strip()
    if not journal_text:
        journal_text = (
            "I feel some pressure because of my project and viva, but I also want to stay focused "
            "and complete my work in a disciplined way."
        )
    return UserContext(
        name=name,
        role=role,
        study_pressure=study_pressure,
        sleep_hours=sleep_hours,
        energy_level=energy_level,
        screen_time_hours=screen_time_hours,
        current_goal=current_goal,
        journal_text=journal_text,
        consent=True,
    )


def display_analysis(result: AnalysisResult) -> None:
    print("\nEmotion Analysis")
    print("-" * 78)
    print(f"Primary emotion : {pretty_label(result.primary_emotion)}")
    print(f"Confidence      : {result.confidence:.2f}")
    print(f"Analysis source : {result.source}")
    print("Supporting cues :")
    for note in result.reasoning:
        print(f"  - {note}")

    print("\nEmotion score distribution")
    for emotion, score in sorted(result.emotion_scores.items(), key=lambda item: item[1], reverse=True):
        print(f"  {pretty_label(emotion):<14} {score:.3f}")


def display_plan(plan: AdaptivePlan, twin: DigitalEmotionalTwin) -> None:
    print("\nAdaptive Plan")
    print("-" * 78)
    print(plan.mood_summary)
    print(f"Primary adaptive action: {pretty_label(plan.chosen_action)}")

    print("\nImportant reflective questions")
    for idx, question in enumerate(plan.questions, start=1):
        print(f"  {idx}. {question}")

    print("\nLifestyle steps")
    for idx, step in enumerate(plan.lifestyle_steps, start=1):
        print(f"  {idx}. {step}")

    print("\nMusic suggestions")
    for idx, item in enumerate(plan.music_suggestions, start=1):
        line = f"  {idx}. {item.get('title')} - {item.get('artist')}"
        if item.get("url"):
            line += f" | {item['url']}"
        elif item.get("query"):
            line += f" | search: {item['query']}"
        print(line)

    print("\nDigital emotional twin")
    print(f"  {twin.summary()}")

    print("\nEthical AI notice")
    print(f"  {plan.ethical_notice}")


def print_free_api_setup() -> None:
    print("\nFree API setup notes")
    print("-" * 78)
    print("1. Hugging Face free token")
    print("   - Create a free account at https://huggingface.co/join")
    print("   - Open Settings -> Access Tokens -> New token -> Read")
    print("   - On Windows CMD:   setx HF_TOKEN your_token_here")
    print(f"   - Optional model:   setx HF_LLM_MODEL {DEFAULT_HF_MODEL}")
    print("   - The program will use the free Inference API when the token is available.")
    print("")
    print("2. Spotify developer credentials")
    print("   - Create a free developer app at https://developer.spotify.com/dashboard")
    print("   - Generate a client ID and client secret")
    print("   - On Windows CMD:   setx SPOTIFY_CLIENT_ID your_id")
    print("   - On Windows CMD:   setx SPOTIFY_CLIENT_SECRET your_secret")
    print("   - If Spotify credentials are not configured, the program uses an offline playlist library.")
    print("")
    print("3. Why Hugging Face is used here")
    print("   - It offers open-source model access and a free tier suitable for academic prototypes.")
    print("   - It avoids dependence on paid proprietary APIs for the core question-generation flow.")
    print("   - The system still works offline through fallback logic when API access is unavailable.")


def print_apk_guide() -> None:
    print("\nFree APK conversion trick")
    print("-" * 78)
    print("Option A: Kivy + Buildozer through WSL or Ubuntu VM")
    print("  1. Keep the project logic in this single Python file.")
    print("  2. Add a lightweight Kivy wrapper screen later if Android UI is needed.")
    print("  3. On Ubuntu or WSL, install buildozer and Android dependencies.")
    print("  4. Run: buildozer init")
    print("  5. Edit buildozer.spec to include only essential packages.")
    print("  6. Run: buildozer android debug")
    print("")
    print("Option B: Google Colab build path")
    print("  1. Upload the project to Colab or GitHub.")
    print("  2. Use an Ubuntu notebook runtime.")
    print("  3. Install buildozer and build the APK remotely for free.")
    print("")
    print("Note: CLI logic is already single-file. For a polished APK, add a minimal mobile UI wrapper.")


def print_project_significance() -> None:
    print("\nProject significance notes")
    print("-" * 78)
    print("Explainable AI and Grad-CAM")
    print("  - Grad-CAM shows which facial regions influenced the CNN prediction.")
    print("  - This improves trust, debugging, and viva explainability.")
    print("")
    print("EfficientNetV2-S instead of MobileNetV2")
    print("  - EfficientNetV2 is a newer CNN family with stronger accuracy-efficiency trade-offs.")
    print("  - It works well with Global Average Pooling and Grad-CAM on resource-limited systems.")
    print("")
    print("Global Average Pooling")
    print("  - GAP reduces parameters compared to large dense heads.")
    print("  - It improves model compactness and helps class activation style explanations.")
    print("")
    print("Ethical AI monitoring")
    print("  - The system checks consent, low confidence, and stress-sensitive cases.")
    print("  - It frames outputs as advisory support, not clinical diagnosis.")
    print("")
    print("Digital emotional twin")
    print("  - The system stores recent emotional states and adaptive actions.")
    print("  - This makes recommendations more personalized over time.")
    print("")
    print("Multi-AI fusion")
    print("  - Text analysis plus optional visual emotion analysis improves contextual awareness.")
    print("  - The fusion layer reduces dependence on a single noisy signal.")
    print("")
    print("Reinforcement learning logic")
    print("  - Feedback updates a small Q-table so future actions fit the user's response pattern.")
    print("  - This bridges static recommendations and adaptive lifestyle assistance.")


def run_text_workflow() -> None:
    config = {
        "hf_token": os.getenv("HF_TOKEN", ""),
        "hf_model": os.getenv("HF_LLM_MODEL", DEFAULT_HF_MODEL),
        "spotify_client_id": os.getenv("SPOTIFY_CLIENT_ID", ""),
        "spotify_client_secret": os.getenv("SPOTIFY_CLIENT_SECRET", ""),
    }
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    context = collect_user_context()
    analyzer = TextEmotionAnalyzer()
    question_generator = HuggingFaceQuestionGenerator(config["hf_token"], config["hf_model"])
    recommender = SpotifyMusicRecommender(config["spotify_client_id"], config["spotify_client_secret"])
    policy = AdaptivePolicyEngine(artifacts_dir / "adaptive_policy.json")
    twin = DigitalEmotionalTwin(artifacts_dir / "digital_emotional_twin.json")
    monitor = EthicalAIMonitor()

    text_result = analyzer.analyze(context.journal_text)
    fusion = MultiAIFusionEngine()
    final_result = fusion.fuse(text_result)
    plan = build_adaptive_plan(context, final_result, policy, question_generator, recommender, monitor)
    twin.record(context, final_result, plan.chosen_action)

    display_analysis(final_result)
    display_plan(plan, twin)

    reward_text = prompt_input("\nRate how useful the main action feels from -2 to +2 [1]: ", "1").strip() or "1"
    reward = float(safe_float(reward_text, 1.0, -2.0, 2.0))
    policy.update_feedback(context, final_result.primary_emotion, plan.chosen_action, reward)
    print("Feedback saved to the adaptive policy file.")


def run_training_menu() -> None:
    module = VisionEmotionModule()
    data_root = prompt_input("Dataset root [data/fer]: ").strip() or "data/fer"
    epochs = safe_int(prompt_input("Epochs [3]: "), 3, 1, 50)
    batch_size = safe_int(prompt_input("Batch size [16]: "), 16, 1, 128)
    output_model = (
        prompt_input("Output model [artifacts/efficientnetv2_emotion.pt]: ").strip()
        or "artifacts/efficientnetv2_emotion.pt"
    )
    try:
        metrics = module.train(data_root=data_root, output_model=output_model, epochs=epochs, batch_size=batch_size)
        print(json.dumps(metrics, indent=2))
    except Exception as exc:
        print(f"Training could not run: {exc}")


def run_gradcam_menu() -> None:
    module = VisionEmotionModule()
    model_path = (
        prompt_input("Model path [artifacts/efficientnetv2_emotion.pt]: ").strip()
        or "artifacts/efficientnetv2_emotion.pt"
    )
    image_path = prompt_input("Image path: ").strip()
    if not image_path:
        print("Image path is required.")
        return
    output_path = (
        prompt_input("Grad-CAM output [artifacts/gradcam_overlay.png]: ").strip()
        or "artifacts/gradcam_overlay.png"
    )
    try:
        result = module.generate_gradcam(model_path=model_path, image_path=image_path, output_path=output_path)
        print(json.dumps(result, indent=2))
    except Exception as exc:
        print(f"Grad-CAM generation could not run: {exc}")


def run_prediction_menu() -> None:
    module = VisionEmotionModule()
    model_path = (
        prompt_input("Model path [artifacts/efficientnetv2_emotion.pt]: ").strip()
        or "artifacts/efficientnetv2_emotion.pt"
    )
    image_path = prompt_input("Image path: ").strip()
    if not image_path:
        print("Image path is required.")
        return
    try:
        result = module.predict_image(model_path=model_path, image_path=image_path)
        print(json.dumps(result, indent=2))
    except Exception as exc:
        print(f"Prediction could not run: {exc}")


def print_menu() -> None:
    print("\nMenu")
    print("1. Text-based mood analysis, adaptive questions, and lifestyle plan")
    print("2. Train EfficientNetV2-S emotion model")
    print("3. Predict emotion from image")
    print("4. Generate Grad-CAM explanation")
    print("5. Show free API setup guide")
    print("6. Show APK conversion guide")
    print("7. Show project significance notes")
    print("8. Exit")


def main() -> None:
    print_header()
    while True:
        print_menu()
        choice = prompt_input("Select an option [1-8]: ", "8").strip()
        if choice == "1":
            run_text_workflow()
        elif choice == "2":
            run_training_menu()
        elif choice == "3":
            run_prediction_menu()
        elif choice == "4":
            run_gradcam_menu()
        elif choice == "5":
            print_free_api_setup()
        elif choice == "6":
            print_apk_guide()
        elif choice == "7":
            print_project_significance()
        elif choice == "8":
            print("Exiting.")
            break
        else:
            print("Invalid option. Choose a number from 1 to 8.")


if __name__ == "__main__":
    main()
