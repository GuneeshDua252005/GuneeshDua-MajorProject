"""
Cognitive Emotion Intelligence & Adaptive Lifestyle System
-----------------------------------------------------------
Lightweight reference implementation of a multi-modal, emotion-aware
adaptive system aligned with the Chapter-4 research methodology.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    plt = None

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    np = None

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pd = None


VALID_EMOTIONS = {
    "happy",
    "calm",
    "neutral",
    "sad",
    "angry",
    "stressed",
    "tired",
    "excited",
}


@dataclass
class SessionInput:
    user_id: str
    emoji_text: Optional[str] = None
    voice_text: Optional[str] = None
    face_image_path: Optional[str] = None
    gesture_image_path: Optional[str] = None
    timestamp: Optional[datetime] = None


@dataclass
class InteractionRecord:
    user_id: str
    timestamp: str
    face_emotion: str
    emoji_emotion: str
    voice_emotion: str
    time_emotion: str
    fused_emotion: str
    fusion_confidence: float
    gesture: str
    recommendation: str
    feedback_reward: float = 0.0


class FaceMoodDetector:
    """
    Lightweight facial mood estimator.
    Uses basic vision statistics instead of deep neural inference.
    """

    def detect(self, image_path: Optional[str]) -> Tuple[str, float]:
        if not image_path:
            return "neutral", 0.35
        if cv2 is None:
            return "neutral", 0.25

        image = cv2.imread(image_path)
        if image is None:
            return "neutral", 0.2

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        edges = cv2.Canny(gray, 75, 150)
        edge_density = float((edges > 0).sum()) / max(1, gray.size)

        # Simple interpretable heuristics for lightweight inference.
        if brightness > 145 and edge_density > 0.11:
            return "excited", 0.72
        if brightness > 150:
            return "happy", 0.67
        if brightness < 70:
            return "sad", 0.64
        if edge_density > 0.16:
            return "stressed", 0.61
        return "neutral", 0.56


class EmojiSentimentAnalyzer:
    EMOJI_TO_EMOTION = {
        "😀": "happy",
        "😄": "happy",
        "😁": "happy",
        "😊": "calm",
        "🙂": "neutral",
        "😌": "calm",
        "😢": "sad",
        "😭": "sad",
        "😞": "sad",
        "😡": "angry",
        "😤": "angry",
        "😰": "stressed",
        "😓": "stressed",
        "😴": "tired",
        "🤩": "excited",
        "🔥": "excited",
    }

    WORD_MAP = {
        "happy": "happy",
        "joy": "happy",
        "great": "happy",
        "good": "calm",
        "relaxed": "calm",
        "ok": "neutral",
        "fine": "neutral",
        "sad": "sad",
        "down": "sad",
        "angry": "angry",
        "mad": "angry",
        "stress": "stressed",
        "anxious": "stressed",
        "tired": "tired",
        "sleepy": "tired",
        "excited": "excited",
        "pumped": "excited",
    }

    def detect(self, emoji_text: Optional[str]) -> Tuple[str, float]:
        if not emoji_text:
            return "neutral", 0.25
        text = emoji_text.lower().strip()

        for symbol, emotion in self.EMOJI_TO_EMOTION.items():
            if symbol in text:
                return emotion, 0.86

        for token in text.split():
            if token in self.WORD_MAP:
                return self.WORD_MAP[token], 0.68
        return "neutral", 0.4


class VoiceSentimentAnalyzer:
    POSITIVE = {"happy", "good", "great", "productive", "awesome", "calm"}
    NEGATIVE = {"sad", "angry", "stressed", "anxious", "tired", "burnout"}

    def detect(self, text: Optional[str]) -> Tuple[str, float]:
        if not text:
            return "neutral", 0.22
        cleaned = text.lower()
        tokens = cleaned.split()
        p = sum(1 for t in tokens if t in self.POSITIVE)
        n = sum(1 for t in tokens if t in self.NEGATIVE)
        exclamation = cleaned.count("!")

        if n > p + 1:
            if "angry" in tokens:
                return "angry", 0.74
            if "tired" in tokens:
                return "tired", 0.72
            return "stressed", 0.69
        if p > n + 1:
            if exclamation > 0:
                return "excited", 0.71
            return "happy", 0.66
        if "calm" in tokens:
            return "calm", 0.63
        return "neutral", 0.45


class GestureRecognitionEngine:
    """
    Lightweight non-contact gesture interpretation using threshold/contour logic.
    """

    def detect(self, image_path: Optional[str]) -> Tuple[str, float]:
        if not image_path or cv2 is None or np is None:
            return "neutral", 0.2

        image = cv2.imread(image_path)
        if image is None:
            return "neutral", 0.2

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)
        white_ratio = float((thresh > 0).sum()) / max(1, thresh.size)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_count = len(contours)

        if white_ratio > 0.42 and contour_count <= 2:
            return "open_palm", 0.7
        if white_ratio < 0.18:
            return "fist", 0.68
        if contour_count > 6:
            return "wave", 0.62
        return "neutral", 0.5


class TimeContextAnalyzer:
    def detect(self, timestamp: datetime) -> Tuple[str, float]:
        hour = timestamp.hour
        if 5 <= hour < 11:
            return "excited", 0.51
        if 11 <= hour < 17:
            return "neutral", 0.48
        if 17 <= hour < 22:
            return "calm", 0.5
        return "tired", 0.58


class EmotionFusionEngine:
    def __init__(self) -> None:
        self.weights = {
            "face": 0.4,
            "emoji": 0.3,
            "voice": 0.2,
            "time": 0.1,
        }
        self.emotion_to_score = {
            "happy": 1.0,
            "excited": 0.8,
            "calm": 0.5,
            "neutral": 0.0,
            "tired": -0.4,
            "sad": -0.8,
            "stressed": -0.9,
            "angry": -1.0,
        }

    def fuse(self, signals: Dict[str, Tuple[str, float]]) -> Dict[str, object]:
        weighted_votes: Dict[str, float] = defaultdict(float)
        signed_score = 0.0
        total_weight = 0.0

        for modality, (emotion, confidence) in signals.items():
            weight = self.weights.get(modality, 0.0)
            contribution = weight * max(0.0, min(1.0, confidence))
            weighted_votes[emotion] += contribution
            signed_score += contribution * self.emotion_to_score.get(emotion, 0.0)
            total_weight += contribution

        if not weighted_votes:
            final_emotion = "neutral"
            confidence = 0.0
        else:
            final_emotion = max(weighted_votes, key=weighted_votes.get)
            confidence = weighted_votes[final_emotion] / max(total_weight, 1e-9)

        return {
            "final_emotion": final_emotion,
            "confidence": round(confidence, 4),
            "energy_index": round(signed_score, 4),
            "weighted_votes": dict(weighted_votes),
        }


class RecommendationEngine:
    def __init__(self) -> None:
        self.catalog = {
            "happy": ["Celebrate wins playlist", "Deep work sprint", "Creative challenge"],
            "excited": ["Workout burst", "Priority planning", "Focus breathing"],
            "calm": ["Reflective journaling", "Skill-building session", "Nature walk"],
            "neutral": ["Pomodoro focus block", "Hydration reminder", "Light stretching"],
            "sad": ["Supportive music", "Reach out to friend", "5-minute gratitude"],
            "angry": ["Box breathing", "Take a brief walk", "Pause before replying"],
            "stressed": ["Stress reset playlist", "Task decomposition", "Desk meditation"],
            "tired": ["Power nap", "Low-intensity task", "Gentle movement"],
        }
        self.q_values: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    def recommend(self, emotion: str, top_k: int = 3) -> List[str]:
        options = self.catalog.get(emotion, self.catalog["neutral"])
        scored = []
        for idx, item in enumerate(options):
            # Small prior bias preserves stable outputs before feedback arrives.
            prior = 0.2 * (len(options) - idx)
            score = self.q_values[emotion][item] + prior
            scored.append((score, item))
        scored.sort(reverse=True)
        return [item for _, item in scored[:top_k]]

    def apply_feedback(self, emotion: str, recommendation: str, reward: float, alpha: float = 0.35) -> None:
        current = self.q_values[emotion][recommendation]
        self.q_values[emotion][recommendation] = current + alpha * (reward - current)

    def recommendation_effectiveness(self) -> Dict[str, Dict[str, float]]:
        return {emotion: dict(items) for emotion, items in self.q_values.items()}


class EmotionalDigitalTwin:
    def __init__(self) -> None:
        self.history: List[InteractionRecord] = []

    def update(self, record: InteractionRecord) -> None:
        self.history.append(record)

    def profile(self) -> Dict[str, object]:
        if not self.history:
            return {
                "dominant_emotion": "neutral",
                "emotional_stability": 0.0,
                "engagement_level": 0.0,
                "stress_ratio": 0.0,
            }

        emotions = [r.fused_emotion for r in self.history]
        dominant = Counter(emotions).most_common(1)[0][0]
        stress_events = sum(1 for e in emotions if e in {"stressed", "angry", "sad"})
        stress_ratio = stress_events / len(emotions)

        # Simple stability metric using transitions between adjacent emotions.
        transitions = sum(1 for a, b in zip(emotions, emotions[1:]) if a != b)
        stability = 1 - (transitions / max(1, len(emotions) - 1))

        first = datetime.fromisoformat(self.history[0].timestamp)
        last = datetime.fromisoformat(self.history[-1].timestamp)
        active_days = max(1, (last.date() - first.date()).days + 1)
        engagement = len(self.history) / active_days

        return {
            "dominant_emotion": dominant,
            "emotional_stability": round(stability, 4),
            "engagement_level": round(engagement, 4),
            "stress_ratio": round(stress_ratio, 4),
        }


class DataPreprocessor:
    @staticmethod
    def normalize_emotion(label: str) -> str:
        cleaned = (label or "").strip().lower()
        if cleaned in VALID_EMOTIONS:
            return cleaned
        synonyms = {
            "joyful": "happy",
            "upset": "angry",
            "fatigue": "tired",
            "anxiety": "stressed",
        }
        return synonyms.get(cleaned, "neutral")

    @staticmethod
    def preprocess(df):  # type: ignore[no-untyped-def]
        if pd is None:
            return df
        if df.empty:
            return df
        required = ["timestamp", "fused_emotion", "recommendation", "feedback_reward"]
        for col in required:
            if col not in df.columns:
                df[col] = None
        df = df.dropna(subset=["timestamp", "fused_emotion"]).copy()
        df["fused_emotion"] = df["fused_emotion"].apply(DataPreprocessor.normalize_emotion)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        return df.sort_values("timestamp")


class ExcelStorage:
    def __init__(self, file_path: str = "emotion_logs.xlsx") -> None:
        self.file_path = Path(file_path)

    def append_interaction(self, record: InteractionRecord) -> None:
        if pd is None:
            return
        row = pd.DataFrame([asdict(record)])
        if self.file_path.exists():
            try:
                existing = pd.read_excel(self.file_path, sheet_name="interactions")
            except Exception:
                existing = pd.DataFrame()
            updated = pd.concat([existing, row], ignore_index=True)
        else:
            updated = row
        with pd.ExcelWriter(self.file_path, engine="openpyxl") as writer:
            updated.to_excel(writer, sheet_name="interactions", index=False)

    def load_interactions(self):
        if pd is None or not self.file_path.exists():
            return pd.DataFrame() if pd is not None else []
        try:
            return pd.read_excel(self.file_path, sheet_name="interactions")
        except Exception:
            return pd.DataFrame()


class EmotionalAnalytics:
    EMOTION_ENERGY = {
        "happy": 85,
        "excited": 90,
        "calm": 70,
        "neutral": 55,
        "tired": 40,
        "sad": 30,
        "stressed": 25,
        "angry": 20,
    }

    def build_summary(self, interactions_df):
        if pd is None:
            return {"error": "pandas_not_installed"}
        clean = DataPreprocessor.preprocess(interactions_df)
        if clean.empty:
            return {
                "dominant_emotion": "neutral",
                "mood_distribution": {},
                "average_energy_score": 0.0,
                "engagement_by_day": {},
                "anomalies": [],
            }

        clean["energy_score"] = clean["fused_emotion"].map(self.EMOTION_ENERGY).fillna(55)
        mood_distribution = clean["fused_emotion"].value_counts(normalize=True).round(4).to_dict()
        dominant_emotion = clean["fused_emotion"].mode().iat[0]
        avg_energy = float(clean["energy_score"].mean())

        daily = clean.set_index("timestamp").resample("D").size()
        engagement = {k.strftime("%Y-%m-%d"): int(v) for k, v in daily.items()}

        anomalies = self.detect_anomalies(clean)
        return {
            "dominant_emotion": dominant_emotion,
            "mood_distribution": mood_distribution,
            "average_energy_score": round(avg_energy, 2),
            "engagement_by_day": engagement,
            "anomalies": anomalies,
        }

    def detect_anomalies(self, clean_df):
        if pd is None or clean_df.empty:
            return []
        anomalies = []
        negative = clean_df["fused_emotion"].isin(["sad", "stressed", "angry"])
        rolling_negative = negative.rolling(window=5, min_periods=3).mean().fillna(0)
        for idx, ratio in rolling_negative.items():
            if ratio >= 0.8:
                ts = clean_df.iloc[idx]["timestamp"]
                anomalies.append(
                    {
                        "type": "prolonged_negative_emotions",
                        "timestamp": ts.isoformat(),
                        "score": round(float(ratio), 2),
                    }
                )
        return anomalies

    def generate_weekly_report_figure(self, interactions_df, output_path: str = "weekly_report.png") -> Optional[str]:
        if pd is None or plt is None:
            return None
        clean = DataPreprocessor.preprocess(interactions_df)
        if clean.empty:
            return None
        week_ago = clean["timestamp"].max() - timedelta(days=7)
        weekly = clean[clean["timestamp"] >= week_ago]
        if weekly.empty:
            return None

        counts = weekly["fused_emotion"].value_counts()
        fig, ax = plt.subplots(figsize=(8, 4))
        counts.plot(kind="bar", ax=ax, color="#5A8DEE")
        ax.set_title("Weekly Mood Distribution")
        ax.set_ylabel("Interaction Count")
        ax.set_xlabel("Emotion")
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        return output_path


class AILifestyleCoach:
    def recommend(self, fused_emotion: str, twin_profile: Dict[str, object], anomalies: List[Dict[str, object]]) -> str:
        stability = float(twin_profile.get("emotional_stability", 0.0))
        stress_ratio = float(twin_profile.get("stress_ratio", 0.0))

        if anomalies:
            return (
                "Your recent mood pattern indicates elevated emotional strain. "
                "Try a short recovery block now: hydrate, slow breathing for 3 minutes, "
                "then complete one low-pressure task."
            )
        if fused_emotion in {"stressed", "angry"} or stress_ratio > 0.5:
            return "Stress is trending high. Prioritize task decomposition and a 10-minute reset walk."
        if fused_emotion in {"sad", "tired"} and stability < 0.4:
            return "Energy appears low and unstable. Use a gentle routine: sunlight, music, and one achievable micro-goal."
        if fused_emotion in {"happy", "excited"}:
            return "Momentum is positive. Allocate this state to meaningful progress on your highest-value task."
        return "Maintain balance with a focused 25-minute session followed by a mindful short break."


class CognitiveEmotionSystem:
    def __init__(self, storage_path: str = "emotion_logs.xlsx") -> None:
        self.face = FaceMoodDetector()
        self.emoji = EmojiSentimentAnalyzer()
        self.voice = VoiceSentimentAnalyzer()
        self.gesture = GestureRecognitionEngine()
        self.time_ctx = TimeContextAnalyzer()
        self.fusion = EmotionFusionEngine()
        self.recommendation = RecommendationEngine()
        self.twin = EmotionalDigitalTwin()
        self.storage = ExcelStorage(storage_path)
        self.analytics = EmotionalAnalytics()
        self.coach = AILifestyleCoach()

    def process(self, session: SessionInput) -> Dict[str, object]:
        timestamp = session.timestamp or datetime.now(timezone.utc)
        face_emotion, face_conf = self.face.detect(session.face_image_path)
        emoji_emotion, emoji_conf = self.emoji.detect(session.emoji_text)
        voice_emotion, voice_conf = self.voice.detect(session.voice_text)
        gesture_name, _ = self.gesture.detect(session.gesture_image_path)
        time_emotion, time_conf = self.time_ctx.detect(timestamp)

        fusion_result = self.fusion.fuse(
            {
                "face": (face_emotion, face_conf),
                "emoji": (emoji_emotion, emoji_conf),
                "voice": (voice_emotion, voice_conf),
                "time": (time_emotion, time_conf),
            }
        )
        fused_emotion = str(fusion_result["final_emotion"])
        top_recommendations = self.recommendation.recommend(fused_emotion, top_k=3)
        selected_recommendation = top_recommendations[0]

        record = InteractionRecord(
            user_id=session.user_id,
            timestamp=timestamp.isoformat(),
            face_emotion=face_emotion,
            emoji_emotion=emoji_emotion,
            voice_emotion=voice_emotion,
            time_emotion=time_emotion,
            fused_emotion=fused_emotion,
            fusion_confidence=float(fusion_result["confidence"]),
            gesture=gesture_name,
            recommendation=selected_recommendation,
            feedback_reward=0.0,
        )
        self.twin.update(record)
        self.storage.append_interaction(record)

        interactions_df = self.storage.load_interactions()
        summary = self.analytics.build_summary(interactions_df) if pd is not None else {}
        coaching_message = self.coach.recommend(fused_emotion, self.twin.profile(), summary.get("anomalies", []))

        return {
            "detected_signals": {
                "face": {"emotion": face_emotion, "confidence": round(face_conf, 3)},
                "emoji": {"emotion": emoji_emotion, "confidence": round(emoji_conf, 3)},
                "voice": {"emotion": voice_emotion, "confidence": round(voice_conf, 3)},
                "time_context": {"emotion": time_emotion, "confidence": round(time_conf, 3)},
                "gesture": gesture_name,
            },
            "fusion": fusion_result,
            "recommendations": top_recommendations,
            "coach_message": coaching_message,
            "digital_twin_profile": self.twin.profile(),
            "analytics_summary": summary,
        }

    def apply_feedback(self, emotion: str, recommendation: str, reward: float) -> None:
        normalized = DataPreprocessor.normalize_emotion(emotion)
        clipped_reward = max(-1.0, min(1.0, reward))
        self.recommendation.apply_feedback(normalized, recommendation, clipped_reward)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Cognitive Emotion Intelligence system.")
    parser.add_argument("--user-id", default="user_1", help="Unique user identifier")
    parser.add_argument("--emoji-text", default=None, help="Emoji/text sentiment input")
    parser.add_argument("--voice-text", default=None, help="Voice transcript or text input")
    parser.add_argument("--face-image-path", default=None, help="Path to face image")
    parser.add_argument("--gesture-image-path", default=None, help="Path to gesture image")
    parser.add_argument("--storage-path", default="emotion_logs.xlsx", help="Excel storage file path")
    parser.add_argument("--feedback-emotion", default=None, help="Emotion for feedback update")
    parser.add_argument("--feedback-item", default=None, help="Recommendation text receiving feedback")
    parser.add_argument("--feedback-reward", type=float, default=None, help="Reward in [-1, 1]")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    system = CognitiveEmotionSystem(storage_path=args.storage_path)

    payload = SessionInput(
        user_id=args.user_id,
        emoji_text=args.emoji_text,
        voice_text=args.voice_text,
        face_image_path=args.face_image_path,
        gesture_image_path=args.gesture_image_path,
        timestamp=datetime.now(timezone.utc),
    )
    result = system.process(payload)

    if (
        args.feedback_emotion
        and args.feedback_item
        and args.feedback_reward is not None
    ):
        system.apply_feedback(args.feedback_emotion, args.feedback_item, args.feedback_reward)
        result["feedback_update"] = {
            "emotion": args.feedback_emotion,
            "item": args.feedback_item,
            "reward": max(-1.0, min(1.0, args.feedback_reward)),
            "status": "applied",
        }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
