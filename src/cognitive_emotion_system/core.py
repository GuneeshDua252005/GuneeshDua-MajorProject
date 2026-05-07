"""Core implementation for a lightweight emotion-aware lifestyle system.

The module intentionally keeps the default runtime dependency-free so the
research prototype can run on standard Python installations. Optional adapters
can be added at the boundaries for OpenCV, speech recognition, Streamlit, or
Excel storage without changing the domain workflow.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from math import sqrt
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence


CANONICAL_EMOTIONS = (
    "joyful",
    "calm",
    "neutral",
    "sad",
    "angry",
    "anxious",
    "stressed",
    "tired",
)

EMOTION_ALIASES = {
    "happy": "joyful",
    "joy": "joyful",
    "excited": "joyful",
    "smile": "joyful",
    "relaxed": "calm",
    "peaceful": "calm",
    "ok": "neutral",
    "normal": "neutral",
    "fine": "neutral",
    "unhappy": "sad",
    "depressed": "sad",
    "mad": "angry",
    "furious": "angry",
    "worry": "anxious",
    "worried": "anxious",
    "fear": "anxious",
    "pressure": "stressed",
    "stress": "stressed",
    "burnout": "stressed",
    "sleepy": "tired",
    "fatigue": "tired",
    "exhausted": "tired",
}

EMOTION_ENERGY_SCORES = {
    "joyful": 0.95,
    "calm": 0.80,
    "neutral": 0.60,
    "tired": 0.42,
    "anxious": 0.34,
    "stressed": 0.30,
    "sad": 0.22,
    "angry": 0.18,
}


def normalize_emotion(label: str | None) -> str:
    """Return a supported emotion label for arbitrary user or model output."""

    if not label:
        return "neutral"
    normalized = label.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in CANONICAL_EMOTIONS:
        return normalized
    return EMOTION_ALIASES.get(normalized, "neutral")


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


@dataclass(frozen=True)
class EmotionEvidence:
    """A single normalized emotional signal from one modality."""

    modality: str
    emotion: str
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "emotion", normalize_emotion(self.emotion))
        object.__setattr__(self, "confidence", clamp(float(self.confidence)))


@dataclass(frozen=True)
class FusionResult:
    """Weighted emotional state inferred from multimodal evidence."""

    final_emotion: str
    confidence: float
    scores: dict[str, float]
    evidence: tuple[EmotionEvidence, ...]


@dataclass(frozen=True)
class Recommendation:
    """Emotion-aware lifestyle or media recommendation."""

    item_id: str
    category: str
    title: str
    reason: str
    emotion: str
    score: float


@dataclass(frozen=True)
class FeedbackEvent:
    """Explicit or implicit user response used as a reward signal."""

    user_id: str
    recommendation_id: str
    reward: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    note: str = ""


@dataclass(frozen=True)
class InteractionRecord:
    """Structured session record for storage and analysis."""

    user_id: str
    timestamp: datetime
    evidence: tuple[EmotionEvidence, ...]
    final_emotion: str
    confidence: float
    recommendation: Recommendation
    feedback_reward: float | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        data["evidence"] = [asdict(item) for item in self.evidence]
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "InteractionRecord":
        evidence = tuple(EmotionEvidence(**item) for item in data.get("evidence", ()))
        recommendation_data = data["recommendation"]
        timestamp = data["timestamp"]
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        return cls(
            user_id=str(data["user_id"]),
            timestamp=timestamp,
            evidence=evidence,
            final_emotion=normalize_emotion(str(data["final_emotion"])),
            confidence=float(data.get("confidence", 0.0)),
            recommendation=Recommendation(**recommendation_data),
            feedback_reward=data.get("feedback_reward"),
        )


class EmojiSentimentAnalyzer:
    """Map emoji and short textual sentiment indicators to emotion evidence."""

    EMOJI_MAP = {
        ":)": "joyful",
        ":-)": "joyful",
        ":D": "joyful",
        "\U0001f600": "joyful",
        "\U0001f604": "joyful",
        "\U0001f60a": "joyful",
        "\U0001f642": "calm",
        "\U0001f60c": "calm",
        "\U0001f610": "neutral",
        "\U0001f636": "neutral",
        ":(": "sad",
        ":-(": "sad",
        "\U0001f622": "sad",
        "\U0001f62d": "sad",
        "\U0001f614": "sad",
        "\U0001f621": "angry",
        "\U0001f92c": "angry",
        "\U0001f620": "angry",
        "\U0001f630": "anxious",
        "\U0001f61f": "anxious",
        "\U0001f628": "anxious",
        "\U0001f62b": "stressed",
        "\U0001f623": "stressed",
        "\U0001f629": "tired",
        "\U0001f971": "tired",
    }

    WORD_MAP = {
        "happy": "joyful",
        "great": "joyful",
        "good": "joyful",
        "calm": "calm",
        "relaxed": "calm",
        "sad": "sad",
        "down": "sad",
        "angry": "angry",
        "mad": "angry",
        "anxious": "anxious",
        "worried": "anxious",
        "stressed": "stressed",
        "pressure": "stressed",
        "tired": "tired",
        "sleepy": "tired",
    }

    def analyze(self, emoji_input: str | None) -> EmotionEvidence | None:
        if not emoji_input:
            return None

        votes: Counter[str] = Counter()
        for token, emotion in self.EMOJI_MAP.items():
            if token in emoji_input:
                votes[emotion] += emoji_input.count(token)

        normalized_words = (
            emoji_input.lower()
            .replace(",", " ")
            .replace(".", " ")
            .replace("!", " ")
            .replace("?", " ")
            .split()
        )
        for word in normalized_words:
            if word in self.WORD_MAP:
                votes[self.WORD_MAP[word]] += 1

        if not votes:
            return EmotionEvidence("emoji", "neutral", 0.35, {"raw": emoji_input})

        emotion, count = votes.most_common(1)[0]
        confidence = min(0.95, 0.55 + (0.12 * count))
        return EmotionEvidence("emoji", emotion, confidence, {"raw": emoji_input, "votes": dict(votes)})


class TextSentimentAnalyzer:
    """Keyword-based sentiment mapping for speech transcripts or typed text."""

    KEYWORDS = {
        "joyful": {
            "happy",
            "joy",
            "excited",
            "grateful",
            "great",
            "excellent",
            "motivated",
            "proud",
        },
        "calm": {"calm", "peaceful", "relaxed", "balanced", "clear", "steady"},
        "sad": {"sad", "lonely", "hopeless", "down", "upset", "cry", "hurt"},
        "angry": {"angry", "furious", "mad", "annoyed", "irritated", "rage"},
        "anxious": {"anxious", "worried", "afraid", "fear", "panic", "nervous"},
        "stressed": {"stress", "stressed", "pressure", "overloaded", "deadline", "burnout"},
        "tired": {"tired", "sleepy", "exhausted", "fatigue", "drained", "rest"},
    }

    def analyze(self, text: str | None, modality: str = "voice") -> EmotionEvidence | None:
        if not text:
            return None

        tokens = [
            token.strip(".,!?;:()[]{}\"'").lower()
            for token in text.split()
            if token.strip(".,!?;:()[]{}\"'")
        ]
        votes: Counter[str] = Counter()
        for emotion, keywords in self.KEYWORDS.items():
            votes[emotion] = sum(1 for token in tokens if token in keywords)

        votes += Counter({normalize_emotion(token): 1 for token in tokens if normalize_emotion(token) != "neutral"})
        if not votes or votes.most_common(1)[0][1] == 0:
            return EmotionEvidence(modality, "neutral", 0.40, {"text": text})

        emotion, count = votes.most_common(1)[0]
        confidence = min(0.90, 0.50 + (0.10 * count))
        return EmotionEvidence(modality, emotion, confidence, {"text": text, "matched_terms": count})


class TimeContextAnalyzer:
    """Infer a contextual emotional prior from time-of-day patterns."""

    def analyze(self, moment: datetime | None = None) -> EmotionEvidence:
        moment = moment or datetime.now(timezone.utc)
        hour = moment.hour

        if 0 <= hour < 5:
            emotion, confidence = "tired", 0.45
        elif 5 <= hour < 9:
            emotion, confidence = "calm", 0.50
        elif 9 <= hour < 17:
            emotion, confidence = "neutral", 0.42
        elif 17 <= hour < 21:
            emotion, confidence = "calm", 0.45
        else:
            emotion, confidence = "tired", 0.42

        weekend = moment.weekday() >= 5
        if weekend and emotion == "neutral":
            emotion, confidence = "calm", 0.44

        return EmotionEvidence(
            "time_context",
            emotion,
            confidence,
            {"hour": hour, "weekday": moment.weekday(), "weekend": weekend},
        )


class FaceMoodDetector:
    """Lightweight facial mood heuristic for prototype and test inputs."""

    def analyze_features(
        self,
        smile_intensity: float | None = None,
        brow_tension: float | None = None,
        eye_openness: float | None = None,
    ) -> EmotionEvidence | None:
        provided = [value for value in (smile_intensity, brow_tension, eye_openness) if value is not None]
        if not provided:
            return None

        smile = clamp(float(smile_intensity or 0.0))
        brow = clamp(float(brow_tension or 0.0))
        eyes = clamp(float(eye_openness if eye_openness is not None else 0.5))

        if smile >= 0.68 and brow < 0.55:
            emotion = "joyful"
            confidence = 0.65 + (0.25 * smile)
        elif brow >= 0.75 and eyes >= 0.45:
            emotion = "stressed"
            confidence = 0.62 + (0.25 * brow)
        elif brow >= 0.62:
            emotion = "angry"
            confidence = 0.58 + (0.22 * brow)
        elif eyes <= 0.28:
            emotion = "tired"
            confidence = 0.60 + (0.25 * (1 - eyes))
        elif smile <= 0.18 and eyes <= 0.45:
            emotion = "sad"
            confidence = 0.58 + (0.18 * (1 - smile))
        else:
            emotion = "neutral"
            confidence = 0.50

        return EmotionEvidence(
            "face",
            emotion,
            clamp(confidence),
            {"smile_intensity": smile, "brow_tension": brow, "eye_openness": eyes},
        )


class GestureRecognizer:
    """Simple gesture classifier from motion and pixel density features."""

    def analyze_features(
        self,
        motion_level: float | None = None,
        open_hand_density: float | None = None,
    ) -> EmotionEvidence | None:
        if motion_level is None and open_hand_density is None:
            return None

        motion = clamp(float(motion_level or 0.0))
        density = clamp(float(open_hand_density if open_hand_density is not None else 0.5))

        if motion > 0.72:
            emotion, confidence = "stressed", 0.58 + (0.22 * motion)
        elif density > 0.70 and motion < 0.45:
            emotion, confidence = "calm", 0.60 + (0.20 * density)
        elif density < 0.28:
            emotion, confidence = "anxious", 0.55 + (0.20 * (1 - density))
        else:
            emotion, confidence = "neutral", 0.42

        return EmotionEvidence(
            "gesture",
            emotion,
            clamp(confidence),
            {"motion_level": motion, "open_hand_density": density},
        )


class EmotionFusionEngine:
    """Weighted fusion engine for multimodal emotional evidence."""

    DEFAULT_WEIGHTS = {
        "face": 0.35,
        "emoji": 0.25,
        "voice": 0.20,
        "gesture": 0.10,
        "time_context": 0.10,
    }

    def __init__(self, modality_weights: Mapping[str, float] | None = None) -> None:
        self.modality_weights = dict(modality_weights or self.DEFAULT_WEIGHTS)

    def fuse(self, evidence: Iterable[EmotionEvidence]) -> FusionResult:
        items = tuple(item for item in evidence if item is not None)
        if not items:
            return FusionResult("neutral", 0.0, {"neutral": 1.0}, ())

        active_weight_sum = sum(self.modality_weights.get(item.modality, 0.05) for item in items)
        if active_weight_sum <= 0:
            active_weight_sum = 1.0

        scores = {emotion: 0.0 for emotion in CANONICAL_EMOTIONS}
        for item in items:
            modality_weight = self.modality_weights.get(item.modality, 0.05) / active_weight_sum
            scores[item.emotion] += modality_weight * item.confidence

        total = sum(scores.values()) or 1.0
        normalized_scores = {emotion: round(score / total, 4) for emotion, score in scores.items()}
        final_emotion, confidence = max(normalized_scores.items(), key=lambda pair: pair[1])
        return FusionResult(final_emotion, confidence, normalized_scores, items)


class RecommendationEngine:
    """Lightweight reinforcement-inspired adaptive recommendation engine."""

    BASE_CATALOG: dict[str, tuple[tuple[str, str, str], ...]] = {
        "joyful": (
            ("music", "Play an upbeat focus playlist", "Use positive energy for creative work."),
            ("activity", "Share one achievement with a friend", "Reinforces healthy social connection."),
            ("coach", "Plan a stretch goal for today", "Momentum is high, so small ambition is useful."),
        ),
        "calm": (
            ("activity", "Do a 10-minute mindful planning session", "Calm states support clear prioritization."),
            ("music", "Play soft instrumental music", "Maintains emotional balance without overstimulation."),
            ("coach", "Protect this calm period for deep work", "Stable emotion can improve productivity."),
        ),
        "neutral": (
            ("activity", "Take a short walk before the next task", "Light movement can improve engagement."),
            ("coach", "Choose one micro-habit to complete", "A small action builds consistency."),
            ("music", "Play a balanced ambient playlist", "Neutral mood can be gently energized."),
        ),
        "sad": (
            ("activity", "Try a grounding exercise and hydrate", "Gentle recovery is better than pressure."),
            ("music", "Play comforting low-tempo music", "Supportive audio can reduce emotional load."),
            ("coach", "Write one sentence about what you need", "Naming needs supports self-regulation."),
        ),
        "angry": (
            ("activity", "Pause for box breathing", "Lowering physiological arousal helps decision quality."),
            ("music", "Play calming acoustic tracks", "Reduced tempo can soften frustration."),
            ("coach", "Delay reactive replies for five minutes", "Creates space before acting."),
        ),
        "anxious": (
            ("activity", "List the next smallest controllable step", "Concrete action reduces uncertainty."),
            ("music", "Play slow breathing-guided audio", "Rhythmic guidance supports nervous-system regulation."),
            ("coach", "Separate facts from worries", "Cognitive sorting can reduce rumination."),
        ),
        "stressed": (
            ("activity", "Take a two-minute breathing break", "Short recovery lowers overload quickly."),
            ("coach", "Split your task into three small steps", "Chunking reduces cognitive pressure."),
            ("music", "Play low-tempo focus music", "Steady sound can reduce perceived pressure."),
        ),
        "tired": (
            ("activity", "Do light stretching and drink water", "Low-energy states need gentle activation."),
            ("coach", "Schedule a recovery break", "Rest prevents deeper fatigue."),
            ("music", "Play gentle energizing music", "Mild stimulation can help without strain."),
        ),
    }

    def __init__(self) -> None:
        self._feedback_weights: defaultdict[tuple[str, str], float] = defaultdict(float)

    def recommend(
        self,
        user_id: str,
        emotion: str,
        analytics_context: Mapping[str, Any] | None = None,
    ) -> Recommendation:
        emotion = normalize_emotion(emotion)
        analytics_context = analytics_context or {}
        candidates = self.BASE_CATALOG[emotion]

        ranked: list[Recommendation] = []
        for index, (category, title, reason) in enumerate(candidates):
            item_id = f"{emotion}:{category}:{index}"
            feedback_boost = self._feedback_weights[(user_id, item_id)]
            stability = float(analytics_context.get("emotional_stability", 0.5))
            stress_penalty = 0.05 if analytics_context.get("stress_alert") and emotion in {"joyful", "neutral"} else 0.0
            score = 1.0 - (index * 0.08) + feedback_boost + (0.04 * stability) - stress_penalty
            ranked.append(Recommendation(item_id, category, title, reason, emotion, round(score, 4)))

        return max(ranked, key=lambda item: item.score)

    def submit_feedback(self, event: FeedbackEvent) -> None:
        reward = clamp(event.reward, -1.0, 1.0)
        key = (event.user_id, event.recommendation_id)
        current = self._feedback_weights[key]
        self._feedback_weights[key] = round((0.80 * current) + (0.20 * reward), 4)

    def feedback_snapshot(self, user_id: str) -> dict[str, float]:
        return {
            item_id: reward
            for (known_user_id, item_id), reward in self._feedback_weights.items()
            if known_user_id == user_id
        }


class DigitalTwin:
    """Virtual behavioral profile derived from interaction history."""

    def build_profile(self, records: Sequence[InteractionRecord]) -> dict[str, Any]:
        if not records:
            return {
                "dominant_emotion": "neutral",
                "emotional_stability": 1.0,
                "engagement_level": "new",
                "average_energy": EMOTION_ENERGY_SCORES["neutral"],
                "stress_alert": False,
                "interaction_count": 0,
            }

        emotion_counts = Counter(record.final_emotion for record in records)
        dominant_emotion = emotion_counts.most_common(1)[0][0]
        energies = [EMOTION_ENERGY_SCORES[normalize_emotion(record.final_emotion)] for record in records]
        average_energy = mean(energies)
        variance = mean([(energy - average_energy) ** 2 for energy in energies]) if len(energies) > 1 else 0.0
        stability = clamp(1.0 - sqrt(variance))
        stress_count = sum(1 for record in records[-5:] if record.final_emotion in {"stressed", "sad", "angry", "anxious"})

        if len(records) >= 10:
            engagement = "high"
        elif len(records) >= 4:
            engagement = "moderate"
        else:
            engagement = "low"

        return {
            "dominant_emotion": dominant_emotion,
            "emotional_stability": round(stability, 4),
            "engagement_level": engagement,
            "average_energy": round(average_energy, 4),
            "stress_alert": stress_count >= 3,
            "interaction_count": len(records),
            "emotion_distribution": dict(emotion_counts),
        }


class AnalyticsEngine:
    """Lightweight analytical routines for emotional records."""

    def preprocess(self, records: Iterable[InteractionRecord], user_id: str | None = None) -> list[InteractionRecord]:
        cleaned = []
        for record in records:
            if user_id and record.user_id != user_id:
                continue
            if not record.final_emotion or not record.timestamp:
                continue
            cleaned.append(record)
        return sorted(cleaned, key=lambda record: record.timestamp)

    def mood_distribution(self, records: Sequence[InteractionRecord]) -> dict[str, int]:
        return dict(Counter(record.final_emotion for record in records))

    def emotional_energy_score(self, records: Sequence[InteractionRecord]) -> float:
        if not records:
            return EMOTION_ENERGY_SCORES["neutral"]
        return round(mean(EMOTION_ENERGY_SCORES[normalize_emotion(record.final_emotion)] for record in records), 4)

    def feedback_summary(self, records: Sequence[InteractionRecord]) -> dict[str, Any]:
        rewards = [record.feedback_reward for record in records if record.feedback_reward is not None]
        if not rewards:
            return {"feedback_count": 0, "average_reward": 0.0, "positive_ratio": 0.0}
        positive = sum(1 for reward in rewards if reward > 0)
        return {
            "feedback_count": len(rewards),
            "average_reward": round(mean(rewards), 4),
            "positive_ratio": round(positive / len(rewards), 4),
        }

    def detect_anomalies(self, records: Sequence[InteractionRecord]) -> list[str]:
        if len(records) < 3:
            return []

        recent = records[-5:]
        alerts = []
        low_energy_count = sum(1 for record in recent if record.final_emotion in {"sad", "stressed", "angry", "anxious"})
        if low_energy_count >= 3:
            alerts.append("Repeated low-energy or high-pressure emotions detected.")

        tired_count = sum(1 for record in recent if record.final_emotion == "tired")
        if tired_count >= 3:
            alerts.append("Fatigue pattern detected across recent interactions.")

        if self.emotional_energy_score(recent) < 0.35:
            alerts.append("Emotional energy score is below the wellness threshold.")

        return alerts

    def weekly_report(self, records: Sequence[InteractionRecord]) -> dict[str, Any]:
        cleaned = self.preprocess(records)
        if not cleaned:
            return {
                "interaction_count": 0,
                "mood_distribution": {},
                "dominant_emotion": "neutral",
                "emotional_energy_score": EMOTION_ENERGY_SCORES["neutral"],
                "feedback": {"feedback_count": 0, "average_reward": 0.0, "positive_ratio": 0.0},
                "anomalies": [],
            }

        distribution = self.mood_distribution(cleaned)
        dominant = max(distribution.items(), key=lambda pair: pair[1])[0]
        return {
            "interaction_count": len(cleaned),
            "mood_distribution": distribution,
            "dominant_emotion": dominant,
            "emotional_energy_score": self.emotional_energy_score(cleaned),
            "feedback": self.feedback_summary(cleaned),
            "anomalies": self.detect_anomalies(cleaned),
        }


class JsonlStorage:
    """Transparent local storage for interaction records."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, record: InteractionRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=True) + "\n")

    def load(self) -> list[InteractionRecord]:
        if not self.path.exists():
            return []

        records: list[InteractionRecord] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                records.append(InteractionRecord.from_dict(json.loads(line)))
        return records


class ExcelStorage:
    """Optional .xlsx storage adapter using openpyxl when installed."""

    HEADERS = (
        "user_id",
        "timestamp",
        "final_emotion",
        "confidence",
        "recommendation_id",
        "recommendation_category",
        "recommendation_title",
        "feedback_reward",
        "evidence_json",
    )

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def _openpyxl(self) -> Any:
        try:
            import openpyxl  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on optional dependency
            raise RuntimeError("Install openpyxl to use ExcelStorage.") from exc
        return openpyxl

    def append(self, record: InteractionRecord) -> None:
        openpyxl = self._openpyxl()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            workbook = openpyxl.load_workbook(self.path)
            sheet = workbook.active
        else:
            workbook = openpyxl.Workbook()
            sheet = workbook.active
            sheet.title = "emotion_logs"
            sheet.append(self.HEADERS)

        sheet.append(
            (
                record.user_id,
                record.timestamp.isoformat(),
                record.final_emotion,
                record.confidence,
                record.recommendation.item_id,
                record.recommendation.category,
                record.recommendation.title,
                record.feedback_reward,
                json.dumps([asdict(item) for item in record.evidence], ensure_ascii=True),
            )
        )
        workbook.save(self.path)


class CognitiveEmotionSystem:
    """Application service that coordinates detection, fusion, analytics, and advice."""

    def __init__(
        self,
        storage: JsonlStorage | ExcelStorage | None = None,
        fusion_engine: EmotionFusionEngine | None = None,
        recommendation_engine: RecommendationEngine | None = None,
    ) -> None:
        self.emoji_analyzer = EmojiSentimentAnalyzer()
        self.text_analyzer = TextSentimentAnalyzer()
        self.time_analyzer = TimeContextAnalyzer()
        self.face_detector = FaceMoodDetector()
        self.gesture_recognizer = GestureRecognizer()
        self.fusion_engine = fusion_engine or EmotionFusionEngine()
        self.recommendation_engine = recommendation_engine or RecommendationEngine()
        self.analytics = AnalyticsEngine()
        self.digital_twin = DigitalTwin()
        self.storage = storage
        self._session_records: list[InteractionRecord] = []

    def process_interaction(
        self,
        user_id: str,
        *,
        emoji_input: str | None = None,
        voice_text: str | None = None,
        face_features: Mapping[str, float] | None = None,
        gesture_features: Mapping[str, float] | None = None,
        timestamp: datetime | None = None,
    ) -> InteractionRecord:
        timestamp = timestamp or datetime.now(timezone.utc)
        evidence: list[EmotionEvidence] = []

        face_features = face_features or {}
        face_evidence = self.face_detector.analyze_features(
            smile_intensity=face_features.get("smile_intensity"),
            brow_tension=face_features.get("brow_tension"),
            eye_openness=face_features.get("eye_openness"),
        )
        if face_evidence:
            evidence.append(face_evidence)

        emoji_evidence = self.emoji_analyzer.analyze(emoji_input)
        if emoji_evidence:
            evidence.append(emoji_evidence)

        voice_evidence = self.text_analyzer.analyze(voice_text, modality="voice")
        if voice_evidence:
            evidence.append(voice_evidence)

        gesture_features = gesture_features or {}
        gesture_evidence = self.gesture_recognizer.analyze_features(
            motion_level=gesture_features.get("motion_level"),
            open_hand_density=gesture_features.get("open_hand_density"),
        )
        if gesture_evidence:
            evidence.append(gesture_evidence)

        evidence.append(self.time_analyzer.analyze(timestamp))
        fusion = self.fusion_engine.fuse(evidence)

        user_records = self.analytics.preprocess(self.records, user_id=user_id)
        profile = self.digital_twin.build_profile(user_records)
        recommendation = self.recommendation_engine.recommend(user_id, fusion.final_emotion, profile)

        record = InteractionRecord(
            user_id=user_id,
            timestamp=timestamp,
            evidence=fusion.evidence,
            final_emotion=fusion.final_emotion,
            confidence=fusion.confidence,
            recommendation=recommendation,
        )
        self._session_records.append(record)
        if self.storage:
            self.storage.append(record)
        return record

    @property
    def records(self) -> list[InteractionRecord]:
        if isinstance(self.storage, JsonlStorage):
            return self.storage.load()
        return list(self._session_records)

    def submit_feedback(
        self,
        user_id: str,
        recommendation_id: str,
        reward: float,
        note: str = "",
    ) -> FeedbackEvent:
        event = FeedbackEvent(user_id=user_id, recommendation_id=recommendation_id, reward=reward, note=note)
        self.recommendation_engine.submit_feedback(event)
        return event

    def profile_for(self, user_id: str) -> dict[str, Any]:
        return self.digital_twin.build_profile(self.analytics.preprocess(self.records, user_id=user_id))

    def weekly_report_for(self, user_id: str) -> dict[str, Any]:
        return self.analytics.weekly_report(self.analytics.preprocess(self.records, user_id=user_id))
