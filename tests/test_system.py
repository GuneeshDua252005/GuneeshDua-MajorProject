from datetime import datetime, timezone
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cognitive_emotion_system import (
    AnalyticsEngine,
    CognitiveEmotionSystem,
    EmotionEvidence,
    EmotionFusionEngine,
)


class EmotionFusionEngineTest(unittest.TestCase):
    def test_face_signal_receives_highest_weight(self) -> None:
        engine = EmotionFusionEngine()
        result = engine.fuse(
            [
                EmotionEvidence("face", "joyful", 0.90),
                EmotionEvidence("emoji", "sad", 0.90),
                EmotionEvidence("voice", "neutral", 0.40),
            ]
        )

        self.assertEqual(result.final_emotion, "joyful")
        self.assertGreater(result.confidence, result.scores["sad"])

    def test_empty_fusion_defaults_to_neutral(self) -> None:
        result = EmotionFusionEngine().fuse([])

        self.assertEqual(result.final_emotion, "neutral")
        self.assertEqual(result.confidence, 0.0)


class CognitiveEmotionSystemTest(unittest.TestCase):
    def test_process_interaction_generates_recommendation_and_profile(self) -> None:
        system = CognitiveEmotionSystem()
        record = system.process_interaction(
            "student-1",
            emoji_input="stressed",
            voice_text="I feel pressure and deadline stress",
            face_features={"smile_intensity": 0.1, "brow_tension": 0.85, "eye_openness": 0.7},
            gesture_features={"motion_level": 0.8, "open_hand_density": 0.2},
            timestamp=datetime(2026, 5, 7, 10, tzinfo=timezone.utc),
        )

        self.assertEqual(record.final_emotion, "stressed")
        self.assertEqual(record.recommendation.emotion, "stressed")
        self.assertEqual(system.profile_for("student-1")["dominant_emotion"], "stressed")

    def test_weekly_report_detects_repeated_low_energy_pattern(self) -> None:
        system = CognitiveEmotionSystem()
        for index in range(3):
            system.process_interaction(
                "student-2",
                voice_text="I am sad and stressed",
                face_features={"smile_intensity": 0.1, "brow_tension": 0.85, "eye_openness": 0.65},
                timestamp=datetime(2026, 5, 7, 10 + index, tzinfo=timezone.utc),
            )

        report = system.weekly_report_for("student-2")

        self.assertEqual(report["dominant_emotion"], "stressed")
        self.assertTrue(report["anomalies"])


class AnalyticsEngineTest(unittest.TestCase):
    def test_energy_score_defaults_to_neutral_for_empty_records(self) -> None:
        score = AnalyticsEngine().emotional_energy_score([])

        self.assertEqual(score, 0.60)


if __name__ == "__main__":
    unittest.main()
