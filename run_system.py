"""Command-line runner for the Cognitive Emotion Intelligence prototype."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cognitive_emotion_system import CognitiveEmotionSystem, JsonlStorage  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a multimodal emotion fusion and adaptive lifestyle recommendation session."
    )
    parser.add_argument("--user-id", default="demo-user", help="User identifier for personalization.")
    parser.add_argument("--emoji", default=None, help="Emoji or short sentiment input, for example ':)' or 'stressed'.")
    parser.add_argument("--voice-text", default=None, help="Speech-to-text transcript or typed voice simulation.")
    parser.add_argument("--smile", type=float, default=None, help="Face feature: smile intensity from 0.0 to 1.0.")
    parser.add_argument("--brow", type=float, default=None, help="Face feature: brow tension from 0.0 to 1.0.")
    parser.add_argument("--eyes", type=float, default=None, help="Face feature: eye openness from 0.0 to 1.0.")
    parser.add_argument("--motion", type=float, default=None, help="Gesture feature: motion level from 0.0 to 1.0.")
    parser.add_argument("--hand-density", type=float, default=None, help="Gesture feature: open hand density from 0.0 to 1.0.")
    parser.add_argument(
        "--storage",
        default="data/emotion_logs.jsonl",
        help="JSONL path for lightweight interaction history.",
    )
    parser.add_argument("--feedback", type=float, default=None, help="Optional reward signal from -1.0 to 1.0.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    system = CognitiveEmotionSystem(storage=JsonlStorage(args.storage))
    record = system.process_interaction(
        user_id=args.user_id,
        emoji_input=args.emoji,
        voice_text=args.voice_text,
        face_features={
            "smile_intensity": args.smile,
            "brow_tension": args.brow,
            "eye_openness": args.eyes,
        },
        gesture_features={
            "motion_level": args.motion,
            "open_hand_density": args.hand_density,
        },
        timestamp=datetime.now(timezone.utc),
    )

    if args.feedback is not None:
        system.submit_feedback(
            user_id=args.user_id,
            recommendation_id=record.recommendation.item_id,
            reward=args.feedback,
            note="CLI feedback",
        )

    response = {
        "final_emotion": record.final_emotion,
        "confidence": record.confidence,
        "evidence": [
            {
                "modality": evidence.modality,
                "emotion": evidence.emotion,
                "confidence": evidence.confidence,
                "metadata": evidence.metadata,
            }
            for evidence in record.evidence
        ],
        "recommendation": {
            "id": record.recommendation.item_id,
            "category": record.recommendation.category,
            "title": record.recommendation.title,
            "reason": record.recommendation.reason,
            "score": record.recommendation.score,
        },
        "digital_twin": system.profile_for(args.user_id),
        "weekly_report": system.weekly_report_for(args.user_id),
    }
    print(json.dumps(response, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
