# GuneeshDua-MajorProject

Cognitive Emotion Intelligence and Adaptive Lifestyle System with multimodal
emotion fusion, reinforcement-inspired recommendation logic, digital emotional
twin analytics, and ethical lightweight AI design.

## What is implemented

- Multimodal emotion evidence collection:
  - facial feature mood heuristics
  - emoji sentiment mapping
  - voice transcript sentiment mapping
  - gesture feature recognition
  - time-context analysis
- Weighted emotion fusion engine.
- Adaptive lifestyle, music, and AI coach recommendation engine.
- Feedback reward updates for recommendation personalization.
- Emotional digital twin profile generation.
- Weekly report analytics, emotional energy score, and anomaly detection.
- JSONL storage by default and optional Excel `.xlsx` storage with OpenPyXL.
- Chapter 4 architecture and workflow flowcharts in
  [`docs/architecture.md`](docs/architecture.md).
- Downloadable Word document containing the workflow architecture models:
  [`docs/chapter4_workflow_architecture_models.docx`](docs/chapter4_workflow_architecture_models.docx).

## Run the prototype

```bash
python3 run_system.py \
  --user-id student-1 \
  --emoji stressed \
  --voice-text "I feel pressure and deadline stress" \
  --smile 0.1 \
  --brow 0.85 \
  --eyes 0.7 \
  --motion 0.8 \
  --hand-density 0.2
```

The command prints:

- final fused emotion and confidence
- modality evidence
- personalized recommendation
- digital twin profile
- weekly analytics report

## Run tests

```bash
python3 -m unittest discover -s tests
```

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for the Chapter 4 research
methodology, system implementation architecture, and Mermaid flowcharts for:

- overall research workflow
- system architecture
- multimodal emotion detection
- weighted fusion engine
- reinforcement-inspired recommendation loop
- emotional digital twin modeling
- gesture recognition workflow
- data analytics and weekly reporting

Download the Word version here:
[`chapter4_workflow_architecture_models.docx`](docs/chapter4_workflow_architecture_models.docx).
