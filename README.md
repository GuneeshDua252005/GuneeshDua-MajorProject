# GuneeshDua-MajorProject

Cognitive Emotion Intelligence & Adaptive Lifestyle System (lightweight implementation) with:

- Multi-modal emotion detection (face, emoji/text, voice text, time context)
- Weighted Emotion Fusion Engine
- Reinforcement-inspired recommendation updates via feedback
- Emotional Digital Twin profiling
- Excel-based interaction storage (`.xlsx`)
- Analytics (trend, energy score, anomaly detection, weekly graph)
- Streamlit dashboard for visualization

## Project Files

- `emotion_adaptive_system.py` - Main implementation and CLI runner
- `dashboard.py` - Streamlit analytics dashboard
- `docs/workflow_architecture.md` - Flowchart architecture (Mermaid)
- `docs/Cognitive_Emotion_Workflow_Architecture_Models.docx` - Word document with workflow flowchart models
- `scripts/generate_workflow_docx.py` - Generator script for the Word flowchart document
- `requirements.txt` - Python dependencies

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the System (CLI)

```bash
python emotion_adaptive_system.py \
  --user-id user_1 \
  --emoji-text "😢 feeling low" \
  --voice-text "I feel stressed and tired today"
```

Output is JSON with:

- detected modality signals
- fused emotion + confidence
- personalized recommendations
- digital twin profile
- analytics summary
- AI coach message

## Apply Feedback (Adaptive Learning)

```bash
python emotion_adaptive_system.py \
  --user-id user_1 \
  --emoji-text "🙂 okay" \
  --feedback-emotion neutral \
  --feedback-item "Pomodoro focus block" \
  --feedback-reward 1.0
```

`feedback-reward` range: `-1.0` (negative) to `1.0` (positive)

## Run Dashboard

```bash
streamlit run dashboard.py
```

By default, dashboard reads `emotion_logs.xlsx`.

## Workflow Architecture (Flowcharts)

See:

- [`docs/workflow_architecture.md`](docs/workflow_architecture.md)
- [`docs/Cognitive_Emotion_Workflow_Architecture_Models.docx`](docs/Cognitive_Emotion_Workflow_Architecture_Models.docx)

Included flowcharts:

1. End-to-end system workflow
2. Weighted emotion fusion model
3. Reinforcement recommendation feedback loop
4. Data collection and analysis pipeline

## Word Document Download Reference Link

After pushing this branch, download directly from:

`https://github.com/GuneeshDua252005/GuneeshDua-MajorProject/raw/cursor/emotion-system-implementation-7db7/docs/Cognitive_Emotion_Workflow_Architecture_Models.docx`

## Regenerate Word Document

```bash
python3 scripts/generate_workflow_docx.py
```
