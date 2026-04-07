# GuneeshDua-MajorProject

Cognitive Emotional Intelligence, reinforcement-learning logic, Digital Emotional Twin logging, and multi-modal AI fusion with ethical AI monitoring.

## What is in this repository

This repository contains a single-file Streamlit application in `app.py` that integrates:

- Multi-modal emotion input and fusion:
  - face image
  - emoji
  - voice-to-text (Hugging Face ASR + manual fallback)
  - free-text context
- Upgraded CNN pipeline with:
  - Global Average Pooling (GAP)
  - optional EfficientNetV2B0 feature backbone
  - Grad-CAM explainability
- Evaluation metrics:
  - precision
  - recall
  - F1-score
  - confusion matrix
  - ROC-AUC where possible
- Digital Emotional Twin CSV logging
- History-aware no-repetition recommender with feedback-driven reward updates
- Spotify optional integration
- Automatic sampled dataset preparation for public emotion datasets
- Resource catalog generation with 100+ entries and CSV export

## Project files

- `app.py` - main Streamlit app (single-source code)
- `requirements.txt` - required Python dependencies
- `.gitignore` - ignores local runtime artifacts and model/data folders
- `docs/EXECUTION_GUIDE_WINDOWS11.md` - full VS Code setup and execution guide
- `docs/IEEE_RESEARCH_PAPER_CEI.md` - complete IEEE-style research paper draft
- `docs/VIVA_QUESTIONS_AND_DEFENSE.md` - viva questions, answers, and defense points
- `deliverables/CEI_Research_Paper_IEEE.docx` - downloadable Word file

Direct download link:

- [Download IEEE Word Paper](deliverables/CEI_Research_Paper_IEEE.docx)

Generated at runtime:

- `resource_catalog.csv`
- `cei_twin_log.csv`
- `recommender_stats.csv`
- `models/`
- `dataset/`

## Recommended environment (target laptop)

- Windows 11
- Intel Core i5-1035G1
- 4 GB RAM

Recommended runtime settings for 4 GB RAM:

- dataset sample size: 600-1200 images
- batch size: 4 or 8
- epochs: 1-3 for demo training
- smaller image size first (64 or 96)

## Quick start

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Optional catalog export:

```bash
python app.py --export-catalog
```

## Free API option

Use a free Hugging Face token:

1. Create account: https://huggingface.co/join
2. Settings -> Access Tokens -> New token (Read scope)
3. Paste token in sidebar fields inside the app

The app still works without paid APIs using local heuristics and fallback logic.
