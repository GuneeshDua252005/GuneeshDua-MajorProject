# GuneeshDua-MajorProject

Cognitive Emotion Intelligence and Adaptive Lifestyle System with reinforcement-learning inspired feedback logic, Digital Emotional Twin logging, multimodal fusion, explainable AI, and ethical AI monitoring.

## What this repository contains

This repository now provides a complete single-file Python major project centered on:

- multimodal mood understanding from text, emoji, optional voice transcript, and face image
- a custom CNN with Global Average Pooling for emotion classification
- Grad-CAM explainability support
- a lightweight adaptive recommender with no downloadable MP3 dependency
- Digital Emotional Twin CSV logging
- free-first API integration using optional Hugging Face tokens
- VS Code and Windows 11 execution guidance
- an IEEE-style research paper package in Markdown, CSV, and Word document form

## Main files

- `app.py` - single-source-code Streamlit application
- `requirements.txt` - essential dependencies for execution
- `.gitignore` - ignores local runtime, dataset, and model artifacts
- `docs/setup_guide.md` - step-by-step VS Code and Windows 11 setup guide
- `docs/free_api_guide.md` - free Hugging Face token and API usage guide
- `docs/research_paper.md` - detailed IEEE-style research paper draft
- `docs/literature_matrix.csv` - paper-by-paper extraction table
- `docs/viva_questions.md` - major viva questions and detailed answers
- `docs/Cognitive_Emotion_Intelligence_Adaptive_Lifestyle_System_IEEE_Paper.docx` - Word document export of the paper
- `assets/paper_flowchart.txt` - text flowchart for the proposed system

## Project features

### 1. Multimodal mood analysis

The application supports:

- free-text context
- emoji self-check
- optional voice note transcription fallback
- optional face image input

The outputs from available modalities are fused using confidence-aware weighting.

### 2. Free API integration

The project does not depend on paid OpenAI or Spotify keys for its main workflow.

Instead, it supports:

- optional Hugging Face inference token for text emotion analysis
- optional Hugging Face generation model for reflective questions
- complete fallback behavior when no token is provided

### 3. Custom CNN + GAP + Grad-CAM

The visual pipeline is built around a compact custom convolutional architecture that uses:

- stacked convolution blocks
- batch normalization
- dropout
- Global Average Pooling
- Grad-CAM compatibility through the final convolution layer

This was intentionally chosen instead of depending on MobileNetV2.

### 4. Digital Emotional Twin

The system stores lightweight user-interaction traces in CSV form:

- `cei_twin_log.csv`
- `recommender_stats.csv`
- `resource_catalog.csv`

### 5. Dataset preparation and training tools

The app supports:

- ZIP-based dataset import
- sampled Hugging Face image dataset preparation
- train/validation/test split creation
- training and evaluation of the custom CNN

## Quick start

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

### Export the resource catalog

```powershell
python app.py --export-catalog
```

## Recommended runtime for low-RAM systems

For Windows 11 systems with 4 GB RAM:

- use Python 3.10 or 3.11
- keep dataset samples around 600 to 1200 images
- keep batch size at 4 or 8
- train for 1 to 3 epochs for demonstration
- avoid large-scale training on the device

## Notes for major project submission

- This system is supportive and assistive, not diagnostic.
- The codebase is intentionally single-file for easier academic demonstration.
- The repository includes both implementation and documentation deliverables.
- The Word paper is meant to be editable before final college submission.
