# GuneeshDua-MajorProject

Cognitive Emotional Intelligence, reinforcement-learning logic, Digital Emotional Twin logging, multi-modal AI fusion, Grad-CAM explainability, and ethical AI monitoring in a single-file Streamlit application.

## What is in this repository

This repository contains a single-source Python major project in `app.py` designed for:

- multimodal emotion input and fusion
  - face image
  - emoji input
  - voice-to-text fallback
  - free-text context
- EfficientNetV2 CNN training with Global Average Pooling
- evaluation metrics
  - precision
  - recall
  - F1
  - confusion matrix
  - ROC-AUC where possible
- Grad-CAM explainability
- a history-aware no-repetition recommender
- free Hugging Face integrations for chat and ASR
- Digital Emotional Twin CSV logging
- a 100+ item resource catalog with CSV export
- automatic sampled dataset preparation for public emotion datasets
- viva support, setup documentation, APK-style deployment guidance, and IEEE-style research-paper drafting support

## Main files

- `app.py` - single-file Streamlit application
- `requirements.txt` - default Python dependencies
- `.gitignore` - Python, Streamlit, dataset, and model ignore rules
- `docs/VS_CODE_WINDOWS_SETUP.md` - exact step-by-step setup guide for Windows 11 + VS Code
- `docs/RESEARCH_PAPER_IEEE.md` - IEEE-style research-paper draft content
- `docs/LITERATURE_MATRIX.md` - extracted details from recent research papers
- `docs/VIVA_GUIDE.md` - high-value viva questions with detailed explanations
- `docs/APK_CONVERSION_GUIDE.md` - free installable/PWA/APK-style deployment guidance

## Design target

Recommended target device from your prompt:

- Windows 11
- Intel Core i5-1035G1
- 8 GB RAM
- Intel UHD Graphics
- VS Code
- Python 3.11

The app is intentionally designed to degrade gracefully:

- it runs even without TensorFlow
- it becomes stronger when TensorFlow is installed and a local model is trained
- it upgrades further when a free Hugging Face token is provided

## Free AI integration choice

Paid APIs like OpenAI and Spotify are not required here. This project uses:

- **Hugging Face** for optional free token-based:
  - chat completion
  - speech-to-text
  - remote facial emotion inference fallback
- **direct search links** for Spotify, YouTube, and YouTube Music instead of paid music APIs

Why Hugging Face is suitable:

1. It allows account-based token generation for free tiers.
2. It supports open models instead of paid proprietary endpoints.
3. The same project can still run fully offline or semi-offline without a token.

## Quick start

### 1. Create a virtual environment

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Optional: install TensorFlow for local CNN training

```powershell
pip install tensorflow-cpu
```

TensorFlow is optional because it is heavier. Without it, the project still works with:

- heuristic face analysis
- text analysis
- emoji analysis
- voice transcription through Hugging Face
- chatbot support through Hugging Face
- recommendation engine
- Digital Emotional Twin logging

### 4. Optional: create a free Hugging Face token

1. Create a free account on Hugging Face.
2. Open account settings.
3. Generate an access token.
4. Paste that token in the Streamlit sidebar when the app starts.

Optional environment variable:

```powershell
$env:HF_TOKEN="your_huggingface_token"
```

### 5. Run the app

```powershell
streamlit run app.py
```

## Optional CLI utilities

Export the 100+ item catalog:

```powershell
python app.py --export-catalog
```

Prepare a sampled dataset:

```powershell
python app.py --prepare-dataset --dataset-source FER2013 --sample-size 900
```

## Folder and output structure

These are created automatically during runtime:

- `dataset/`
  - `train/`
  - `val/`
  - `test/`
  - `dataset_manifest.json`
- `models/`
  - `efficientnetv2_emotion.keras`
  - `efficientnetv2_emotion_metadata.json`
- `cei_twin_log.csv`
- `recommender_stats.csv`
- `resource_catalog.csv`

## Practical low-RAM settings

For the target laptop:

- keep automatic download samples around 600-1200 images
- keep batch size at 4 or 8
- keep epochs at 1-3
- start with fewer classes or smaller samples before larger experiments

## Important note about the CNN architecture

Your latest requirement asked to avoid older MobileNetV2 as the main architecture.
This implementation therefore uses:

- **EfficientNetV2B0**
- **Global Average Pooling**
- **Dropout**
- **Dense softmax head**
- **Grad-CAM explainability**

This is a better fit for a modern lightweight academic prototype than using MobileNetV2 as the primary pipeline.

## Documentation

Read the detailed guides in `docs/`:

- `VS_CODE_WINDOWS_SETUP.md`
- `RESEARCH_PAPER_IEEE.md`
- `LITERATURE_MATRIX.md`
- `VIVA_GUIDE.md`
- `APK_CONVERSION_GUIDE.md`

## Academic and ethical positioning

This project should be presented as:

- an **emotion-aware lifestyle support prototype**
- an **explainable AI system**
- a **consent-based Digital Emotional Twin framework**
- a **student major project**, not a medical diagnosis system

It uses confidence warnings, user override, and optional logging to support more responsible use of emotion AI.
