# Cognitive Emotion Intelligence and Adaptive Lifestyle System

Single-file PyTorch + Streamlit major project for:

- cognitive emotional intelligence
- adaptive therapeutic chatbot interaction
- text, voice, and facial emotion fusion
- explainable AI with Grad-CAM
- digital emotional twin logging
- ethical AI monitoring
- adaptive lifestyle and music-style recommendation

The repository is intentionally kept simple and VS Code friendly:

- `app.py` - complete single-file application
- `requirements.txt` - essential Python dependencies
- `.gitignore` - excludes virtual environments, caches, logs, datasets, and trained models

## What this project does

The application combines:

- text emotion detection with Hugging Face transformers plus NLTK-based enrichment
- voice capture, speech-to-text, and lightweight voice-tone heuristics
- webcam-based facial emotion detection with OpenCV
- a PyTorch `EfficientNetV2-S` facial emotion model
- Grad-CAM explainability
- a fused emotional state score
- an animated mascot chatbot that adapts its tone
- CSV logging for a digital emotional twin
- a no-repetition recommendation engine with feedback memory
- dataset training and evaluation inside the same app

## Windows 11 + VS Code setup

Recommended target configuration:

- Windows 11
- Intel Core i5
- 8 GB RAM
- Python 3.14 compatible codebase

Practical note:

- the cloud environment used to build this project currently has Python 3.12
- the code is written to remain compatible with Python 3.14 syntax expectations
- for low-RAM laptops, keep training very small: batch size `4` or `8`, epochs `1` to `3`

### 1. Create project folder

Open VS Code and place these files inside one folder:

- `app.py`
- `requirements.txt`
- `.gitignore`

Optional folders created automatically or by you:

- `dataset/`
- `models/`

Recommended dataset layout if you want to train the face model:

```text
dataset/
  train/
    happy/
    sad/
    angry/
    neutral/
  val/
    happy/
    sad/
    angry/
    neutral/
  test/
    happy/
    sad/
    angry/
    neutral/
```

### 2. Create virtual environment

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Command Prompt:

```bat
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

Direct package command:

```powershell
python -m pip install torch torchvision transformers streamlit opencv-python numpy nltk SpeechRecognition pyttsx3
```

### 4. Optional free Hugging Face token

This project avoids paid OpenAI APIs.

Use Hugging Face instead:

1. Create a free account at `https://huggingface.co`
2. Go to Settings -> Access Tokens
3. Create a token with read access
4. Set it in PowerShell:

```powershell
$env:HF_TOKEN="your_huggingface_token"
```

Important:

- do not use your password directly inside code
- use an access token
- public transformer models can still run without a token, but a token is useful for API inference

### 5. Run the application

```powershell
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

### 6. Optional command line utility

Export the internal 100+ item resource catalog:

```powershell
python app.py --export-catalog
```

This generates:

- `resource_catalog.csv`

## Files generated during execution

The program may create:

- `models/efficientnet_emotion.pth`
- `models/efficientnet_emotion_metadata.json`
- `dataset_manifest.json`
- `resource_catalog.csv`
- `cei_twin_log.csv`
- `recommender_stats.csv`

## Why Hugging Face is the best free integration here

The user requested a free API alternative to paid GPT/OpenAI access.

Hugging Face is the best fit because:

- it offers free public transformer models
- it supports both local inference and hosted inference
- it works well with Python and Streamlit
- it does not force a paid API for a basic academic demo
- it allows token-based security instead of embedding passwords

Spotify is kept as a free search-link integration in this implementation rather than authenticated playlist control, so the demo remains simple and free.

## Why EfficientNetV2 instead of MobileNetV2

The earlier project concept mentioned MobileNetV2, but this implementation upgrades the face model path to `EfficientNetV2-S` because it offers:

- better feature extraction capacity
- strong transfer learning performance
- modern PyTorch support
- practical inference speed for laptop demos
- good compatibility with Grad-CAM visual explanation

## Explainability and architecture notes

### Grad-CAM

Grad-CAM highlights which image regions most influenced the facial emotion prediction. In viva and review settings, this helps explain:

- where the model focused
- whether it looked at the mouth, eyes, or broader facial region
- whether the output appears reasonable

### GAP - Global Average Pooling

Global Average Pooling is used in modern CNNs to summarize deep feature maps before the final classifier layer. It helps:

- reduce parameter count
- improve generalization
- make class activation visualization more meaningful

### Ethical AI monitoring

This project includes ethical monitoring notes because emotion systems can be biased or overconfident. The app surfaces:

- fallback warnings
- low-confidence warnings
- non-diagnostic disclaimers
- partial modality alerts

## APK conversion trick for free

For a free app-like experience:

1. run the Streamlit app locally or on a simple host
2. open it in Chrome or Edge on Android
3. use "Add to Home Screen" or "Install App"

This gives a lightweight installable workflow without building a separate native Android application.

## Research and viva support

Additional academic support files can be added to the repository, such as:

- IEEE-style research paper draft
- viva questions and answers
- future scope and research gap analysis

## Notes on realism

The user asked for a "production-ready" system with animated mascot, voice, vision, therapeutic dialogue, and miraculous results.

This repository implements a realistic academic major-project version of that vision:

- single-file architecture
- real Streamlit demo
- real PyTorch model support
- real Grad-CAM
- practical multimodal fusion
- free API option

It intentionally avoids fake claims, paid-only dependencies, and unstable multi-service architecture so that the project remains executable on a student laptop.
