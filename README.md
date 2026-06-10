# GuneeshDua-MajorProject

Cognitive Emotional Intelligence, reinforcement-learning style recommendation, Digital Emotional Twin logging, and multi-modal AI fusion with an animated mascot chatbot.

## What is in this repository

This repository contains a single-file Streamlit application in `app.py` built for a major project demo. It uses:

- PyTorch only for the CNN pipeline
- EfficientNetV2-S instead of TensorFlow or Keras
- Grad-CAM explainability
- text, voice, face, and emoji based emotion fusion
- a Digital Emotional Twin CSV log
- a no-repeat adaptive recommender
- free Hugging Face integration for chatbot responses
- a lightweight animated mascot interface

## Files in the project

- `app.py` - main single-file Streamlit application
- `requirements.txt` - VS Code compatible dependency list
- `.gitignore` - ignores local runtime, datasets, and model artifacts

Generated during execution:

- `cei_twin_log.csv`
- `recommender_stats.csv`
- `resource_catalog.csv`
- `models/`
- `dataset/`

## Recommended target environment

This project is designed to stay practical on Windows 11 laptops with Intel i5 class CPUs and 8 GB RAM.

Recommended demo settings:

- keep batch size at 4 or 8
- keep epochs at 1 to 3
- start with fewer classes first
- use small sampled datasets for classroom demo runs

## Step-by-step execution in VS Code

### 1. Create the folder contents

Keep these files in one folder:

- `app.py`
- `requirements.txt`
- `.gitignore`

Optional folders created later:

- `dataset/`
- `models/`

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Optional free Hugging Face token

If you want free API-based chatbot generation:

1. Create a free account on https://huggingface.co
2. Open Settings -> Access Tokens
3. Create a read token
4. Set it in the VS Code terminal:

```powershell
$env:HF_TOKEN="your_hugging_face_token"
```

If you do not set a token, the app still works by falling back to local transformers or rule-based support mode.

### 5. Run the app

```powershell
streamlit run app.py
```

Then open the local URL shown by Streamlit, usually:

- `http://localhost:8501`

## What to write in `requirements.txt`

```text
torch
torchvision
transformers
streamlit
opencv-python
numpy
nltk
SpeechRecognition
pyttsx3
```

## Dataset folder structure for training

Create the dataset like this if you want to train your own model:

```text
dataset/
|-- train/
|   |-- happy/
|   |-- sad/
|   |-- angry/
|   `-- neutral/
|-- val/
|   |-- happy/
|   |-- sad/
|   |-- angry/
|   `-- neutral/
`-- test/
    |-- happy/
    |-- sad/
    |-- angry/
    `-- neutral/
```

## Why Hugging Face is used for free integration

- It provides free user accounts
- It supports free read tokens
- It works well with open models and local fallback
- It avoids paid GPT-only dependencies

## APK conversion trick

For a free near-app mobile deployment:

- host or run the Streamlit app
- open it in Chrome on Android
- use Add to Home Screen

If you later need an APK, wrap the same URL with a simple WebView project.

## Project highlights for viva

- PyTorch-only CNN architecture
- EfficientNetV2-S transfer learning
- Grad-CAM with global average pooling
- multimodal cognitive emotion fusion
- animated mascot chatbot
- free Hugging Face integration
- ethical AI monitoring
- Digital Emotional Twin logging
- reinforcement-learning style recommendation memory
