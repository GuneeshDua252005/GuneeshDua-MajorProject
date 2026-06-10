# Cognitive Emotion Intelligence and Adaptive Lifestyle System

Single-file Streamlit + PyTorch major-project implementation for:

- cognitive emotional intelligence
- multimodal emotion fusion
- digital emotional twin logging
- explainable AI with Grad-CAM
- EfficientNetV2-based facial emotion modeling
- history-aware lifestyle and media recommendation
- ethical AI monitoring
- optional Hugging Face and Spotify integrations

This repository is designed for VS Code execution on Windows 11 and remains a single Python application in `app.py` with no separate backend service.

## Important compatibility note

For the requested Windows laptop configuration, the most stable setup is:

- Python 3.11 or 3.12
- VS Code
- CPU execution
- PyTorch + torchvision

Stable PyTorch support on Windows can lag behind the newest Python release, so **Python 3.14 is not the safest choice today for this project**. If you need reliable installation on Windows 11, create the virtual environment with Python 3.11 or 3.12.

## What is inside this repository

- `app.py` - the complete single-file Streamlit application
- `requirements.txt` - essential packages only
- `.gitignore` - project-safe ignore rules for caches, datasets, generated CSVs, and models
- `docs/RESEARCH_PAPER_IEEE_DRAFT.md` - IEEE-style report draft content
- `docs/VIVA_GUIDE.md` - viva questions, answers, project differentiation, significance, research gaps, and future scope
- `models/.gitkeep` - keeps the model folder in git
- `dataset/.gitkeep` - keeps the dataset folder in git

## Folder structure to keep in your project

Create or keep the following structure:

```text
MajorProject/
|-- app.py
|-- requirements.txt
|-- .gitignore
|-- README.md
|-- dataset/
|   |-- train/
|   |-- val/
|   `-- test/
|-- models/
|-- docs/
|   |-- RESEARCH_PAPER_IEEE_DRAFT.md
|   `-- VIVA_GUIDE.md
```

The app can also auto-create missing runtime folders when it starts.

## Step-by-step VS Code setup on Windows 11

### 1. Install the required software

Install the following on your laptop:

- Python 3.11 or Python 3.12
- VS Code
- VS Code Python extension

Optional:

- Git

### 2. Open the project in VS Code

- Launch VS Code
- Open the folder that contains this repository
- Open the VS Code terminal

### 3. Create the virtual environment

Use PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 4. Install the dependencies

```powershell
pip install -r requirements.txt
```

### 5. Optional environment variables

You can run the app without any paid API.

Optional free integrations:

- Hugging Face Inference API token
- Spotify Developer API credentials

PowerShell examples:

```powershell
$env:HF_TOKEN="your_hugging_face_token"
$env:SPOTIPY_CLIENT_ID="your_spotify_client_id"
$env:SPOTIPY_CLIENT_SECRET="your_spotify_client_secret"
```

### 6. Start the app

```powershell
streamlit run app.py
```

The browser will open on localhost. Use:

- text input
- emoji input
- face image upload or camera capture
- browser microphone input

### 7. Prepare the dataset

You have two ways to prepare data:

#### Option A - automatic Hugging Face dataset sampling

Use the **Dataset Preparation** section inside the app.

It can:

- download a public image dataset from Hugging Face
- detect image and label columns
- normalize labels
- create:
  - `dataset/train`
  - `dataset/val`
  - `dataset/test`
- save `dataset_manifest.json`

#### Option B - manual local dataset

Organize images as:

```text
dataset/
|-- train/
|   |-- happy/
|   |-- sad/
|   |-- angry/
|   `-- ...
|-- val/
|   |-- happy/
|   `-- ...
`-- test/
    |-- happy/
    `-- ...
```

### 8. Train the PyTorch model

Inside the app:

- open the **Training and Grad-CAM** section
- choose CPU-friendly settings
- recommended:
  - batch size: 4 or 8
  - epochs: 1 to 3 for a demo run

Saved outputs:

- `models/efficientnet_emotion.pth`
- `models/efficientnet_emotion_metadata.json`

### 9. Generated files during execution

The program can generate:

- `cei_twin_log.csv`
- `recommender_stats.csv`
- `resource_catalog.csv`
- `dataset_manifest.json`
- `models/efficientnet_emotion.pth`
- `models/efficientnet_emotion_metadata.json`

## What the single-file application does

The app combines:

- PyTorch-based facial emotion analysis with EfficientNetV2-S
- Grad-CAM explainability
- multimodal fusion from image, text, emoji, and voice transcript
- digital emotional twin logging
- ethical AI alerts and risk-aware messaging
- lightweight reinforcement-learning style recommendation logic
- no-repeat recommendation memory
- optional Hugging Face text generation
- optional Spotify search
- CSV export for logs and catalog
- viva and report guidance inside the repository

## Why EfficientNetV2 instead of MobileNetV2

This project replaces the older MobileNetV2-style direction with a stronger PyTorch-based CNN backbone:

- better modern performance/efficiency trade-off
- built-in global average pooling pipeline
- cleaner PyTorch support
- more suitable for Grad-CAM demonstrations in a major project

## Free API guidance

### Hugging Face

Hugging Face is the recommended free optional integration here.

Why it is suitable:

- free account creation
- free token generation
- easy REST API calls
- useful for text generation or classification

How to create the token:

1. Create a free account at https://huggingface.co/
2. Open **Settings -> Access Tokens**
3. Create a new token
4. Copy the token into the app sidebar or set `HF_TOKEN`

The app does **not** and should **not** create a token automatically from your username and password. Manual token creation is the secure and correct approach.

### Spotify

Spotify developer credentials are also free to create, but optional.

How to create them:

1. Go to https://developer.spotify.com/dashboard
2. Log in with your Spotify account
3. Create an app
4. Copy the client ID and client secret

If you do not provide Spotify credentials, the app still works using the built-in resource catalog and search links.

## APK conversion trick using a free route

This Streamlit project is best demonstrated in one of these ways:

### Option 1 - install as a web app

- run the app locally or deploy it
- open in Chrome or Edge on Android
- choose **Add to Home Screen**

This gives a PWA-like installable experience.

### Option 2 - free APK wrapper idea

If your HOD specifically asks for APK packaging, use a free WebView wrapper or PWA-to-APK workflow around the hosted Streamlit app instead of rewriting the whole system in native Android code.

This is the easiest free demonstration path, but the repository itself remains a Python Streamlit application.

## Recommended demo settings for the requested laptop

For Intel i5 + 8 GB RAM:

- CPU mode
- batch size 4
- epochs 1 to 3
- sample-size 600 to 1200 images
- use compressed images when possible
- do not enable multiple large integrations at the same time

## Safety note

This project is for educational and project-demonstration use. It is not a clinical diagnosis tool and should not replace professional mental-health or medical support.
