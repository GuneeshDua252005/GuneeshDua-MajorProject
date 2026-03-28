# GuneeshDua-MajorProject

Cognitive Emotional Intelligence, reinforcement-learning logic, Digital Emotional Twin logging, and multi-modal AI fusion with practical deployment guidance.

## What is in this repository

This repository now contains a single-file Streamlit application in `app.py` that brings together:

- multi-modal emotion input and fusion:
  - face image
  - emoji
  - voice-to-text fallback
  - free-text context
- transfer learning with MobileNetV2
- evaluation metrics:
  - precision
  - recall
  - F1
  - confusion matrix
  - ROC-AUC where possible
- Grad-CAM explainability
- a history-aware no-repetition recommender
- Spotify and OpenAI integrations
- Digital Emotional Twin CSV logging
- a structured 100+ item resource catalog with CSV export
- automatic sampled dataset preparation for recent public emotion datasets

## Files

- `app.py` - main Streamlit app
- `requirements.txt` - Python dependencies
- `resource_catalog.csv` - generated when exported from the app or CLI
- `cei_twin_log.csv` - generated after user interactions
- `recommender_stats.csv` - generated after recommendations are shown

## Recommended environment for the target Windows 11 device

Target device from the prompt:

- Windows 11
- Intel Core i5-1035G1
- 4 GB RAM

Because the RAM budget is small, use the app in **sample mode** for datasets:

- keep automatic download samples around **600-1200 images**
- keep batch size at **4 or 8**
- keep epochs at **1-3** for demo training
- start with fewer classes before trying larger emotion taxonomies

## Setup in VS Code

### 1) Create and activate a virtual environment

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```powershell
pip install -r requirements.txt
```

If TensorFlow is too heavy for your machine, the app still works with heuristic face analysis and the rest of the pipeline. You can comment TensorFlow out of `requirements.txt` for a lighter setup.

### 3) Optional API keys

Set these only if you want external integrations:

```powershell
$env:OPENAI_API_KEY="your_openai_key"
$env:SPOTIPY_CLIENT_ID="your_spotify_client_id"
$env:SPOTIPY_CLIENT_SECRET="your_spotify_client_secret"
```

### 4) Run the app

```powershell
streamlit run app.py
```

## Option A implemented: automatic dataset download and pre-splitting

The app includes an **Automatic dataset preparation** section that can:

- download a sampled subset from recent public Hugging Face emotion datasets
- detect likely image and label columns
- normalize labels
- pre-split into:
  - `dataset/train`
  - `dataset/val`
  - `dataset/test`
- save a `dataset_manifest.json` summary

### Included recent dataset options

- `FER2025` - auto-download enabled
- `EmoNet-Face-Big` - auto-download enabled
- `MER2024` - listed for documentation/planning, manual-only
- `MER2023` - listed for documentation/planning, manual-only

Notes:

- MER datasets may have additional access terms, so the app does not auto-download them.
- RAF-DB and CK+ are not from 2023-2026, so they are not auto-downloaded here, but the app supports **manual ZIP import** if you already have approved local copies.

## Option B implemented: expanded catalog

The app generates a structured resource catalog with more than 100 entries across:

- YouTube direct links
- YouTube search links
- Spotify search links
- YouTube Music search links

Each entry includes:

- mood
- title
- URL
- source
- type
- offline fallback guidance

You can also export the catalog directly from the command line:

```powershell
python app.py --export-catalog
```

## Training flow

Once a dataset exists under `dataset/`, the app can:

1. build a MobileNetV2 transfer-learning model
2. train it on `dataset/train`
3. validate on `dataset/val`
4. test/evaluate on `dataset/test` when available
5. save:
   - `models/mobile_transfer.keras`
   - `models/mobile_transfer_metadata.json`

## Distribution recommendation

The most practical zero-cost installable experience is:

- host the Streamlit app
- open it on Android in Chrome and use **Add to Home Screen**
- open it on Windows 11 in Edge/Chrome and use **Install app**

This gives a PWA-like installable experience without paid wrapper APK tools.

## Current limitations

- the automatic downloader depends on Hugging Face dataset schemas being compatible with generic image/label detection
- training on 4 GB RAM hardware must stay small and sampled
- true native Android APK generation still requires a separate toolchain such as Kivy/Buildozer or Chaquopy
