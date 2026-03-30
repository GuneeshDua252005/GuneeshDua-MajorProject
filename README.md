# Cognitive Emotion Intelligence & Adaptive Lifestyle System

This repository contains a single-file Streamlit project for a major project titled **Cognitive Emotion Intelligence & Adaptive Lifestyle System**. The application is designed to run from **one Python source file (`app.py`)** and combines:

- multi-modal emotion input and fusion
- CNN transfer learning using MobileNetV2
- Grad-CAM explainability
- adaptive no-repetition recommendation logic
- Digital Emotional Twin CSV logging
- structured lifestyle and media catalog export
- free emotional AI chat using Hugging Face or an offline fallback
- optional Spotify search integration
- practical Windows 11 and VS Code execution guidance

## Repository files

The main files created in this project folder are:

- `app.py` - complete single-file Streamlit application
- `requirements.txt` - Python libraries required for execution
- `.gitignore` - ignore file for Python cache, virtual environment, models, dataset, and generated CSV artifacts
- `README.md` - step-by-step setup and execution guide
- `docs/research_paper_ieee_draft.md` - IEEE-style research paper draft content

Generated at runtime when needed:

- `cei_twin_log.csv` - Digital Emotional Twin interaction log
- `recommender_stats.csv` - recommendation statistics and feedback history
- `resource_catalog.csv` - exported structured media/resource catalog
- `models/mobile_transfer.keras` - trained transfer-learning model
- `models/mobile_transfer_metadata.json` - saved model metadata
- `dataset/` - prepared train, val, and test image folders

## What the single-file application includes

### 1. Multi-modal emotion analysis

The application accepts:

- face image from camera or file upload
- emoji signal
- voice-to-text fallback as typed text
- free-text emotional context

It combines these signals using a weighted fusion rule and predicts a final mood from:

- happy
- sad
- calm
- energetic

### 2. CNN + Grad-CAM + training

The program supports:

- MobileNetV2 transfer learning
- dataset loading from `dataset/train`, `dataset/val`, and `dataset/test`
- evaluation metrics including precision, recall, F1-score, confusion matrix, and ROC-AUC where possible
- Grad-CAM heatmap generation for model explanation

### 3. Digital Emotional Twin

Every analysis session can log:

- username
- detected face mood
- emoji mood
- voice mood
- text mood
- fused mood
- recommended item
- user feedback

This data is saved in `cei_twin_log.csv`.

### 4. Adaptive recommendation logic

The project includes:

- a 100+ item structured recommendation catalog
- no-repeat recommendation scoring
- simple reinforcement-learning-style exposure and feedback tracking
- optional Spotify search results

### 5. Emotional AI chat

The emotional AI chat is implemented in a free-friendly way:

- first choice: **Hugging Face Inference API** using a free user token
- optional: **OpenAI** if the user already has a paid API key
- fallback: **local offline emotional coach** if no external API is configured

## Why Hugging Face is suitable for free API integration

You asked why Hugging Face can be used and whether it is suitable.

Hugging Face is suitable here because:

- it allows free user token generation through a normal account
- it supports hosted inference for open-weight text-generation models
- it is simpler for student projects than building a full custom LLM backend
- it can be used only for text chat while the rest of the project runs locally

Limitations:

- free inference can be slower than paid services
- some models may be temporarily unavailable or rate-limited
- output quality depends on the selected hosted model

That is why this project keeps a **local fallback mode**, so the application still works even if Hugging Face is not configured.

## Spotify integration note

Spotify does not provide an unrestricted public no-auth music recommendation API for this exact use case. However, **Spotify developer credentials are free to create**, so the app supports optional search integration using:

- `SPOTIPY_CLIENT_ID`
- `SPOTIPY_CLIENT_SECRET`

If Spotify credentials are not set, the app still works with:

- YouTube links
- YouTube Music search links
- offline fallback guidance

## Target device compatibility

Recommended target machine from your requirements:

- Windows 11
- Intel Core i5
- 4 GB RAM

To keep the project stable on low RAM:

- keep dataset samples around **600 to 1200 images**
- use batch size **4 or 8**
- use **1 to 3 epochs** for demo training
- start with fewer classes before using larger emotion taxonomies
- close unnecessary applications while training

## Folder structure to create in VS Code

Open a folder in VS Code and keep this structure:

```text
project_folder/
│
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
├── docs/
│   └── research_paper_ieee_draft.md
├── dataset/
│   ├── train/
│   ├── val/
│   └── test/
└── models/
```

Notes:

- `dataset/` and `models/` can also be created automatically by the application when needed.
- If you use automatic dataset preparation, the app will populate `dataset/` itself.

## Complete step-by-step execution guide in VS Code

### Step 1. Install software

Install the following on Windows 11:

1. **Python 3.10 or 3.11**
2. **VS Code**
3. **Git** (recommended)

Optional but useful:

- Microsoft Python extension in VS Code

### Step 2. Open the project folder

In VS Code:

1. Open VS Code
2. Select **File -> Open Folder**
3. Choose the project folder

### Step 3. Create virtual environment

Open the VS Code terminal and run:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Step 4. Install required Python libraries

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5. Create the required files

The essential files already included in this repository are:

- `app.py`
- `requirements.txt`
- `.gitignore`
- `README.md`
- `docs/research_paper_ieee_draft.md`

If building from scratch manually, create these files and copy the repository content into them.

### Step 6. Optional free API token setup

#### Hugging Face free token

1. Create an account at `https://huggingface.co`
2. Go to **Settings -> Access Tokens**
3. Create a token
4. In PowerShell, set:

```powershell
$env:HF_TOKEN="your_hugging_face_token"
```

Optional model override:

```powershell
$env:HF_TEXT_MODEL="HuggingFaceH4/zephyr-7b-beta"
```

#### Optional Spotify developer credentials

1. Create an account at `https://developer.spotify.com`
2. Create an app
3. Copy client ID and client secret
4. Set them in PowerShell:

```powershell
$env:SPOTIPY_CLIENT_ID="your_spotify_client_id"
$env:SPOTIPY_CLIENT_SECRET="your_spotify_client_secret"
```

#### Optional OpenAI

This is **not required** for free execution. If you already have a key:

```powershell
$env:OPENAI_API_KEY="your_openai_api_key"
```

### Step 7. Run the application

```powershell
streamlit run app.py
```

If Streamlit does not open automatically, copy the local URL shown in the terminal into your browser.

## How to use the application

### Analyze and recommend

1. Enter a username
2. Select an emoji signal
3. Upload or capture a face image if available
4. Enter voice/text fallback text
5. Enter free-text context
6. Click **Analyze & Recommend**
7. View:
   - fused mood
   - recommendation
   - optional Grad-CAM overlay if a trained model exists

### Chat with Emotional AI

1. Run the mood analysis or directly enter a text prompt
2. Type a message in the chat box
3. Click **Send Chat**

The app will:

- use Hugging Face if `HF_TOKEN` exists
- otherwise use OpenAI if `OPENAI_API_KEY` exists
- otherwise use the local offline emotional coach

### Prepare a dataset

You can either:

- auto-download a sampled public dataset from Hugging Face
- upload your own ZIP file with `train/`, `val/`, and `test/` folders

Expected structure:

```text
dataset/
├── train/
│   ├── happy/
│   ├── sad/
│   └── ...
├── val/
│   ├── happy/
│   ├── sad/
│   └── ...
└── test/
    ├── happy/
    ├── sad/
    └── ...
```

### Train the CNN model

1. Prepare or import the dataset
2. Select batch size
3. Select epochs
4. Click **Train current dataset**

Output files:

- `models/mobile_transfer.keras`
- `models/mobile_transfer_metadata.json`

### Export the resource catalog

From the UI:

- use the download button

From CLI:

```powershell
python app.py --export-catalog
```

## What to write in `requirements.txt`

This project uses the following essential package list:

```text
streamlit
numpy
pandas
pillow
opencv-python-headless
tensorflow
scikit-learn
spotipy
openai
datasets
```

Notes:

- `openai` is optional at runtime but included because the app supports it if available.
- microphone capture is optional; for many Windows setups, typed voice-text fallback is more reliable than live microphone capture.

## Suggested major-project execution order

1. Create virtual environment
2. Install `requirements.txt`
3. Run the app first without training
4. Test:
   - multi-modal fusion
   - free emotional AI chat
   - resource catalog export
5. Prepare a small public dataset
6. Train MobileNetV2 for 1 to 3 epochs
7. Evaluate metrics
8. Test Grad-CAM visualization
9. Export logs and catalog for report screenshots

## Free APK conversion path

There is no true one-click Python-to-Android APK conversion for this Streamlit app without an additional wrapper or toolchain.

The practical free options are:

### Option A. Installable web app

1. Deploy the Streamlit app
2. Open it on Android in Chrome
3. Use **Add to Home Screen**

This gives a PWA-like experience and is the easiest free option.

### Option B. WebView wrapper

Create a simple Android WebView wrapper project in Android Studio that points to the deployed Streamlit URL.

### Option C. Rebuild as native mobile stack

If you need a true offline APK, port the project to:

- Kivy
- Buildozer
- Chaquopy

For most student major projects, **Option A or a WebView wrapper** is the most practical free route.

## Current limitations

- full CNN training on 4 GB RAM must stay small
- Hugging Face free inference may be slower than paid APIs
- Spotify search depends on optional developer credentials
- real APK generation requires an extra wrapper or separate mobile build flow
- webcam and microphone behavior can vary depending on the local browser and OS permissions

## Research paper deliverable

An IEEE-style research paper draft has been added at:

`docs/research_paper_ieee_draft.md`

That draft includes:

- abstract
- introduction
- literature review
- problem statement
- proposed work
- results and discussion framework
- conclusion and future scope
- IEEE-style references

## Important note on academic use

The paper draft is intended as a **project writing foundation**, not a final camera-ready journal submission. Before formal academic submission, you should:

- verify every reference against the actual publisher PDF
- replace placeholder result values with your own experimental findings
- add your university formatting requirements
- include your own screenshots, tables, and figures
