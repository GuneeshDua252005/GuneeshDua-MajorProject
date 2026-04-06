# Cognitive Emotion Intelligence & Adaptive Lifestyle System

This document explains how to execute the single-file Python major project in VS Code on Windows 11, what to install, which folders are required, which files are created automatically, and how the free API path works.

## 1. Project overview

The repository contains a single Streamlit application in `app.py`. It combines:

- multimodal emotion input and fusion
  - face image
  - emoji
  - free-text context
  - optional voice transcript
- a modern CNN training path using EfficientNetV2-B0 with GlobalAveragePooling2D
- Grad-CAM explainability
- a local reinforcement-learning style recommender using lightweight bandit logic
- Digital Emotional Twin CSV logging
- ethical AI monitoring
- Hugging Face token-based question generation with local fallback templates
- a structured resource catalog and CSV export
- automatic sampled dataset preparation from Hugging Face datasets

The app is intentionally kept in one Python file so it is easier to present as a major project in VS Code without separate frontend and backend folders.

## 2. Target machine compatibility

Recommended target based on your requirement:

- Windows 11
- Intel Core i5
- 4 GB RAM laptop

To keep the project usable on low RAM:

- keep dataset sample size around 600 to 1200 images
- use image size 160, 192, or 224
- use batch size 4 or 8
- keep epochs to 1, 2, or 3 for demo training
- start with fewer classes when possible
- rely on Hugging Face token inference or local fallback questions instead of heavy local language models

## 3. Files in the repository

### Files you should keep in the folder

- `app.py`  
  Main Streamlit application and CLI utility

- `requirements.txt`  
  Python packages required for execution

- `.gitignore`  
  Prevents runtime artifacts and caches from being committed

- `README.md`  
  Repository title

- `PROJECT_GUIDE.md`  
  Complete setup and execution guide

- `RESEARCH_PAPER.md`  
  Full IEEE-style research paper draft content

- `VIVA_GUIDE.md`  
  Viva, HOD, and interviewer questions with answers

### Files created automatically when the program runs

- `resource_catalog.csv`
- `cei_twin_log.csv`
- `recommender_stats.csv`
- `models/efficientnetv2_emotion.keras`
- `models/efficientnetv2_metadata.json`
- `models/recommender_bandit_state.json`
- `dataset/dataset_manifest.json`

### Folders created automatically when needed

- `models/`
- `dataset/`
- `dataset/train/`
- `dataset/val/`
- `dataset/test/`
- `dataset/manual_import/`

## 4. Folder structure after bootstrap or runtime

```text
GuneeshDua-MajorProject/
│
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
├── PROJECT_GUIDE.md
├── RESEARCH_PAPER.md
├── VIVA_GUIDE.md
├── resource_catalog.csv
├── cei_twin_log.csv
├── recommender_stats.csv
│
├── dataset/
│   ├── dataset_manifest.json
│   ├── train/
│   ├── val/
│   ├── test/
│   └── manual_import/
│
└── models/
    ├── efficientnetv2_emotion.keras
    ├── efficientnetv2_metadata.json
    └── recommender_bandit_state.json
```

## 5. Software to install before execution

Install these on Windows 11:

1. Python 3.10 or Python 3.11  
   Download from the official Python site and make sure `python` and `pip` work in the terminal.

2. VS Code  
   Install the Python extension in VS Code.

3. Optional Git  
   Useful for version control and GitHub push.

No separate Node.js, React, backend server, Docker, or database is required for this project.

## 6. Create the project environment in VS Code

Open the project folder in VS Code and then use the integrated terminal.

### Step 1: Create a virtual environment

PowerShell:

```powershell
python -m venv .venv
```

### Step 2: Activate the virtual environment

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Command Prompt:

```cmd
.venv\Scripts\activate
```

### Step 3: Upgrade pip

```powershell
python -m pip install --upgrade pip
```

### Step 4: Install the required packages

```powershell
pip install -r requirements.txt
```

## 7. What to put in requirements.txt

This project uses the following essential packages:

```text
streamlit
numpy
tensorflow
datasets
huggingface_hub
scikit-learn
Pillow
matplotlib
SpeechRecognition
```

### Why these packages are used

- `streamlit` for the UI
- `numpy` for numerical operations and evaluation helpers
- `pillow` for image loading and processing
- `matplotlib` for Grad-CAM and training plots
- `scikit-learn` for split logic and evaluation metrics
- `tensorflow` for CNN training and Grad-CAM
- `datasets` for Hugging Face dataset loading
- `huggingface_hub` for dataset search and token-based inference access
- `SpeechRecognition` for free speech-to-text fallback

### Optional lighter mode

If TensorFlow is too heavy for the laptop, the app can still run with:

- heuristic image analysis
- text analysis
- emoji analysis
- voice transcript text analysis
- ethical AI monitoring
- recommendation engine
- Digital Emotional Twin logging
- resource catalog export

In that case, remove or comment out `tensorflow` and avoid the training tab.

## 8. Bootstrap the project

Before the first full run, initialize the runtime files:

```powershell
python app.py --bootstrap-project
```

This creates:

- starter CSV files
- dataset and model folders
- the exported resource catalog

## 9. Run the Streamlit application

```powershell
streamlit run app.py
```

Streamlit will open a local browser page, usually at:

```text
http://localhost:8501
```

## 10. How to use the application step by step

### A. Live mood assessment

1. Enter a local profile name.
2. Enter a local security key.  
   This does not create an online API account. It is only hashed locally to build a local user ID for personalization and CSV logs.
3. Optionally paste a Hugging Face token.
4. Upload a face image if available.
5. Select an emoji if relevant.
6. Add free-text context describing mood and situation.
7. Optionally upload a WAV, AIFF, AIF, or FLAC voice clip.
8. Click **Analyze mood and recommend**.

The app then:

- analyzes each available modality
- fuses them into a final emotion score
- generates short reflective questions
- creates a lifestyle action plan
- recommends music and resource links
- logs the event in the Digital Emotional Twin CSV

### B. Automatic dataset preparation

1. Go to the **Dataset and Training** tab.
2. Search Hugging Face datasets such as:
   - `facial emotion`
   - `FER2013`
   - `emotion face`
3. Copy a suitable dataset ID.
4. Enter the dataset ID.
5. Keep sample size between 600 and 1200 for low-RAM demo.
6. Click **Prepare sampled dataset**.

The app will:

- download the dataset
- detect image and label columns
- normalize labels
- split into `train`, `val`, and `test`
- save `dataset_manifest.json`

### C. Model training

1. After dataset preparation, stay in the training tab.
2. Select:
   - epochs: 1 to 3
   - batch size: 4 or 8
   - image size: 160, 192, or 224
   - learning rate
   - number of unfrozen layers
3. Click **Train and evaluate model**.

The app saves:

- `models/efficientnetv2_emotion.keras`
- `models/efficientnetv2_metadata.json`

Evaluation metrics shown:

- accuracy
- precision
- recall
- F1-score
- confusion matrix
- ROC-AUC where possible

### D. Grad-CAM explanation

After a model is trained and saved:

1. Return to the **Live Assessment** tab.
2. Upload a face image.
3. Run the analysis.
4. Click **Generate Grad-CAM explanation**.

This produces an overlay visualization showing which image regions influenced the CNN prediction.

## 11. Why Hugging Face is the free API option here

### What is used

The app supports a free Hugging Face token through the environment variable:

```powershell
$env:HF_TOKEN="your_huggingface_token"
```

or by pasting the token into the app sidebar.

### What is not used

The app does not use:

- OpenAI paid API by default
- Spotify paid API by default
- username/password based API login

### Why username and password should not be used directly

Hugging Face API access is token-based. Using account passwords directly in code is unsafe and unnecessary. The correct approach is:

1. Create a Hugging Face account.
2. Generate a user access token from the Hugging Face settings page.
3. Use that token in `HF_TOKEN`.

### Why Hugging Face is suitable here

- low friction for student projects
- free-tier access
- easy Python integration
- simple token-based authentication
- strong dataset ecosystem
- flexible model switching

### What if the token is not available

The app still works by using:

- local rule-based text emotion analysis
- local template-based reflective question generation
- search-link recommendations instead of paid APIs

## 12. Why MobileNetV2 was replaced

You requested replacing MobileNetV2 because it is older. This project now uses:

- EfficientNetV2-B0
- GlobalAveragePooling2D
- Grad-CAM on the final convolutional feature maps

### Why this is better for the project

- more modern CNN backbone
- good accuracy-efficiency balance
- still manageable for a student laptop in demo mode
- compatible with explainable AI workflows
- lightweight enough for small-sample transfer learning

## 13. Free music and recommendation strategy

The project avoids paid streaming dependencies by default.

Instead it uses:

- curated YouTube links
- YouTube search links
- Spotify public search links
- YouTube Music search links
- local reinforcement-learning style personalization based on previous feedback

This means:

- no downloadable MP3 generation
- no need for paid keys
- browser-based playback/search remains easy to demonstrate

## 14. Local reinforcement-learning logic

The recommender uses a small local bandit-style logic:

- user feedback updates item values
- repeat recommendations are reduced
- new or less-used items receive exploration bonus

This simulates adaptive personalization while staying lightweight and explainable.

## 15. Ethical AI monitoring in the app

The ethical AI monitor checks:

- single-modality dependence
- low confidence fusion
- strong disagreement across modalities
- heuristic fallback use instead of a trained face model

The app then displays:

- warning flags
- risk level
- safeguards

This makes the project stronger during viva because it shows responsible AI thinking instead of only prediction output.

## 16. Digital Emotional Twin

The Digital Emotional Twin concept in this project means:

- creating a structured log of mood-related interactions over time
- storing fused emotion, confidence, modality contributions, and recommendation context
- enabling history-aware personalization

The main log file is:

- `cei_twin_log.csv`

This can be shown in the viva as evidence of temporal emotional state tracking.

## 17. Free APK conversion trick

You asked for a free APK conversion option. The most practical free method is:

1. Deploy the Streamlit app to any reachable server or cloud host.
2. Open the app in Chrome on Android.
3. Use **Add to Home Screen**.

This gives an installable app-like shortcut at zero cost.

If an APK is strictly required:

1. Create a basic Android WebView shell project.
2. Point the WebView to the hosted Streamlit URL.
3. Build the APK using Android Studio.

This is free, but it is a wrapper around the hosted web app, not a native Python APK.

## 18. Recommended viva demonstration order

1. Show the project folder.
2. Show `requirements.txt`.
3. Run `python app.py --bootstrap-project`.
4. Run `streamlit run app.py`.
5. Show multimodal assessment with text and emoji first.
6. Show CSV logs being generated.
7. Show resource catalog export.
8. If possible, show a prepared dataset and a short 1-epoch demo training run.
9. Show Grad-CAM output.
10. Explain ethical AI monitoring and Digital Emotional Twin.

## 19. Important execution notes

- Use small sample sizes for low RAM laptops.
- Do not store account passwords in code.
- Use Hugging Face token authentication only.
- Keep training short during demos.
- Use the heuristic fallback mode if TensorFlow cannot be installed successfully on the machine.

## 20. Final recommended execution commands

### PowerShell sequence

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py --bootstrap-project
streamlit run app.py
```

### Optional Hugging Face token

```powershell
$env:HF_TOKEN="your_hf_token_here"
streamlit run app.py
```

## 21. Summary

This project is a complete single-file, VS Code-compatible Python major project that satisfies the main academic goals:

- multimodal emotion intelligence
- adaptive lifestyle guidance
- explainable AI with Grad-CAM
- a modern CNN with GAP
- free API-ready question generation
- reinforcement-style recommendation logic
- ethical AI monitoring
- Digital Emotional Twin logging
- low-cost, low-complexity deployment
