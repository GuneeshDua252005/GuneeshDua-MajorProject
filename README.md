# GuneeshDua-MajorProject

Cognitive Emotional Intelligence, reinforcement-learning logic, Digital Emotional Twin logging, multi-modal AI fusion, and ethical AI monitoring for a final-year major project.

## Project overview

This repository contains a single-file Streamlit application in `app.py` for the project:

**Cognitive Emotion Intelligence & Adaptive Lifestyle System**

The application is designed as a major-project prototype that combines:

- multimodal emotion input and fusion
  - face image
  - emoji
  - voice-to-text fallback
  - free-text context
- modern CNN-based training with:
  - **EfficientNetV2**
  - **Global Average Pooling**
  - optional **Grad-CAM** explainability
- a lightweight adaptive recommendation layer
- Spotify search integration
- optional Hugging Face text-generation guidance
- Digital Emotional Twin CSV logging
- a structured 100+ item resource catalog with CSV export
- automatic sampled dataset preparation from public Hugging Face mirrors

This codebase stays intentionally simple:

- **single Python source file**
- no separate frontend/backend
- VS Code friendly
- Windows 11 friendly
- low-RAM demo mode supported

## What is in this repository

- `app.py` - main Streamlit application
- `requirements.txt` - required Python dependencies
- `.gitignore` - ignore rules for Python, datasets, models, and local runtime artifacts
- `docs/` - setup guides, free API instructions, APK/PWA notes, and viva help
- `paper/` - literature matrix, IEEE-style draft paper source, Word document output, and generator script

## Main features

### 1. Multimodal emotion fusion

The app accepts several low-cost input channels and fuses them into one current-state estimate:

- face image based emotion estimation
- emoji based self-report
- text based context analysis
- voice transcript fallback

When a trained TensorFlow model is not available, the app still works using heuristic emotion estimation, so the project remains demo-ready on weaker machines.

### 2. Modern CNN architecture

The training path uses:

- **EfficientNetV2B0**
- **GlobalAveragePooling2D**
- dropout and dense head
- confusion matrix
- weighted precision, recall, F1
- ROC-AUC where possible
- Grad-CAM based explainability

This replaces an older MobileNetV2-first approach with a newer and stronger baseline while keeping the system lightweight enough for student demonstration.

### 3. Ethical AI monitoring

The app surfaces deployment warnings such as:

- missing consent for logging
- low-confidence predictions
- low-modality reliability
- likely dataset imbalance

This keeps the project human-in-the-loop instead of presenting emotion detection as fully certain.

### 4. Digital Emotional Twin

The system stores structured state logs in CSV form, such as:

- timestamp
- user/demo id
- fused emotion
- confidence
- active modalities
- goal
- energy
- stress
- sleep hours

This creates a practical and auditable Digital Emotional Twin instead of only a one-time prediction.

### 5. Recommendation layer

The project includes:

- history-aware no-repetition recommendation behavior
- mood-aware resource suggestions
- Spotify search links or API results
- offline fallback routines

### 6. 100+ item resource catalog

The app automatically builds a large structured catalog and can export it by CLI or UI:

```bash
python app.py --export-catalog
```

## Recommended environment for the target Windows 11 device

Target device from the prompt:

- Windows 11
- Intel Core i5-1035G1
- 4 GB RAM

Recommended demo settings:

- use sampled datasets only
- keep sample size around **600-1200 images**
- keep batch size at **4 or 8**
- keep epochs at **1-3**
- start with fewer classes if the machine becomes slow
- use heuristic mode if TensorFlow installation is too heavy

## Setup in VS Code

### 1. Create and activate a virtual environment

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Command Prompt:

```cmd
python -m venv .venv
.\.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

## Free API guidance

### Hugging Face

Hugging Face is the recommended free API option here because:

- account creation is free
- token creation is free
- many text generation models have free or low-cost inference options
- it integrates well with Python

The app expects a **Hugging Face access token**, not a raw username/password login flow.

Important:

- modern providers do **not** securely support generating production API access directly from username and password inside student code
- the correct path is: create account -> create token -> paste token into app or environment variable

### Spotify

Spotify developer credentials are free to create from the Spotify developer dashboard, but still optional for this project.

If no credentials are supplied, the app falls back to search links instead of authenticated API responses.

### Why not OpenAI by default?

Because the user specifically asked for a free path. This project therefore defaults to:

- local heuristics
- optional Hugging Face token based inference
- optional Spotify search/API support

## Dataset preparation

The app contains an automatic dataset preparation section that can:

- download a sampled subset from public Hugging Face dataset mirrors
- detect likely image and label columns
- normalize labels
- pre-split into:
  - `dataset/train`
  - `dataset/val`
  - `dataset/test`
- save `dataset/dataset_manifest.json`

Current public mirror options in the app:

- `Jeneral/fer-2013`
- `AutumnQiu/fer2013`
- `UniqueData/facial-emotion-recognition-dataset`

## Training flow

Once the dataset exists under `dataset/`, the app can:

1. build an EfficientNetV2 + GAP model
2. train on `dataset/train`
3. validate on `dataset/val`
4. test/evaluate on `dataset/test`
5. save:
   - `models/efficientnetv2_cei.keras`
   - `models/efficientnetv2_cei_metadata.json`

## Distribution and APK conversion

The easiest free installable route is:

1. run the app locally or host it
2. open it on Android in Chrome and use **Add to Home Screen**
3. open it on Windows in Edge/Chrome and use **Install app**

This gives a PWA-style install experience without paid APK builders.

More detail is included in:

- `docs/APK_CONVERSION_GUIDE.md`

## Research paper bundle

The repository also includes a complete paper package:

- literature matrix
- IEEE-style long-form draft
- Word `.docx` export
- regeneration script

See:

- `paper/README.md`
- `paper/Cognitive_Emotion_Intelligence_Adaptive_Lifestyle_System_IEEE_Paper.docx`

## Important honesty note

This repository is meant to strengthen a major project technically and academically. However:

- no software can honestly guarantee a patent
- no code can guarantee publication
- no repository can guarantee a specific SGPA

What it can do is improve:

- project quality
- technical depth
- explainability
- documentation
- viva readiness
- presentation quality

## Current limitations

- real-time face detection is simplified to keep the project single-file and lightweight
- automatic dataset download depends on the schema of public dataset mirrors
- TensorFlow may be heavy on 4 GB RAM machines
- the single-file architecture is ideal for a student major project and demo, but not yet a production microservice design
- true offline native Android APK generation would require a separate app stack such as Kivy or Flutter

## License / academic use

Use this repository responsibly for:

- academic demonstration
- major project prototyping
- paper drafting
- explainability and deployment experimentation

Always verify dataset licenses, citation rules, and institutional submission requirements before final university submission.
