# Cognitive Emotion Intelligence & Adaptive Lifestyle System

Single-file major project implementation in Python (`app.py`) for:

- Cognitive Emotional Intelligence
- Multi-modal emotion fusion (face image + emoji + text + voice-to-text fallback via text box)
- Adaptive recommendation logic (RL-style Q-table update)
- Digital Emotional Twin logging
- Ethical AI monitoring
- CNN + GAP training and Grad-CAM explainability
- Free API integration path (Hugging Face token-based inference)

---

## 1) Project Objective

Build a practical, deployable, and explainable emotion-aware lifestyle assistant that:

1. Detects likely mood from multiple inputs.
2. Asks mindset-aware reflective questions.
3. Recommends mood-aligned resources (Spotify links / YouTube / fallback).
4. Learns from user feedback.
5. Logs interactions for analysis (Digital Emotional Twin).
6. Supports transparent decisions with Grad-CAM and ethical risk flags.

---

## 2) Repository Files

### Mandatory source files

- `app.py` -> complete single-source executable Streamlit application.
- `requirements.txt` -> essential Python dependencies.
- `.gitignore` -> runtime/model/dataset exclusions for clean git history.

### Runtime-generated files

- `cei_twin_log.csv` -> Digital Emotional Twin log.
- `recommender_stats.csv` -> recommendation usage stats.
- `resource_catalog.csv` -> resource catalog export.
- `models/cnn_gap_emotion.keras` -> trained model.
- `models/cnn_gap_emotion_metadata.json` -> training metadata.
- `dataset/dataset_manifest.json` -> automatic dataset preparation summary.

### Runtime-generated folders

- `dataset/train`, `dataset/val`, `dataset/test`
- `models/`

---

## 3) Exact `.gitignore` Rules Used

```gitignore
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
.pytest_cache/
.mypy_cache/
.venv/
venv/
env/
ENV/

.streamlit/
cei_twin_log.csv
recommender_stats.csv
resource_catalog.csv

models/
dataset/
```

---

## 4) `requirements.txt` (essential only)

```txt
streamlit>=1.39.0
numpy>=1.26.4
pandas>=2.2.2
pillow>=10.4.0
scikit-learn>=1.5.1
matplotlib>=3.9.0
requests>=2.32.3
python-dotenv>=1.0.1
datasets>=2.20.0
huggingface_hub>=0.24.6
spotipy>=2.24.0
tensorflow>=2.16.1
```

Notes:
- If TensorFlow is too heavy for your machine, keep the app in heuristic mode (no training) by removing/commenting TensorFlow in `requirements.txt`.
- For Windows 11 (4 GB RAM), use small sample sizes, low batch size (4/8), and 1-3 epochs.

---

## 5) Complete VS Code Setup (Windows 11, Intel i5, 4 GB RAM)

### Step 1: Open terminal in VS Code

Use PowerShell terminal in project root.

### Step 2: Create virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Step 3: Upgrade pip and install packages

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Optional environment variables (API keys)

#### Free Hugging Face token (recommended)

```powershell
$env:HF_TOKEN="your_hf_token"
```

Alternative name also supported:

```powershell
$env:HUGGINGFACEHUB_API_TOKEN="your_hf_token"
```

#### Optional Spotify API credentials (paid platform account required)

```powershell
$env:SPOTIPY_CLIENT_ID="your_spotify_client_id"
$env:SPOTIPY_CLIENT_SECRET="your_spotify_client_secret"
```

If Spotify keys are missing, app automatically falls back to free public search links.

### Step 5: Run application

```powershell
streamlit run app.py
```

### Step 6: Export catalog from CLI (without opening UI)

```powershell
python app.py --export-catalog
```

---

## 6) Free API Token Generation (Hugging Face)

1. Create account at `https://huggingface.co`.
2. Go to **Settings -> Access Tokens**.
3. Create token with at least **Read** permission.
4. Copy token and set environment variable:
   - `HF_TOKEN` or `HUGGINGFACEHUB_API_TOKEN`
5. Start app; mood-aware question generation will use HF inference API.

Why Hugging Face here:
- Easy token management.
- Free-tier access for many inference workloads.
- Good model ecosystem for NLP/emotion pipelines.

If HF rate-limits or model is unavailable:
- App automatically uses local fallback questions.
- Best no-cost backup option: local inference with Ollama (can be integrated later).

---

## 7) Major Features Implemented

1. **Multi-modal emotion fusion**
   - Text keyword scoring
   - Emoji prior mapping
   - Face image inference
   - Weighted fusion for final mood probability

2. **Free API integration**
   - Hugging Face inference endpoint for generating reflective questions
   - Optional Spotify integration with graceful fallback

3. **CNN + GAP architecture**
   - EfficientNetB0 backbone
   - Extra convolutional refinement layers
   - Global Average Pooling (GAP)
   - Softmax classifier

4. **Grad-CAM explainability**
   - Heatmap generated from `conv_last`
   - Overlay on input image for visual explanation

5. **Evaluation metrics**
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - Confusion matrix
   - ROC-AUC (when feasible)

6. **Automatic sampled dataset preparation**
   - Detects image + label columns
   - Creates train/val/test split
   - Writes `dataset_manifest.json`

7. **Digital Emotional Twin**
   - CSV logging of inferred mood, confidence, context, and strategy

8. **Ethical AI monitoring**
   - Low-confidence warnings
   - Modality completeness checks
   - Bias/privacy advisories

9. **Adaptive recommendation logic**
   - RL-style Q-table update from user reward feedback

10. **Resource catalog**
   - 100+ structured links with offline fallback guidance

---

## 8) APK Conversion Trick (Free Practical Method)

For free installable experience without paid wrappers:

1. Run or host Streamlit app.
2. Open app URL in Android Chrome.
3. Tap **Add to Home Screen**.
4. Use as PWA-like installed app shortcut.

For true native APK packaging, external toolchains are needed:
- Kivy + Buildozer
- Chaquopy with Android Studio

---

## 9) Limitations

- Free APIs can be rate-limited.
- Emotion classification quality depends on dataset diversity and annotation quality.
- 4 GB RAM requires reduced training scale.
- MER datasets may have additional access constraints (manual-only in this setup).

---

## 10) Research/Viva Support

Detailed IEEE-format draft content and viva guide are included in:

- `RESEARCH_PAPER_IEEE_DRAFT.md`
- `VIVA_QA_GUIDE.md`

These files cover:
- Abstract, Introduction, Literature Review, Proposed Method, Results, Conclusion
- Research gaps and how this project addresses them
- Explainability significance (Grad-CAM, GAP, ethical monitoring)
- Top viva questions and detailed answers

