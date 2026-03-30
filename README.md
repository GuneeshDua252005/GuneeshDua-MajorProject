# GuneeshDua-MajorProject

Cognitive Emotional Intelligence, reinforcement-learning logic, Digital Emotional Twin logging, and multi-modal AI fusion with practical deployment guidance.

This repository contains a **single-file Streamlit application** (`app.py`) that demonstrates:

- Multi-modal emotion input and fusion:
  - Face image
  - Emoji
  - Voice-to-text fallback
  - Free-text context
- Transfer learning with MobileNetV2
- Evaluation metrics:
  - Precision
  - Recall
  - F1 score
  - Confusion matrix
  - ROC-AUC (where supported)
- Grad-CAM explainability
- History-aware no-repetition recommender
- Optional Spotify and OpenAI integrations
- Free-tier API options (Hugging Face, Groq, and offline local mode)
- Digital Emotional Twin CSV logging
- Structured 100+ item resource catalog with CSV export
- Automatic sampled dataset preparation (Hugging Face or synthetic fallback)

---

## Project structure

Essential files:

- `app.py` - single executable program for Streamlit UI, model flow, and API integrations
- `requirements.txt` - required Python dependencies
- `.gitignore` - pycache/runtime/model/dataset ignore rules
- `README.md` - execution and deployment guide

Runtime-generated files/folders:

- `resource_catalog.csv`
- `cei_twin_log.csv`
- `recommender_stats.csv`
- `dataset/` (`train`, `val`, `test`, `dataset_manifest.json`)
- `models/` (`mobile_transfer.keras`, `mobile_transfer_metadata.json`)

---

## Windows 11 + VS Code complete execution guide

Target machine profile from your requirement:

- Windows 11
- Intel Core i5-1035G1
- 4 GB RAM

### 1) Create project folder and open in VS Code

1. Create a folder, for example: `GuneeshDua-MajorProject`
2. Put these files inside:
   - `app.py`
   - `requirements.txt`
   - `.gitignore`
   - `README.md`
3. Open VS Code in this folder.

### 2) Open terminal and create virtual environment

PowerShell commands:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 3) Install dependencies

```powershell
pip install -r requirements.txt
```

If TensorFlow is too heavy on your machine, you can temporarily remove/comment `tensorflow-cpu` from `requirements.txt`. The app still works with heuristic emotion logic, recommender, chat, and logging modules.

### 4) Optional environment variables for APIs

PowerShell:

```powershell
$env:OPENAI_API_KEY="your_openai_key"
$env:SPOTIPY_CLIENT_ID="your_spotify_client_id"
$env:SPOTIPY_CLIENT_SECRET="your_spotify_client_secret"
$env:HF_TOKEN="your_huggingface_token"
$env:GROQ_API_KEY="your_groq_api_key"
```

Security rule:

- Use token/key/secret-based auth.
- Do not use account passwords directly in code.
- Do not hardcode keys in source files.

### 5) Run the program

```powershell
streamlit run app.py
```

### 6) Optional CLI utilities

Export 100+ catalog rows:

```powershell
python app.py --export-catalog
```

Prepare dataset directly from terminal:

```powershell
python app.py --prepare-dataset --hf-dataset-id nateraw/fer2013 --split train --sample-size 900
```

Or offline synthetic fallback:

```powershell
python app.py --prepare-dataset --synthetic --sample-size 900
```

---

## Recommended runtime settings for 4 GB RAM

- Dataset sample size: `600-1200`
- Batch size: `4` or `8`
- Epochs: `1-3` for demo
- Start with fewer classes first

---

## Dataset preparation support

Automatic section in app can:

- Download sampled subset from selected Hugging Face dataset
- Detect likely image and label columns
- Normalize labels
- Split into:
  - `dataset/train`
  - `dataset/val`
  - `dataset/test`
- Save `dataset/dataset_manifest.json`

Included options in app:

- FER2025 - auto-download enabled (mapped to practical FER source)
- EmoNet-Face-Big - auto-download enabled (mapped source option)
- MER2024 - manual only
- MER2023 - manual only
- Custom Hugging Face dataset ID
- Synthetic offline fallback

Notes:

- MER datasets may include additional access terms and are marked manual-only in this program.
- Manual dataset import is supported by placing class folders under `dataset/train`, `dataset/val`, and `dataset/test`.

---

## Free API integration recommendation

Your question included paid API constraints and asked for fully free options.

Practical free-first order:

1. **Local offline mode** (always free, no token)
2. **Hugging Face free token** (rate-limited, model queues possible)
3. **Groq free API key** (recommended upgrade path, OpenAI-compatible endpoint)
4. **OpenAI paid key** (optional)

Why Hugging Face may not be suitable in some cases:

- Free-tier rate limits
- cold start latency
- queue delays for popular hosted models

Best free upgrade for smoother response:

- Groq free API key for chat-completion compatible calls

---

## Spotify integration note

Spotify search integration is optional and credential-based. The app does not download MP3 files and only returns links/metadata.

---

## APK conversion trick (free)

For a free installable experience:

1. Host Streamlit app on a URL.
2. On Android Chrome: **Add to Home Screen** (PWA-like shortcut).
3. On Windows 11 Edge/Chrome: **Install app**.
4. Optional fully free wrapper method: create a lightweight Android WebView app using Android Studio.

This avoids paid APK wrapper services while keeping deployment simple.

---

## Research paper file

A detailed IEEE-style research paper draft is included at:

- `RESEARCH_PAPER_CEI_ADAPTIVE_LIFESTYLE_SYSTEM_IEEE.md`

---

## Disclaimer

This project is educational and research-oriented. It is not a clinical diagnostic tool. If user text indicates crisis/self-harm risk, direct emergency guidance should be prioritized immediately.
