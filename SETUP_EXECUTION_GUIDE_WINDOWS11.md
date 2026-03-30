# Step-by-Step Complete Execution Guide (VS Code + Windows 11)

## 1. Create the folder and files

Create a folder, for example:

`C:\Users\<YourName>\Desktop\GuneeshDua-MajorProject`

Make sure the folder contains these files:

- `app.py`
- `requirements.txt`
- `.gitignore`
- `README.md`
- `SETUP_EXECUTION_GUIDE_WINDOWS11.md`
- `RESEARCH_PAPER_CEI_ADAPTIVE_LIFESTYLE_SYSTEM_IEEE.md`

Runtime-generated files/folders (created automatically when the app runs):

- `resource_catalog.csv`
- `cei_twin_log.csv`
- `recommender_stats.csv`
- `dataset\train`
- `dataset\val`
- `dataset\test`
- `dataset\dataset_manifest.json`
- `models\mobile_transfer.keras`
- `models\mobile_transfer_metadata.json`

## 2. Install software on Windows 11

Install:

1. Python 3.10+ (recommended 3.10 or 3.11)
2. VS Code
3. VS Code Python extension

Verify in PowerShell:

```powershell
python --version
pip --version
```

## 3. Open project in VS Code

1. Open VS Code
2. File -> Open Folder -> select your project folder
3. Open terminal in VS Code: Terminal -> New Terminal

## 4. Create virtual environment

Run in VS Code terminal (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 5. Install required packages

```powershell
pip install -r requirements.txt
```

This installs only essential packages required by the single-file application.

## 6. Optional API key setup

Set API keys only if you want external integrations:

```powershell
$env:OPENAI_API_KEY="your_openai_key"
$env:SPOTIPY_CLIENT_ID="your_spotify_client_id"
$env:SPOTIPY_CLIENT_SECRET="your_spotify_client_secret"
$env:HF_TOKEN="your_huggingface_token"
$env:GROQ_API_KEY="your_groq_api_key"
```

Important:

- Do not use account passwords in source code.
- APIs require token/key/secret-based authentication.
- Keep keys in environment variables, not hardcoded.

## 7. Run the application

```powershell
streamlit run app.py
```

The browser will open automatically with all modules:

- Emotion fusion (face + emoji + voice + text)
- Emotional AI chat
- Recommender
- Dataset preparation
- CNN training with MobileNetV2
- Grad-CAM

## 8. Run command line utilities (optional)

Export resource catalog:

```powershell
python app.py --export-catalog
```

Prepare Hugging Face sampled dataset:

```powershell
python app.py --prepare-dataset --hf-dataset-id nateraw/fer2013 --split train --sample-size 900
```

Prepare fully offline synthetic dataset:

```powershell
python app.py --prepare-dataset --synthetic --sample-size 900
```

## 9. RAM-safe settings for Intel i5 + 4 GB RAM

Use these settings inside the app:

- Sample size: 600 to 1200
- Batch size: 4 or 8
- Epochs: 1 to 3
- Start with fewer classes before larger datasets

## 10. Free API integration recommendation

Use in this order:

1. Local offline mode (completely free)
2. Hugging Face free token
3. Groq free API key (recommended if HF latency/queue is high)
4. OpenAI paid key (optional)

Why Hugging Face may feel slow sometimes:

- free-tier rate limits
- model cold starts
- inference queue delays

Best free upgrade:

- Groq OpenAI-compatible free API

## 11. Free APK conversion trick

No paid wrapper required:

1. Host Streamlit app on a URL.
2. On Android Chrome: "Add to Home Screen".
3. For full APK, create free Android Studio WebView wrapper.
4. On Windows Edge/Chrome: "Install app".

## 12. Troubleshooting

### TensorFlow installation issue

- Use Python 3.10/3.11.
- Update pip and retry.
- If system is low-resource, run non-training modules first.

### Speech transcription issue

- Ensure microphone permission and stable network for Google recognizer endpoint.
- Voice module is optional and has text fallback.

### API request errors

- Re-check tokens.
- Test with "Local (Free Offline)" chat mode first.
- Verify internet and API quota limits.

