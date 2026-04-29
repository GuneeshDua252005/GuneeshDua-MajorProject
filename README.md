# GuneeshDua-MajorProject

Cognitive Emotion Intelligence and Adaptive Lifestyle System.

This repository contains a single-file Streamlit major project in `app.py`.
It uses PyTorch and EfficientNetV2-S, not TensorFlow or Keras.

## Files to keep in the project folder

- `app.py`
- `requirements.txt`
- `.gitignore`

The app creates these runtime files/folders automatically when needed:

- `models/`
- `dataset/`
- `cei_twin_log.csv`
- `recommender_stats.csv`
- `resource_catalog.csv`

## VS Code setup on Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Optional free Hugging Face API

Create a free Hugging Face account, generate a read token from Settings -> Access Tokens,
then set it before running the app:

```powershell
$env:HF_TOKEN="your_hugging_face_token"
streamlit run app.py
```

If no Hugging Face token is set, the chatbot still works using the local rule-based coach.
