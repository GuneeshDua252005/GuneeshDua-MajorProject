# GuneeshDua-MajorProject

Cognitive Emotional Intelligence, reinforcement-learning logic, digital emotional twin, and multimodal AI fusion along with ethical AI monitoring.

## Single-file application

This repository now includes a single-file CEI-ALOS implementation in `app.py`.

### Run

```bash
python3 -m pip install Flask
python3 app.py
```

Open `http://127.0.0.1:8000` in your browser.

### What is included

- Multi-modal emotion analysis using face image input, voice transcript, emoji, and text
- Reinforcement-style recommendation logic that avoids repeating recent content
- Digital Emotional Twin logging and analytics
- Spotify + OpenAI integration with built-in local API setup panel
- Structured 100+ resource catalog
- Installable PWA workflow for Android and Windows 11
- Optional CNN transfer-learning lab with evaluation metrics and Grad-CAM explainability

### API setup

You can either:

- set environment variables before running the app, or
- save keys from the in-app setup panel

Supported keys:

- `OPENAI_API_KEY`
- `SPOTIFY_CLIENT_ID`
- `SPOTIFY_CLIENT_SECRET`

### Optional CNN dependencies

The CNN training and Grad-CAM module is optional and requires:

```bash
python3 -m pip install numpy Pillow tensorflow scikit-learn
```
