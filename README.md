# GuneeshDua-MajorProject

Cognitive Emotion Intelligence, reinforcement learning logic, digital emotional twin, multi-model AI fusion, and ethical AI monitoring in a single Python major-project package.

## What is included

1. `cognitive_emotion_intelligence_system.py`
   - single-file Python application
   - text-based mood analysis
   - adaptive reflective question generation
   - lifestyle planning
   - optional free Hugging Face API integration
   - optional Spotify metadata integration
   - reinforcement-learning style feedback adaptation
   - training-ready EfficientNetV2-S and Grad-CAM support

2. `docs/research_paper.md`
   - complete IEEE-style research paper draft for the title:
     `Cognitive Emotion Intelligence and Adaptive Lifestyle System`

3. `docs/viva_questions.md`
   - important viva questions with detailed answers

4. `docs/project_tools_and_notes.md`
   - best tools for writing the paper, APK conversion notes, and project-positioning notes

5. `deliverables/Cognitive_Emotion_Intelligence_Adaptive_Lifestyle_System_IEEE_Paper.docx`
   - generated downloadable Word document

## Why this project is useful

This project extends older 2023-2025 emotion-system prototypes in five practical ways:

- it fuses emotion understanding with adaptive lifestyle support rather than only mood detection
- it keeps the full software in a single Python file for simple major-project demonstration
- it adds explainable AI through Grad-CAM
- it introduces a digital emotional twin for continuity across sessions
- it includes RL-style feedback adaptation instead of static one-time recommendations

## Quick start

### 1. Create a virtual environment

On Windows:

```bat
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the program

```bash
python3 cognitive_emotion_intelligence_system.py
```

If you are on Windows and `python3` is unavailable, use:

```bat
py cognitive_emotion_intelligence_system.py
```

## Free API integration

### Hugging Face

This project uses Hugging Face as the free alternative to paid proprietary text APIs.

1. Create a free account at `https://huggingface.co/join`
2. Open `Settings -> Access Tokens`
3. Create a token with `Read` permission
4. Set it in the terminal

Windows:

```bat
setx HF_TOKEN your_token_here
setx HF_LLM_MODEL google/flan-t5-base
```

Linux/macOS:

```bash
export HF_TOKEN=your_token_here
export HF_LLM_MODEL=google/flan-t5-base
```

If the token is not configured, the app still works with an offline fallback question generator.

### Spotify

Spotify is optional. The program does not download any mp3 files. It only fetches track metadata and links when credentials are available.

1. Create a free developer app at `https://developer.spotify.com/dashboard`
2. Copy the client ID and client secret
3. Set:

Windows:

```bat
setx SPOTIFY_CLIENT_ID your_client_id
setx SPOTIFY_CLIENT_SECRET your_client_secret
```

Linux/macOS:

```bash
export SPOTIFY_CLIENT_ID=your_client_id
export SPOTIFY_CLIENT_SECRET=your_client_secret
```

Without Spotify keys, the app uses the offline recommendation library included in the single Python file.

## Optional deep-learning workflow

The core text workflow runs with lightweight requirements. The image branch needs:

- torch
- torchvision
- pillow
- numpy

Expected FER-style dataset structure:

```text
data/
  fer/
    train/
      angry/
      disgust/
      fear/
      happy/
      neutral/
      sad/
      surprise/
    val/
      angry/
      disgust/
      fear/
      happy/
      neutral/
      sad/
      surprise/
```

The code uses EfficientNetV2-S with Global Average Pooling and supports Grad-CAM generation after training.

## Free APK conversion trick

The repository includes notes in `docs/project_tools_and_notes.md`.

Short version:

- keep the core logic in the existing single Python file
- add a minimal Kivy wrapper only if Android UI is needed
- build the APK with Buildozer using WSL, Ubuntu, or Google Colab

## Important note

This package is designed to help you prepare a strong academic submission, but no software can honestly guarantee a particular SGPA, publication placement, patent grant, or newspaper feature. Final outcomes depend on implementation quality, originality, institutional review, and evaluation standards.
