# Complete VS Code Execution Guide (Windows 11, Single-File CEI Project)

This guide explains exactly what to install, what files to keep in the folder, and how to run the project end-to-end on:

- Windows 11
- Intel Core i5-1035G1
- 4 GB RAM

The project is intentionally kept as a single Python source file (`app.py`) to satisfy major-project constraints.

---

## 1) Final folder structure (what files should exist)

Create or keep this structure:

```text
GuneeshDua-MajorProject/
|- app.py
|- requirements.txt
|- .gitignore
|- README.md
|- docs/
|  |- EXECUTION_GUIDE_WINDOWS11.md
|  |- IEEE_RESEARCH_PAPER_CEI.md
|  |- VIVA_QUESTIONS_AND_DEFENSE.md
|- deliverables/
|  |- CEI_Research_Paper_IEEE.docx
|- models/                       (auto-created after training)
|- dataset/                      (auto-created after dataset prep)
|- resource_catalog.csv          (auto-created or exported)
|- cei_twin_log.csv              (auto-created on runtime)
|- recommender_stats.csv         (auto-created on runtime)
```

`models/`, `dataset/`, and generated CSV files are runtime artifacts and are ignored by Git.

---

## 2) Install prerequisites on Windows 11

### A. Install Python
- Install Python 3.10+ (recommended 3.10 or 3.11 for stability with TensorFlow CPU).
- During installation, enable:
  - "Add Python to PATH"

Verify:

```powershell
python --version
pip --version
```

### B. Install VS Code
- Install Visual Studio Code.
- Recommended extensions:
  - Python (Microsoft)
  - Pylance
  - Jupyter (optional)
  - Markdown All in One (optional)

---

## 3) Create and activate virtual environment

Open VS Code terminal in project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks scripts:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

---

## 4) Install dependencies

```powershell
pip install -r requirements.txt
```

### If TensorFlow is heavy on your machine
For 4 GB RAM laptops, CPU-only TensorFlow can still work in small mode. If needed:
- keep sample dataset around 600-1200 images,
- use image size 64 or 96,
- keep batch size 4 or 8,
- keep epochs 1-3.

---

## 5) Run the single-file project

```powershell
streamlit run app.py
```

Streamlit will open a local browser URL (typically `http://localhost:8501`).

---

## 6) API integration setup (free-first)

## 6.1 Hugging Face (recommended free API path)

This project is fully usable with free Hugging Face token + local fallbacks.

Steps:
1. Create account: `https://huggingface.co/join`
2. Go to Settings -> Access Tokens
3. Create token with `Read` permission
4. Paste token in sidebar field inside app

Use cases in app:
- text emotion classification
- optional question generation
- optional speech-to-text (Whisper inference)

## 6.2 Spotify (optional)

Spotify API is optional and may have usage restrictions.
If needed:
1. Create Spotify Developer app.
2. Copy client ID and client secret.
3. Paste in sidebar fields.

If not provided, app still works with generated catalog links and non-Spotify sources.

## 6.3 OpenAI (optional, not required)

OpenAI is paid in most practical scenarios. For free-first execution, use Hugging Face and local fallback logic already implemented in this project.

---

## 7) Automatic dataset preparation in app

Open tab: `Automatic Dataset Preparation`

Recommended values for target laptop:
- sample size: `600-1200`
- image size: `96`

The app prepares:
- `dataset/train`
- `dataset/val`
- `dataset/test`
- `dataset/dataset_manifest.json`

Supported options in current build:
- FER2025 (auto sample mode)
- EmoNet-Face-Big (auto sample mode)
- MER2024 (manual only)
- MER2023 (manual only)

---

## 8) Training in app (CNN + GAP + Grad-CAM)

Open tab: `Train CNN + Grad-CAM`

Use either architecture:
- `Custom-CNN-GAP` (lighter, recommended for 4 GB RAM)
- `EfficientNetV2B0-GAP` (stronger but heavier)

Safe demo config:
- epochs: `2`
- batch size: `8`
- image size: `96`

Outputs after training:
- `models/mobile_transfer.keras`
- `models/mobile_transfer_metadata.json`

Metrics shown:
- precision
- recall
- F1-score
- confusion matrix
- ROC-AUC (when calculable)

Grad-CAM:
- upload an image in same tab
- click `Generate Grad-CAM`

---

## 9) RL-style recommender feedback loop

In runtime tab:
1. Generate recommendations.
2. Use feedback form:
   - Helpful (+1)
   - Not helpful (-1)
3. Model updates reward stats in:
   - `recommender_stats.csv`

This creates a practical reinforcement-learning style adaptation loop.

---

## 10) Exports and logs

The app maintains:
- `cei_twin_log.csv` (digital emotional twin sessions)
- `recommender_stats.csv` (reward and usage statistics)
- `resource_catalog.csv` (100+ structured resources)

CLI export command:

```powershell
python app.py --export-catalog
```

---

## 11) APK conversion trick (free)

Strictly free, practical path:
1. Host Streamlit app locally/cloud.
2. Open in Android Chrome.
3. Use `Add to Home Screen`.

This behaves like an installable web app (PWA-style).  
If a true APK binary is mandatory, use a free WebView wrapper pipeline (Kivy/Buildozer or Android Studio WebView shell) around the same backend endpoint.

---

## 12) Troubleshooting

## Problem: TensorFlow install fails
- Upgrade pip first.
- Use Python 3.10/3.11.
- Recreate virtual environment.

## Problem: Dataset download slow
- Reduce sample size (e.g., 500-700).
- Retry with better internet.

## Problem: Hugging Face API returns error
- verify token validity,
- verify model availability,
- disable HF checkbox and use local fallback.

## Problem: Low RAM crash during training
- lower image size to 64,
- set batch size to 4,
- set epochs to 1,
- close browser tabs/background apps.

---

## 13) Major-project checklist for final demo

- [ ] Streamlit app runs without crash
- [ ] Emotion fusion works from at least 2 modalities
- [ ] Recommendations generated without repetition
- [ ] Feedback updates recommender stats
- [ ] One training run completed successfully
- [ ] Grad-CAM visualization generated
- [ ] CSV logs created and exported
- [ ] Research paper and viva notes ready

