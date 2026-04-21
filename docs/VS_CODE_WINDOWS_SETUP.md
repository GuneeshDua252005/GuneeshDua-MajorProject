# VS Code Windows 11 setup guide

This guide explains how to run the project on a Windows 11 laptop with:

- Intel Core i5
- 8 GB RAM
- Python 3.11
- VS Code

The project is designed to work in three modes:

1. Basic mode without TensorFlow training
2. Full free-AI mode with Hugging Face token
3. Full local CNN training mode with TensorFlow CPU

## 1. Project folder structure

Create or keep the project folder with the following structure:

```text
GuneeshDua-MajorProject/
|
|-- app.py
|-- requirements.txt
|-- .gitignore
|-- README.md
|
|-- docs/
|   |-- VS_CODE_WINDOWS_SETUP.md
|   |-- RESEARCH_PAPER_IEEE.md
|   |-- LITERATURE_MATRIX.md
|   |-- VIVA_GUIDE.md
|   `-- APK_CONVERSION_GUIDE.md
|
|-- dataset/
|   |-- train/
|   |-- val/
|   |-- test/
|   `-- dataset_manifest.json
|
|-- models/
|   |-- efficientnetv2_emotion.keras
|   `-- efficientnetv2_emotion_metadata.json
|
|-- cei_twin_log.csv
|-- recommender_stats.csv
`-- resource_catalog.csv
```

Important note:

- `dataset/`, `models/`, and the CSV files are generated automatically by the app.
- You do not need to create them manually before first run.

## 2. Install required software

Install the following on Windows 11:

### A. Python 3.11

Recommended:

- Download Python 3.11 from the official Python website
- During installation, select:
  - Add Python to PATH
  - Install for all users if available

### B. VS Code

Install:

- Visual Studio Code
- Python extension by Microsoft

### C. Optional but recommended

- Git for Windows
- Microsoft Visual C++ Redistributable

The Visual C++ redistributable is useful if TensorFlow CPU fails with DLL errors.

## 3. Open terminal in VS Code

Open the folder in VS Code, then open:

- Terminal -> New Terminal

Use PowerShell or Command Prompt.

## 4. Create a virtual environment

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

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate again.

## 5. Upgrade pip

```powershell
python -m pip install --upgrade pip
```

## 6. Install project dependencies

Install the default requirements:

```powershell
pip install -r requirements.txt
```

This installs:

- Streamlit
- NumPy
- Pandas
- Pillow
- scikit-learn
- Matplotlib
- Seaborn
- Hugging Face Hub client
- Datasets library

These are enough for:

- Streamlit UI
- text, emoji, and voice workflow
- Hugging Face API integration
- resource catalog export
- digital emotional twin logging
- recommender logic

## 7. Optional TensorFlow installation for CNN training

TensorFlow is intentionally not forced into `requirements.txt` because:

- it is large
- some student laptops fail during installation
- the app can still run without it

If you want CNN training and Grad-CAM from a locally trained model, install:

```powershell
pip install tensorflow-cpu
```

Then verify:

```powershell
python -c "import tensorflow as tf; print(tf.__version__)"
```

If TensorFlow fails on Windows:

1. Ensure Python is 64-bit
2. Ensure Python version is 3.11
3. Install Microsoft Visual C++ Redistributable
4. Upgrade pip
5. Retry `pip install tensorflow-cpu`

If TensorFlow still does not work, the app remains usable without local training.

## 8. Optional Hugging Face free token

To unlock free chat, speech-to-text, and remote image model usage:

1. Create a free account on Hugging Face
2. Go to Settings -> Access Tokens
3. Create a new token
4. Copy it

You can use it in two ways:

### Method A: Paste in the sidebar

Run the app and paste the token into:

- `Hugging Face access token (optional)`

### Method B: Set environment variable in terminal

PowerShell:

```powershell
$env:HF_TOKEN="your_hugging_face_token"
```

Command Prompt:

```cmd
set HF_TOKEN=your_hugging_face_token
```

## 9. Run the application

```powershell
streamlit run app.py
```

The terminal will print a local URL similar to:

```text
http://localhost:8501
```

Open it in the browser.

## 10. What the app can do

The project is a single-file Streamlit application that supports:

- face image upload or camera capture
- emoji mood input
- free-text emotional context
- optional voice-to-text using Hugging Face
- multimodal emotion fusion
- digital emotional twin CSV logging
- ethical AI monitoring
- history-aware resource recommendation
- free Hugging Face chatbot integration
- optional local EfficientNetV2 training
- optional Grad-CAM generation
- CSV resource catalog export

## 11. How to prepare a sampled dataset

The project can auto-prepare a small dataset for low-RAM training.

From terminal:

```powershell
python app.py --prepare-dataset --dataset-source FER2013 --sample-size 900
```

This creates:

- `dataset/train`
- `dataset/val`
- `dataset/test`
- `dataset/dataset_manifest.json`

Recommended for 8 GB RAM:

- sample size: 600 to 1200 images
- batch size: 4 or 8
- epochs: 1 to 3

## 12. How to export the 100+ item resource catalog

```powershell
python app.py --export-catalog
```

This generates:

- `resource_catalog.csv`

## 13. How to train the local CNN

Open the app and go to:

- `Training and Grad-CAM`

Then:

1. Prepare dataset
2. Select batch size 4 or 8
3. Select epochs 1 to 3
4. Click `Train local emotion model`

Generated files:

- `models/efficientnetv2_emotion.keras`
- `models/efficientnetv2_emotion_metadata.json`

## 14. How to generate Grad-CAM

After local training:

1. Open the `Training and Grad-CAM` tab
2. Upload an image or reuse the latest analyzed face image
3. Click `Generate Grad-CAM overlay`

The app will show:

- original image
- Grad-CAM overlay image

## 15. If you want completely free execution

Use this mode:

1. Install requirements
2. Do not install paid APIs
3. Use Hugging Face free token
4. If TensorFlow is too heavy, skip TensorFlow
5. Run app with:

```powershell
streamlit run app.py
```

In this mode:

- chatbot works through free Hugging Face models
- ASR can work through free Hugging Face inference
- face analysis works through:
  - local trained model if present
  - Hugging Face image model if token is present
  - heuristic fallback otherwise

## 16. Which files are mandatory

Mandatory source files:

- `app.py`
- `requirements.txt`
- `.gitignore`

Mandatory documentation files for project submission:

- `README.md`
- `docs/VS_CODE_WINDOWS_SETUP.md`
- `docs/RESEARCH_PAPER_IEEE.md`
- `docs/LITERATURE_MATRIX.md`
- `docs/VIVA_GUIDE.md`
- `docs/APK_CONVERSION_GUIDE.md`

Automatically generated files:

- `resource_catalog.csv`
- `cei_twin_log.csv`
- `recommender_stats.csv`
- `dataset/dataset_manifest.json`
- `models/efficientnetv2_emotion.keras`
- `models/efficientnetv2_emotion_metadata.json`

## 17. Recommended viva demonstration flow

During final presentation:

1. Start app
2. Enter username and session key
3. Paste Hugging Face token
4. Upload face image
5. Enter emotional context text
6. Select emoji
7. Fuse emotion
8. Show ethical warnings
9. Show digital emotional twin logging
10. Open adaptive recommendations
11. Ask chatbot:
   - Why EfficientNetV2?
   - What is Grad-CAM?
   - What are research gaps?
   - What is the future scope?

## 18. Important project note

The app is a major-project prototype and decision-support tool.

It should be presented as:

- an adaptive emotion-aware lifestyle support system
- a multimodal explainable AI prototype
- a digital emotional twin framework

It should not be presented as:

- a medical diagnosis system
- a psychiatric diagnosis engine
- a perfect emotion truth detector
