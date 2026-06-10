# VS Code and Windows 11 Execution Guide

## 1. Recommended system

- OS: Windows 11
- CPU: Intel Core i5
- RAM: 4 GB minimum
- IDE: Visual Studio Code
- Python: 3.10 or 3.11 recommended

Python 3.12 works for parts of the project, but TensorFlow is more reliable on Windows with Python 3.10 or 3.11. If you want the CNN training and Grad-CAM features, install one of those versions.

## 2. Folder structure to create

After cloning or downloading the project, your folder should look like this:

```text
GuneeshDua-MajorProject/
|-- app.py
|-- requirements.txt
|-- README.md
|-- .gitignore
|-- docs/
|   |-- setup_guide.md
|   |-- research_paper.md
|   |-- literature_matrix.csv
|   `-- viva_questions.md
|-- assets/
|   `-- paper_flowchart.txt
|-- dataset/
|   |-- train/
|   |-- val/
|   `-- test/
`-- models/
```

The app will create `dataset/`, `models/`, `cei_twin_log.csv`, `resource_catalog.csv`, and `recommender_stats.csv` automatically if they are missing.

## 3. Install Python in Windows

1. Download Python 3.10 or 3.11 from the official Python website.
2. During installation, enable **Add Python to PATH**.
3. Verify installation in PowerShell or Command Prompt:

```powershell
python --version
```

## 4. Install VS Code and extensions

Install:

- Visual Studio Code
- Python extension by Microsoft
- Jupyter extension by Microsoft (optional)

## 5. Open the project in VS Code

1. Open VS Code.
2. Use **File -> Open Folder**.
3. Select the project folder.
4. Open the integrated terminal.

## 6. Create a virtual environment

In PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks scripts, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 7. Upgrade pip and install packages

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If TensorFlow installation is too heavy for the machine, you can still run the project in lightweight mode:

```powershell
pip uninstall tensorflow
```

The app will still support:

- text-based mood detection
- emoji self-check
- optional voice transcript fallback
- Hugging Face free API integration
- CSV logging
- adaptive recommendations

Only CNN training, full image inference, and Grad-CAM will be unavailable without TensorFlow.

## 8. Optional Hugging Face free API setup

This project uses Hugging Face as the recommended free API option.

### Steps

1. Create a free account at https://huggingface.co
2. Sign in.
3. Open **Settings -> Access Tokens**
4. Create a new token with read access.
5. Copy the token.

You do not need OpenAI or Spotify keys for this project.

### Use the token in Windows terminal

PowerShell:

```powershell
$env:HUGGINGFACE_API_TOKEN="your_token_here"
```

Command Prompt:

```cmd
set HUGGINGFACE_API_TOKEN=your_token_here
```

Or paste the token directly in the Streamlit sidebar field when the app opens.

## 9. Run the project

```powershell
streamlit run app.py
```

Then open the browser link shown in the terminal.

## 10. How to use the app

### A. Mood analysis

1. Open the **Mood Analysis** page.
2. Enter free-text context.
3. Select an emoji.
4. Optionally upload:
   - a face image
   - a voice note
5. Click **Analyse and generate support**.

The app will:

- detect mood using available inputs
- fuse modalities
- show confidence
- generate reflective questions
- provide resource links
- log the digital emotional twin

### B. Dataset preparation

Open **Dataset Tools**.

You have two ways:

#### Option 1: ZIP import

Create a ZIP file with folders like:

```text
emotion_dataset.zip
|-- happy/
|-- sad/
|-- angry/
|-- anxious/
|-- calm/
|-- focused/
|-- tired/
`-- neutral/
```

Put images into each folder, then upload the ZIP.

#### Option 2: Hugging Face dataset sample

Enter:

- dataset id
- split name
- image column if needed
- label column if needed

The app will download a small sample and split it into:

- `dataset/train`
- `dataset/val`
- `dataset/test`

## 11. Train the custom CNN

Open **Training and Grad-CAM**.

Recommended low-RAM settings:

- image size: 96 or 128
- batch size: 4 or 8
- epochs: 1 to 3

Then:

1. Click **Train CNN model**
2. Wait for training to complete
3. Review training history
4. Run evaluation
5. Upload a face image to generate Grad-CAM

Saved files:

- `models/cei_cnn_gap.keras`
- `models/cei_cnn_gap_metadata.json`

## 12. Export the resource catalog from terminal

```powershell
python app.py --export-catalog
```

This creates:

- `resource_catalog.csv`

## 13. Common files generated during execution

- `resource_catalog.csv`
- `cei_twin_log.csv`
- `recommender_stats.csv`
- `models/cei_cnn_gap.keras`
- `models/cei_cnn_gap_metadata.json`

## 14. Free APK conversion trick

This project is a Streamlit web app, so a practical free installable approach is:

1. Run the app locally or host it on a free platform.
2. Open it in Chrome on Android.
3. Use **Add to Home Screen**.

This gives a PWA-style install experience without paying for a wrapper.

If you need a native APK, a free academic workaround is:

1. Build a lightweight web wrapper using Android Studio WebView
2. Point it to the hosted Streamlit URL

This is simpler and more stable than converting a heavy Python runtime directly into an APK on low-end hardware.

## 15. Important notes for viva and deployment

- Do not claim the system performs medical diagnosis.
- Mention that Hugging Face is optional and used for free inference.
- Explain that the custom CNN with GAP was chosen to improve explainability and reduce parameters.
- Mention that the reinforcement learning idea is implemented as a lightweight reward-update loop from user feedback.
- Keep datasets small on 4 GB RAM machines.
