# VS Code Execution Guide for Windows 11

## 1. Folder structure to create

After cloning or copying the project folder, keep this structure:

```text
GuneeshDua-MajorProject/
├── app.py
├── requirements.txt
├── README.md
├── docs/
├── paper/
├── dataset/
│   ├── train/
│   ├── val/
│   └── test/
└── models/
```

The `dataset/` and `models/` folders are created automatically by the application if they do not already exist.

## 2. Software to install

Install the following on the target Windows 11 laptop:

1. Python 3.11 or Python 3.12
2. VS Code
3. VS Code Python extension
4. Optional Git

## 3. Create a virtual environment

Open VS Code terminal in the project folder and run:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

## 4. Install required packages

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

If TensorFlow is too heavy for the device, you can temporarily comment out the `tensorflow` line in `requirements.txt` and run the app in heuristic mode.

## 5. Run the application

```powershell
streamlit run app.py
```

The browser should open automatically. If it does not, copy the local URL shown in the terminal into Chrome or Edge.

## 6. Free API setup

### Hugging Face token

1. Create a free account at https://huggingface.co/
2. Open **Settings -> Access Tokens**
3. Create a new read token
4. Paste it into the Streamlit sidebar when you want optional AI-generated viva guidance

### Spotify credentials

Spotify developer credentials can be created for free:

1. Visit https://developer.spotify.com/dashboard
2. Log in
3. Create an app
4. Copy client ID and client secret
5. Paste them in the sidebar

The project also runs without Spotify credentials by using search links.

## 7. Dataset preparation

The app supports two modes:

- automatic sampled download from supported Hugging Face mirrors
- manual image folder import

For low RAM:

- use 600 to 1200 images only
- batch size 4 or 8
- epochs 1 to 3

## 8. Training mode

If TensorFlow is installed:

1. Prepare sampled dataset from the app
2. Click the training button
3. Wait for the model to save under `models/`
4. Open the Results tab for metrics and Grad-CAM

## 9. What files get generated after execution

- `dataset/dataset_manifest.json`
- `models/efficientnetv2_cei.keras`
- `models/efficientnetv2_cei_metadata.json`
- `cei_twin_log.csv`
- `recommender_stats.csv`
- `resource_catalog.csv`

## 10. Common issues

### Problem: `python` command not found

Use:

```powershell
py -3 -m venv .venv
```

### Problem: TensorFlow install fails

Use heuristic mode without training, or install a compatible Python version such as 3.11.

### Problem: Streamlit is not recognized

Run:

```powershell
python -m streamlit run app.py
```

### Problem: app is slow

- reduce batch size
- reduce sample size
- skip training and use demo mode
- do not upload very large images

## 11. Free APK conversion trick

The easiest free path is:

1. Run or deploy the Streamlit app
2. Open the app in Chrome on Android
3. Use **Add to Home Screen**
4. For a shareable Android package, use PWABuilder on the deployed URL

This is not a full offline native APK, but it is the easiest free installable experience.
