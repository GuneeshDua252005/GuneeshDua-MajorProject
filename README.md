# GuneeshDua-MajorProject
Cognitive Emotional Intelligence , Reinforcement learning logic , digital emotional twin and multi-model AI fusion along with ethical AI monitoring. 

## Fixing `model.h5` FileNotFoundError in Streamlit

If `streamlit run CEIALOSfinalmajorproject.py` stops with:

```text
FileNotFoundError: Unable to open file ... Major_AI_Project\model.h5
```

the app is trying to load `model.h5` from the same folder as
`CEIALOSfinalmajorproject.py`, but the file is not there.

### Option 1: Put the model file in the expected folder

Copy or move your trained model file to:

```text
C:\Users\Meenu\OneDrive\Desktop\app.py\Major_AI_Project\model.h5
```

The file name must be exactly `model.h5`.

### Option 2: Update the app so Streamlit does not crash

In `CEIALOSfinalmajorproject.py`, replace the direct model load:

```python
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.h5")

model = load_model(model_path)
```

with:

```python
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
model_path = BASE_DIR / "model.h5"

@st.cache_resource
def load_emotion_model():
    if not model_path.exists():
        st.error(
            f"Missing model file: {model_path}. "
            "Place model.h5 in the same folder as CEIALOSfinalmajorproject.py "
            "or update model_path to the correct trained-model location."
        )
        st.stop()

    return load_model(str(model_path), compile=False)

model = load_emotion_model()
```

This keeps the Streamlit localhost open and shows a clear in-app error message
when `model.h5` is missing.
