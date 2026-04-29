# GuneeshDua-MajorProject
Cognitive Emotional Intelligence , Reinforcement learning logic , digital emotional twin and multi-model AI fusion along with ethical AI monitoring. 

## Fix: missing `model.h5` in Streamlit

If Streamlit fails with:

```text
FileNotFoundError: Unable to synchronously open file ... model.h5
```

the app is calling `load_model(...)` before confirming that the trained Keras
model file exists at that path. Put `model.h5` in the same folder as your
Streamlit Python file, or update the loading code in `CEIALOSfinalmajorproject.py`
to resolve the model path from the script location:

```python
from pathlib import Path
import os
import streamlit as st
from keras.models import load_model

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("CEI_MODEL_PATH", APP_DIR / "model.h5")).expanduser()

if not MODEL_PATH.is_file():
    st.error(
        "model.h5 was not found. Place it next to CEIALOSfinalmajorproject.py "
        "or set CEI_MODEL_PATH to the full model file path."
    )
    st.code(str(MODEL_PATH), language="text")
    st.stop()

model = load_model(str(MODEL_PATH), compile=False)
```

On Windows PowerShell, you can point the app to the real model file without
hard-coding a local path:

```powershell
$env:CEI_MODEL_PATH="C:\Users\Meenu\OneDrive\Desktop\Major_AI_Project\model.h5"
streamlit run CEIALOSfinalmajorproject.py
```
