# GuneeshDua Major Project

Cognitive Emotional Intelligence, reinforcement-learning-style recommendation logic, digital emotional twin logging, multimodal AI fusion and ethical AI monitoring.

## Generated deliverables

- `app.py`: complete single-file Python 3 Streamlit implementation using PyTorch, not TensorFlow/Keras.
- `requirements.txt`: dependencies for execution in VS Code.
- `docs/GGSIPU_Major_Project_Report.docx`: downloadable GGSIPU Major Project - Dissertation report.
- `docs/GGSIPU_Major_Project_Report.html`: printable A4 HTML version.
- `docs/GGSIPU_Major_Project_Report.md`: editable markdown source.
- `docs/GGSIPU_Major_Project_Deliverables.zip`: downloadable bundle containing the report and code.
- `docs/download_reference.md`: local download reference links.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
$env:HF_TOKEN="your_hugging_face_token"
streamlit run app.py
```

Keep Hugging Face tokens in environment variables only. Do not hardcode API tokens in source files.
