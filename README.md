# GuneeshDua-MajorProject
Cognitive Emotional Intelligence , Reinforcement learning logic , digital emotional twin and multi-model AI fusion along with ethical AI monitoring. 

## Run the Streamlit app

```bash
pip install -r requirements.txt
streamlit run app.py
```

The project includes `.streamlit/config.toml` with `server.fileWatcherType = "none"`.
This avoids a known Streamlit/PyTorch shutdown traceback on Windows/Python 3.11:
`RuntimeError: Event loop is closed`.
