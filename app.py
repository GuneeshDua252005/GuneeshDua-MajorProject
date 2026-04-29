"""
Streamlit app entrypoint with robust Keras model loading.

This version prevents startup crashes when `model.h5` is missing by:
1) checking candidate paths before loading,
2) handling loading exceptions gracefully, and
3) keeping the Streamlit UI alive with actionable error messages.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Iterable

import streamlit as st


# Configure TensorFlow logging behavior before importing TensorFlow/Keras.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# Optional: reduce oneDNN warning noise and numerical-difference warning message.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")


def _iter_candidate_paths(custom_path: str | None) -> Iterable[Path]:
    """Yield likely model file locations in priority order."""
    base_dir = Path(__file__).resolve().parent

    if custom_path:
        yield Path(custom_path).expanduser()

    env_path = os.getenv("MODEL_PATH", "").strip()
    if env_path:
        yield Path(env_path).expanduser()

    # Common local defaults
    yield base_dir / "model.h5"
    yield base_dir / "Major_AI_Project" / "model.h5"
    yield Path.cwd() / "model.h5"


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []

    for p in paths:
        # strict=False avoids raising if path does not exist.
        key = str(p.resolve(strict=False))
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def resolve_existing_model_path(custom_path: str | None) -> tuple[Path | None, list[Path]]:
    """Return first existing model path + the candidate paths checked."""
    candidates = _dedupe_paths(_iter_candidate_paths(custom_path))
    for path in candidates:
        if path.is_file():
            return path, candidates
    return None, candidates


@st.cache_resource(show_spinner=False)
def load_keras_model(model_path: str):
    """
    Load a Keras model from a resolved path.
    Cached to avoid repeated heavy reloads between reruns.
    """
    from keras.models import load_model

    return load_model(model_path, compile=False)


def _render_model_summary(model) -> None:
    buffer = io.StringIO()
    model.summary(print_fn=lambda line: buffer.write(line + "\n"))
    st.code(buffer.getvalue(), language="text")


def _attempt_model_load(custom_path: str | None):
    resolved_path, checked_paths = resolve_existing_model_path(custom_path)
    if resolved_path is None:
        checked = "\n".join(f"- {p}" for p in checked_paths)
        st.error(
            "Model file not found. Place `model.h5` in one of the paths below, "
            "or enter a valid absolute path."
        )
        st.code(checked, language="text")
        return None

    try:
        model = load_keras_model(str(resolved_path))
        st.success(f"Model loaded successfully from: {resolved_path}")
        return model
    except FileNotFoundError as exc:
        # Extra guard for race conditions where file disappears after check.
        st.error(f"Model path became unavailable while loading: {exc}")
    except OSError as exc:
        st.error(
            "Keras could not open the model file. The file may be corrupt or not a valid `.h5` model."
        )
        st.exception(exc)
    except Exception as exc:  # broad catch keeps Streamlit app alive
        st.error("Unexpected error while loading model.")
        st.exception(exc)

    return None


def main() -> None:
    st.set_page_config(page_title="Major AI Project", layout="wide")
    st.title("Emotionally Intelligent Animated Mascot Chatbot")
    st.caption("Stable startup version with safe model loading.")

    st.markdown("### Model Configuration")
    custom_path = st.text_input(
        "Model path (optional)",
        value="",
        placeholder=r"Example: C:\Users\Meenu\OneDrive\Desktop\Major_AI_Project\model.h5",
        help="Leave empty to auto-search common local paths.",
    ).strip()

    if "model" not in st.session_state:
        st.session_state.model = None

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Load / Reload Model", type="primary"):
            st.session_state.model = _attempt_model_load(custom_path)
    with col2:
        st.info(
            "Tip: keep `model.h5` in the same folder as `app.py` for the simplest setup."
        )

    # Automatic first load attempt (without custom path).
    if st.session_state.model is None and "autoload_done" not in st.session_state:
        st.session_state.model = _attempt_model_load(custom_path or None)
        st.session_state.autoload_done = True

    st.markdown("---")
    st.markdown("### Runtime Status")
    if st.session_state.model is None:
        st.warning(
            "App is running, but no model is loaded yet. "
            "This is intentional to prevent Streamlit from crashing."
        )
    else:
        st.success("Model is available and ready.")
        with st.expander("Model summary", expanded=False):
            _render_model_summary(st.session_state.model)


if __name__ == "__main__":
    main()
