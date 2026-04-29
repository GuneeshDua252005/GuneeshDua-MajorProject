"""
Emotionally Intelligent Animated Mascot Chatbot

Run from VS Code or PowerShell with:
    streamlit run app.py

Place your trained Keras/TensorFlow model beside this file as `model.h5`, or
set the CEI_MODEL_PATH environment variable to the full model file path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Keep TensorFlow startup noise low before TensorFlow is imported lazily.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import streamlit as st
from PIL import Image, ImageOps


APP_TITLE = "Emotionally Intelligent Animated Mascot Chatbot"
APP_SUBTITLE = (
    "A Streamlit app with safe TensorFlow/Keras model loading, image emotion "
    "prediction, empathetic chatbot responses, and a demo mode when no model is present."
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_NAME = "model.h5"
DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

EMOTION_STYLES: Dict[str, Dict[str, str]] = {
    "angry": {"face": ">:(", "color": "#ef4444", "advice": "Pause, breathe slowly, and name what triggered the frustration."},
    "disgust": {"face": ":-/", "color": "#84cc16", "advice": "Step away from the stimulus and reset your attention with grounding."},
    "fear": {"face": ":-O", "color": "#8b5cf6", "advice": "Focus on what is controllable and take one small safe action."},
    "happy": {"face": ":-)", "color": "#f59e0b", "advice": "Capture what helped and share the positive energy."},
    "neutral": {"face": ":-|", "color": "#64748b", "advice": "Check in with yourself and choose the next intentional step."},
    "sad": {"face": ":-(", "color": "#3b82f6", "advice": "Be gentle with yourself and reach out to someone supportive."},
    "surprise": {"face": ":-o", "color": "#06b6d4", "advice": "Take a moment to understand the new information before reacting."},
}


@dataclass(frozen=True)
class ModelStatus:
    model: Optional[object]
    path: Optional[Path]
    labels: List[str]
    message: str
    ready: bool


def page_setup() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon=":robot_face:", layout="wide")
    st.markdown(
        """
        <style>
        .mascot-card {
            border-radius: 24px;
            padding: 1.2rem;
            background: linear-gradient(135deg, #111827 0%, #1f2937 48%, #0f172a 100%);
            color: white;
            box-shadow: 0 18px 50px rgba(15, 23, 42, 0.25);
        }
        .status-box {
            border-radius: 16px;
            padding: 1rem;
            border: 1px solid rgba(148, 163, 184, 0.35);
            background: rgba(248, 250, 252, 0.8);
        }
        .big-face {
            font-size: 5rem;
            line-height: 1;
            text-align: center;
            animation: float 2.4s ease-in-out infinite;
        }
        @keyframes float {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-8px); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def candidate_model_paths() -> List[Path]:
    paths: List[Path] = []
    env_path = os.getenv("CEI_MODEL_PATH", "").strip()
    if env_path:
        paths.append(Path(env_path).expanduser())

    paths.extend(
        [
            BASE_DIR / DEFAULT_MODEL_NAME,
            BASE_DIR / "models" / DEFAULT_MODEL_NAME,
            BASE_DIR / "model.keras",
            BASE_DIR / "models" / "model.keras",
            BASE_DIR / "models" / "emotion_model.keras",
        ]
    )

    seen = set()
    unique_paths: List[Path] = []
    for path in paths:
        normalized = str(path.resolve()) if path.exists() else str(path)
        if normalized not in seen:
            unique_paths.append(path)
            seen.add(normalized)
    return unique_paths


def find_existing_model_path() -> Optional[Path]:
    for path in candidate_model_paths():
        if path.is_file():
            return path
    return None


def load_labels(model_path: Optional[Path]) -> List[str]:
    search_dirs = [BASE_DIR]
    if model_path is not None:
        search_dirs.insert(0, model_path.parent)

    for directory in search_dirs:
        for name in ("labels.txt", "classes.txt", "class_names.txt"):
            label_file = directory / name
            if label_file.is_file():
                labels = [line.strip() for line in label_file.read_text(encoding="utf-8").splitlines() if line.strip()]
                if labels:
                    return labels
    return DEFAULT_LABELS.copy()


@st.cache_resource(show_spinner=False)
def import_tensorflow():
    try:
        import tensorflow as tf  # type: ignore

        return tf, None
    except Exception as exc:  # pragma: no cover - depends on local environment
        return None, exc


@st.cache_resource(show_spinner="Checking TensorFlow model...")
def safe_load_model(model_path_text: str) -> ModelStatus:
    model_path = Path(model_path_text) if model_path_text else None
    labels = load_labels(model_path)

    if model_path is None:
        searched = "\n".join(f"- {path}" for path in candidate_model_paths())
        return ModelStatus(
            model=None,
            path=None,
            labels=labels,
            ready=False,
            message=(
                "No model file was found. The app is still running in demo mode.\n\n"
                "Put your trained model at one of these locations or set CEI_MODEL_PATH:\n"
                f"{searched}"
            ),
        )

    tf, import_error = import_tensorflow()
    if tf is None:
        return ModelStatus(
            model=None,
            path=model_path,
            labels=labels,
            ready=False,
            message=f"TensorFlow/Keras could not be imported: {import_error}",
        )

    try:
        model = tf.keras.models.load_model(str(model_path), compile=False)
    except FileNotFoundError:
        return ModelStatus(
            model=None,
            path=model_path,
            labels=labels,
            ready=False,
            message=f"Model file not found at: {model_path}",
        )
    except OSError as exc:
        return ModelStatus(
            model=None,
            path=model_path,
            labels=labels,
            ready=False,
            message=f"Unable to open model file '{model_path}': {exc}",
        )
    except Exception as exc:  # pragma: no cover - depends on saved model format
        return ModelStatus(
            model=None,
            path=model_path,
            labels=labels,
            ready=False,
            message=f"Model file exists, but Keras could not load it: {exc}",
        )

    return ModelStatus(
        model=model,
        path=model_path,
        labels=labels,
        ready=True,
        message=f"Loaded model successfully from: {model_path}",
    )


def get_model_status() -> ModelStatus:
    model_path = find_existing_model_path()
    return safe_load_model(str(model_path) if model_path else "")


def preprocess_image(image: Image.Image, target_size: Tuple[int, int]) -> np.ndarray:
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize(target_size)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def predict_emotion(image: Image.Image, status: ModelStatus) -> Tuple[str, float, List[Tuple[str, float]]]:
    if not status.ready or status.model is None:
        raise RuntimeError("Prediction requires a loaded model.")

    input_shape = getattr(status.model, "input_shape", None)
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    target_size = DEFAULT_IMAGE_SIZE
    if input_shape and len(input_shape) >= 3 and input_shape[1] and input_shape[2]:
        target_size = (int(input_shape[2]), int(input_shape[1]))

    batch = preprocess_image(image, target_size)
    predictions = status.model.predict(batch, verbose=0)
    scores = np.asarray(predictions[0], dtype=np.float64)

    if scores.ndim != 1:
        scores = scores.reshape(-1)
    if scores.sum() > 0 and not np.isclose(scores.sum(), 1.0):
        scores = scores / scores.sum()

    labels = status.labels
    if len(labels) < len(scores):
        labels = labels + [f"class_{idx}" for idx in range(len(labels), len(scores))]
    elif len(labels) > len(scores):
        labels = labels[: len(scores)]

    best_idx = int(np.argmax(scores))
    ranked = sorted(zip(labels, scores.tolist()), key=lambda item: item[1], reverse=True)
    return labels[best_idx], float(scores[best_idx]), ranked


def demo_emotion_from_text(text: str) -> str:
    lowered = text.lower()
    keyword_map = {
        "happy": ["happy", "great", "good", "joy", "excited", "love", "awesome"],
        "sad": ["sad", "down", "cry", "lonely", "lost", "upset", "hurt"],
        "angry": ["angry", "mad", "furious", "annoyed", "irritated"],
        "fear": ["fear", "scared", "afraid", "anxious", "worried", "panic"],
        "surprise": ["surprised", "shock", "unexpected", "wow", "amazed"],
        "disgust": ["disgust", "gross", "awful", "nasty"],
    }
    for emotion, keywords in keyword_map.items():
        if any(keyword in lowered for keyword in keywords):
            return emotion
    return "neutral"


def response_for_emotion(emotion: str, user_text: str = "") -> str:
    style = EMOTION_STYLES.get(emotion, EMOTION_STYLES["neutral"])
    prefix = f"{style['face']} I sense **{emotion}**."
    if user_text.strip():
        prefix += " Thank you for sharing that with me."
    return f"{prefix} {style['advice']}"


def render_mascot(emotion: str, confidence: Optional[float] = None) -> None:
    style = EMOTION_STYLES.get(emotion, EMOTION_STYLES["neutral"])
    confidence_text = f"<p>Confidence: <b>{confidence:.1%}</b></p>" if confidence is not None else ""
    st.markdown(
        f"""
        <div class="mascot-card">
            <div class="big-face">{style['face']}</div>
            <h3 style="text-align:center; color:{style['color']}; margin-bottom:0;">{emotion.title()} Mascot</h3>
            {confidence_text}
            <p>{style['advice']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(status: ModelStatus) -> None:
    with st.sidebar:
        st.header("Model setup")
        if status.ready:
            st.success(status.message)
        else:
            st.warning(status.message)

        st.markdown("### How to fix `FileNotFoundError`")
        st.code(
            "\n".join(
                [
                    "# Option 1: put the file next to app.py",
                    "Major_AI_Project/",
                    "  app.py",
                    "  model.h5",
                    "",
                    "# Option 2: point to the real model path in PowerShell",
                    '$env:CEI_MODEL_PATH="C:\\Users\\Meenu\\OneDrive\\Desktop\\Major_AI_Project\\model.h5"',
                    "streamlit run app.py",
                ]
            ),
            language="powershell",
        )

        st.markdown("### Paths checked")
        for path in candidate_model_paths():
            icon = "[found]" if path.is_file() else "[missing]"
            st.caption(f"{icon} {path}")


def render_prediction_tab(status: ModelStatus) -> None:
    left, right = st.columns([1.1, 0.9])
    with left:
        st.subheader("Image emotion prediction")
        uploaded_file = st.file_uploader("Upload a face/image", type=["png", "jpg", "jpeg", "webp"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded image", use_container_width=True)
            if status.ready:
                try:
                    emotion, confidence, ranked = predict_emotion(image, status)
                    st.success(response_for_emotion(emotion))
                    st.bar_chart({label: score for label, score in ranked[: min(7, len(ranked))]})
                    st.session_state["last_emotion"] = emotion
                    st.session_state["last_confidence"] = confidence
                except Exception as exc:
                    st.error(f"Prediction failed safely without stopping Streamlit: {exc}")
            else:
                st.info("Add `model.h5` to enable real image prediction. The app remains usable in demo mode.")
        else:
            st.info("Upload an image after your model is available, or use the chatbot demo tab.")

    with right:
        emotion = st.session_state.get("last_emotion", "neutral")
        confidence = st.session_state.get("last_confidence")
        render_mascot(emotion, confidence)


def render_chat_tab() -> None:
    st.subheader("Empathetic chatbot demo")
    st.caption("This tab does not require `model.h5`; it keeps the local host alive while you set up the model.")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hi, I am your CEI mascot. Tell me how you feel today."}
        ]

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Type your feeling or situation...")
    if prompt:
        emotion = demo_emotion_from_text(prompt)
        answer = response_for_emotion(emotion, prompt)
        st.session_state["messages"].append({"role": "user", "content": prompt})
        st.session_state["messages"].append({"role": "assistant", "content": answer})
        st.session_state["last_emotion"] = emotion
        st.rerun()


def render_about_tab() -> None:
    st.subheader("Why the original error happened")
    st.markdown(
        """
        The traceback shows that Streamlit reached this startup code:

        ```python
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(BASE_DIR, "model.h5")
        model = load_model(model_path)
        ```

        If `model.h5` is not actually inside the same folder as the Python file,
        Keras raises `FileNotFoundError` and Streamlit stops before drawing the UI.

        This app fixes that by:

        1. using `Path(__file__).resolve().parent` for reliable relative paths;
        2. checking whether the model exists before calling `load_model`;
        3. importing TensorFlow only when a model file is present;
        4. catching file/model loading errors and showing them in the sidebar;
        5. keeping chatbot/demo features available even without the model.
        """
    )


def main() -> None:
    page_setup()
    status = get_model_status()
    render_sidebar(status)

    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)

    if status.ready:
        st.success("Model mode is active.")
    else:
        st.info("Demo mode is active because no loadable model was found. Streamlit will not crash.")

    tabs = st.tabs(["Predict", "Chatbot", "About / Fix"])
    with tabs[0]:
        render_prediction_tab(status)
    with tabs[1]:
        render_chat_tab()
    with tabs[2]:
        render_about_tab()


if __name__ == "__main__":
    main()
