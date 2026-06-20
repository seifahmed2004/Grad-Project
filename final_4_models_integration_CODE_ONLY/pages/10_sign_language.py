from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict

import streamlit as st


st.title("🤟 Sign Language to Text")
st.caption("Updated Landmark Transformer sign model integration with persistent input and output.")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIGN_INPUT_DIR = PROJECT_ROOT / "outputs" / "sign_inputs"
SIGN_INPUT_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_PATH_KEY = "sign_page_video_path"
VIDEO_NAME_KEY = "sign_page_video_name"
RESULT_KEY = "sign_page_last_result"
ERROR_KEY = "sign_page_last_error"
MODE_KEY = "sign_page_prediction_mode"

# Advanced settings keys
TOP_K_KEY = "sign_page_top_k"
THRESHOLD_KEY = "sign_page_threshold"
MIN_PAUSE_KEY = "sign_page_min_pause_sec"
MIN_SEGMENT_KEY = "sign_page_min_segment_sec"
CONFIDENCE_KEY = "sign_page_confidence_threshold"
DECODER_KEY = "sign_page_use_language_decoder"


def init_state() -> None:
    st.session_state.setdefault(VIDEO_PATH_KEY, "")
    st.session_state.setdefault(VIDEO_NAME_KEY, "")
    st.session_state.setdefault(RESULT_KEY, {})
    st.session_state.setdefault(ERROR_KEY, "")
    st.session_state.setdefault(MODE_KEY, "Single isolated sign")

    st.session_state.setdefault(TOP_K_KEY, 5)
    st.session_state.setdefault(THRESHOLD_KEY, 0.08)
    st.session_state.setdefault(MIN_PAUSE_KEY, 0.35)
    st.session_state.setdefault(MIN_SEGMENT_KEY, 0.30)
    st.session_state.setdefault(CONFIDENCE_KEY, 0.08)
    st.session_state.setdefault(DECODER_KEY, True)


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:24]


def save_video(data: bytes, filename: str) -> Path:
    suffix = Path(filename or "video.mp4").suffix.lower() or ".mp4"
    if suffix not in [".mp4", ".webm", ".avi", ".mov", ".mkv"]:
        suffix = ".mp4"

    path = SIGN_INPUT_DIR / f"sign_{hash_bytes(data)}{suffix}"
    if not path.exists():
        path.write_bytes(data)

    return path


def result_to_text(result: Any) -> str:
    if result is None:
        return ""

    if isinstance(result, str):
        return result.strip()

    if isinstance(result, dict):
        for key in [
            "sentence",
            "decoded_sentence",
            "text",
            "prediction",
            "label",
            "class_name",
            "predicted_label",
            "sign",
        ]:
            value = result.get(key)
            if value not in (None, ""):
                return str(value).strip()

        # Try common nested structures.
        for nested_key in ["result", "output", "best", "top_prediction"]:
            nested = result.get(nested_key)
            text = result_to_text(nested)
            if text:
                return text

        # Try first prediction in list.
        preds = result.get("predictions") or result.get("top_k") or result.get("top_predictions")
        if isinstance(preds, list) and preds:
            text = result_to_text(preds[0])
            if text:
                return text

    if isinstance(result, list) and result:
        return result_to_text(result[0])

    return str(result)


def predict_sign(video_path: Path, sentence_mode: bool) -> Dict[str, Any]:
    from model_adapters.sign_language import predict_sign_language

    kwargs = {
        "video_path": video_path,
        "sentence": sentence_mode,
        "top_k": int(st.session_state[TOP_K_KEY]),
        "threshold": float(st.session_state[THRESHOLD_KEY]),
        "min_pause_sec": float(st.session_state[MIN_PAUSE_KEY]),
        "min_segment_sec": float(st.session_state[MIN_SEGMENT_KEY]),
        "confidence_threshold": float(st.session_state[CONFIDENCE_KEY]),
        "use_language_decoder": bool(st.session_state[DECODER_KEY]),
    }

    attempts = [
        lambda: predict_sign_language(**kwargs),
        lambda: predict_sign_language(video_path=str(video_path), sentence=sentence_mode),
        lambda: predict_sign_language(str(video_path), sentence_mode),
        lambda: predict_sign_language(str(video_path)),
    ]

    last_error = None
    for attempt in attempts:
        try:
            raw = attempt()
            return {
                "ok": True,
                "result": raw,
                "text": result_to_text(raw),
                "mode": "Multi-sign sentence" if sentence_mode else "Single isolated sign",
                "video_path": str(video_path),
                "video_name": st.session_state.get(VIDEO_NAME_KEY, ""),
            }
        except TypeError as exc:
            last_error = exc
            continue

    raise RuntimeError(f"Could not call predict_sign_language with supported signatures. Last error: {last_error}")


def clear_saved_sign() -> None:
    st.session_state[VIDEO_PATH_KEY] = ""
    st.session_state[VIDEO_NAME_KEY] = ""
    st.session_state[RESULT_KEY] = {}
    st.session_state[ERROR_KEY] = ""


init_state()

mode = st.radio(
    "Prediction mode",
    ["Single isolated sign", "Multi-sign sentence"],
    key=MODE_KEY,
    horizontal=True,
)

uploaded_video = st.file_uploader(
    "Upload a sign video",
    type=["mp4", "webm", "avi", "mov", "mkv"],
    key="sign_video_upload_widget",
)

if uploaded_video is not None:
    data = uploaded_video.getvalue()
    saved_path = save_video(data, uploaded_video.name)
    st.session_state[VIDEO_PATH_KEY] = str(saved_path)
    st.session_state[VIDEO_NAME_KEY] = uploaded_video.name
    st.session_state[ERROR_KEY] = ""

current_video = st.session_state.get(VIDEO_PATH_KEY, "")
current_name = st.session_state.get(VIDEO_NAME_KEY, "")

if current_video and Path(current_video).exists():
    st.info(f"Saved input video → {current_name or Path(current_video).name}")
    st.video(current_video)
    st.caption("This video stays here when you switch pages and come back.")

with st.expander("Advanced settings", expanded=False):
    st.number_input("Top K", min_value=1, max_value=10, step=1, key=TOP_K_KEY)

    st.slider("Sentence threshold", 0.0, 1.0, key=THRESHOLD_KEY, step=0.01)
    st.slider("Minimum pause seconds", 0.0, 2.0, key=MIN_PAUSE_KEY, step=0.05)
    st.slider("Minimum segment seconds", 0.0, 2.0, key=MIN_SEGMENT_KEY, step=0.05)
    st.slider("Confidence threshold", 0.0, 1.0, key=CONFIDENCE_KEY, step=0.01)
    st.checkbox("Use language decoder", key=DECODER_KEY)

col_run, col_clear = st.columns([3, 1])

with col_run:
    run_clicked = st.button("Run Sign Model", type="primary", use_container_width=True)

with col_clear:
    clear_clicked = st.button("Clear saved", use_container_width=True)

if clear_clicked:
    clear_saved_sign()
    st.success("Saved sign input/output cleared.")

if run_clicked:
    current_video = st.session_state.get(VIDEO_PATH_KEY, "")

    if not current_video or not Path(current_video).exists():
        st.warning("Upload a sign video first.")
    else:
        with st.spinner("Running sign model..."):
            try:
                sentence_mode = st.session_state[MODE_KEY] == "Multi-sign sentence"
                output = predict_sign(Path(current_video), sentence_mode)

                st.session_state[RESULT_KEY] = output
                st.session_state[ERROR_KEY] = ""

            except Exception as exc:
                st.session_state[ERROR_KEY] = str(exc)
                st.session_state[RESULT_KEY] = {}

last_error = st.session_state.get(ERROR_KEY, "")
last_result = st.session_state.get(RESULT_KEY, {}) or {}

if last_error:
    st.error(last_error)

if last_result:
    st.divider()
    st.subheader("Last sign prediction")

    text = last_result.get("text", "")
    if text:
        st.success(text)
    else:
        st.info("Prediction completed.")

    st.caption("This output stays here when you switch pages and come back, until you clear it or run a new prediction.")

    with st.expander("Raw sign model result", expanded=False):
        st.json(last_result)
