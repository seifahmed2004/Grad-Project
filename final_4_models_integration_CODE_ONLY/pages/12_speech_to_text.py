from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict

import requests
import streamlit as st


st.title("🎙️ Speech to Text")
st.caption("Fast STT using the warmed FastAPI server, with persistent audio and transcript.")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "outputs" / "stt_inputs"
INPUT_DIR.mkdir(parents=True, exist_ok=True)

ENDPOINT_KEY = "stt_api_endpoint"
AUDIO_PATH_KEY = "stt_page_audio_path"
AUDIO_NAME_KEY = "stt_page_audio_name"
TEXT_KEY = "stt_page_last_text"
RAW_KEY = "stt_page_last_raw"
ERROR_KEY = "stt_page_last_error"
CACHE_KEY = "stt_page_cache"


def init_state() -> None:
    st.session_state.setdefault(ENDPOINT_KEY, "http://127.0.0.1:8000/api/stt")
    st.session_state.setdefault(AUDIO_PATH_KEY, "")
    st.session_state.setdefault(AUDIO_NAME_KEY, "")
    st.session_state.setdefault(TEXT_KEY, "")
    st.session_state.setdefault(RAW_KEY, {})
    st.session_state.setdefault(ERROR_KEY, "")
    st.session_state.setdefault(CACHE_KEY, {})


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:24]


def save_audio(data: bytes, name: str, prefix: str) -> Path:
    suffix = Path(name or "audio.wav").suffix.lower() or ".wav"
    if suffix not in [".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"]:
        suffix = ".wav"

    digest = sha_bytes(data)
    path = INPUT_DIR / f"{prefix}_{digest}{suffix}"

    if not path.exists():
        path.write_bytes(data)

    return path


def send_to_stt_api(endpoint: str, audio_path: Path) -> Dict[str, Any]:
    with audio_path.open("rb") as f:
        response = requests.post(
            endpoint,
            files={"file": (audio_path.name, f, "application/octet-stream")},
            timeout=180,
        )

    try:
        data = response.json()
    except Exception:
        data = {"ok": False, "error": response.text}

    if not response.ok and "ok" not in data:
        data["ok"] = False
        data["error"] = data.get("error") or f"HTTP {response.status_code}"

    return data


def extract_text(data: Dict[str, Any]) -> str:
    result = data.get("result", data)
    if isinstance(result, dict):
        return str(
            result.get("text")
            or result.get("transcript")
            or result.get("clean_text")
            or result.get("prediction")
            or result.get("sentence")
            or ""
        ).strip()
    return str(result or "").strip()


def warmup_stt(endpoint: str) -> Dict[str, Any]:
    warmup_url = endpoint.replace("/api/stt", "/api/stt/warmup")
    response = requests.get(warmup_url, timeout=180)
    try:
        return response.json()
    except Exception:
        return {"ok": response.ok, "raw": response.text}


def transcribe_current_audio() -> None:
    path_text = st.session_state.get(AUDIO_PATH_KEY, "")
    if not path_text or not Path(path_text).exists():
        st.session_state[ERROR_KEY] = "Record or upload audio first."
        return

    audio_path = Path(path_text)
    digest = sha_bytes(audio_path.read_bytes())
    cache: Dict[str, Any] = st.session_state.setdefault(CACHE_KEY, {})

    if digest in cache:
        cached = cache[digest]
        st.session_state[TEXT_KEY] = cached.get("text", "")
        st.session_state[RAW_KEY] = {**cached.get("raw", {}), "cached_local": True}
        st.session_state[ERROR_KEY] = ""
        return

    data = send_to_stt_api(st.session_state[ENDPOINT_KEY], audio_path)

    if not data.get("ok"):
        st.session_state[ERROR_KEY] = str(data.get("error") or data.get("detail") or "STT failed.")
        st.session_state[RAW_KEY] = data
        return

    text = extract_text(data)

    st.session_state[TEXT_KEY] = text
    st.session_state[RAW_KEY] = data
    st.session_state[ERROR_KEY] = ""

    cache[digest] = {"text": text, "raw": data}
    st.session_state[CACHE_KEY] = cache


init_state()

with st.expander("STT settings", expanded=False):
    st.text_input("STT API endpoint", key=ENDPOINT_KEY)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Warm up STT model", use_container_width=True):
            with st.spinner("Warming up STT model..."):
                try:
                    result = warmup_stt(st.session_state[ENDPOINT_KEY])
                    if result.get("ok") or result.get("stt_warmed_up"):
                        st.success("STT warmup completed.")
                    else:
                        st.warning(result)
                except Exception as exc:
                    st.error(f"Warmup failed: {exc}")

    with col2:
        if st.button("Clear saved STT", use_container_width=True):
            st.session_state[AUDIO_PATH_KEY] = ""
            st.session_state[AUDIO_NAME_KEY] = ""
            st.session_state[TEXT_KEY] = ""
            st.session_state[RAW_KEY] = {}
            st.session_state[ERROR_KEY] = ""
            st.success("Saved STT cleared.")


recorded_audio = None
if hasattr(st, "audio_input"):
    recorded_audio = st.audio_input("Record audio", key="stt_recorded_audio_widget")
else:
    st.warning("Your Streamlit version does not support st.audio_input. Use upload instead.")

uploaded_audio = st.file_uploader(
    "Or upload audio",
    type=["wav", "mp3", "m4a", "ogg", "flac", "webm"],
    key="stt_uploaded_audio_widget",
)

if recorded_audio is not None:
    data = recorded_audio.getvalue()
    name = getattr(recorded_audio, "name", "recorded.wav") or "recorded.wav"
    audio_path = save_audio(data, name, "recorded")
    st.session_state[AUDIO_PATH_KEY] = str(audio_path)
    st.session_state[AUDIO_NAME_KEY] = "Recorded audio"

elif uploaded_audio is not None:
    data = uploaded_audio.getvalue()
    audio_path = save_audio(data, uploaded_audio.name, "uploaded")
    st.session_state[AUDIO_PATH_KEY] = str(audio_path)
    st.session_state[AUDIO_NAME_KEY] = uploaded_audio.name


current_audio = st.session_state.get(AUDIO_PATH_KEY, "")
if current_audio and Path(current_audio).exists():
    st.audio(current_audio)
    st.caption(f"Current audio: {st.session_state.get(AUDIO_NAME_KEY, '')}")

if st.button("Convert Speech to Text", type="primary", use_container_width=True):
    with st.spinner("Transcribing with fast warmed STT API..."):
        transcribe_current_audio()

last_error = st.session_state.get(ERROR_KEY, "")
last_text = st.session_state.get(TEXT_KEY, "")
raw = st.session_state.get(RAW_KEY, {}) or {}

if last_error:
    st.warning(last_error)

if last_text:
    st.divider()
    st.subheader("Last transcription")
    st.success(last_text)
    st.caption("This audio and transcription stay here when you switch pages, until you clear them or record/upload a new audio.")

    with st.expander("Raw STT result", expanded=False):
        st.json(raw)
