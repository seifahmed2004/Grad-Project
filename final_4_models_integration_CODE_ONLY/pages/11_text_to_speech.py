from __future__ import annotations

import requests
import streamlit as st


st.title("🗣️ Text to Speech")
st.caption("Fast local TTS using the same warmed-up FastAPI server used by Live Sign Translation.")


# -----------------------------
# Persistent keys
# -----------------------------
TEXT_KEY = "tts_page_input_text"
ENDPOINT_KEY = "tts_page_api_endpoint"
LAST_AUDIO_KEY = "tts_page_last_audio_url"
LAST_TEXT_KEY = "tts_page_last_text"
LAST_RESULT_KEY = "tts_page_last_result"
LAST_ERROR_KEY = "tts_page_last_error"


def _init_state() -> None:
    st.session_state.setdefault(TEXT_KEY, "Hello, I need help.")
    st.session_state.setdefault(ENDPOINT_KEY, "http://127.0.0.1:8000/api/tts")
    st.session_state.setdefault(LAST_AUDIO_KEY, "")
    st.session_state.setdefault(LAST_TEXT_KEY, "")
    st.session_state.setdefault(LAST_RESULT_KEY, {})
    st.session_state.setdefault(LAST_ERROR_KEY, "")


def get_profile_value(keys, default=""):
    user = st.session_state.get("user", {}) or {}
    if isinstance(user, dict):
        for key in keys:
            value = user.get(key)
            if value not in (None, ""):
                return value

    for key in keys:
        value = st.session_state.get(key)
        if value not in (None, ""):
            return value

    for container_key in ["profile", "user_profile", "current_profile"]:
        profile = st.session_state.get(container_key)
        if isinstance(profile, dict):
            for key in keys:
                value = profile.get(key)
                if value not in (None, ""):
                    return value

    return default


def make_absolute_audio_url(endpoint: str, audio_url: str | None) -> str:
    if not audio_url:
        return ""
    audio_url = str(audio_url)
    if audio_url.startswith("http://") or audio_url.startswith("https://"):
        return audio_url
    if audio_url.startswith("/"):
        base_url = endpoint.replace("/api/tts", "").rstrip("/")
        return base_url + audio_url
    return audio_url


def call_tts_api(endpoint: str, text: str, gender: str, age: str) -> dict:
    response = requests.post(
        endpoint,
        data={
            "text": text,
            "gender": str(gender or ""),
            "age": str(age or ""),
        },
        timeout=120,
    )
    try:
        data = response.json()
    except Exception:
        data = {"ok": False, "error": response.text}

    if not response.ok and "ok" not in data:
        data["ok"] = False
        data["error"] = data.get("error") or f"HTTP {response.status_code}"

    return data


def warmup_tts(endpoint: str) -> dict:
    warmup_url = endpoint.replace("/api/tts", "/api/tts/warmup")
    response = requests.get(warmup_url, timeout=120)
    try:
        return response.json()
    except Exception:
        return {"ok": response.ok, "raw": response.text}


_init_state()

gender = get_profile_value(["gender", "profile_gender", "predicted_gender"], "Unknown")
age = get_profile_value(["age", "profile_age", "predicted_age"], "Unknown")

st.info(f"Voice profile → Gender: {gender or 'Unknown'} | Age: {age or 'Unknown'}")

with st.expander("TTS server settings", expanded=False):
    st.text_input(
        "FastAPI TTS endpoint",
        key=ENDPOINT_KEY,
        help="Keep sign_live_api.py running on 127.0.0.1:8000. This page uses the same TTS API as the Live page.",
    )

    col_warm, col_clear = st.columns(2)
    with col_warm:
        if st.button("Warm up TTS model", use_container_width=True):
            with st.spinner("Warming up TTS model..."):
                try:
                    result = warmup_tts(st.session_state[ENDPOINT_KEY])
                    if result.get("ok") or result.get("tts_warmed_up"):
                        st.success("TTS warmup completed.")
                    else:
                        st.warning(result)
                except Exception as exc:
                    st.error(f"Warmup failed: {exc}")

    with col_clear:
        if st.button("Clear saved speech", use_container_width=True):
            st.session_state[LAST_AUDIO_KEY] = ""
            st.session_state[LAST_TEXT_KEY] = ""
            st.session_state[LAST_RESULT_KEY] = {}
            st.session_state[LAST_ERROR_KEY] = ""
            st.success("Saved speech cleared.")

st.text_area(
    "Enter text",
    key=TEXT_KEY,
    height=150,
    placeholder="Type the text you want to convert to speech...",
)

generate = st.button("Generate Speech", type="primary", use_container_width=True)

if generate:
    clean_text = " ".join(str(st.session_state.get(TEXT_KEY, "")).split())

    if not clean_text:
        st.warning("Please enter text first.")
    else:
        with st.spinner("Generating speech from the fast live TTS API..."):
            try:
                data = call_tts_api(
                    endpoint=st.session_state[ENDPOINT_KEY],
                    text=clean_text,
                    gender=str(gender or ""),
                    age=str(age or ""),
                )

                if not data.get("ok"):
                    error = data.get("error") or data.get("detail") or "TTS failed."
                    st.session_state[LAST_ERROR_KEY] = str(error)
                    st.error(str(error))
                else:
                    result = data.get("result", {})
                    audio_url = make_absolute_audio_url(
                        st.session_state[ENDPOINT_KEY],
                        result.get("audio_url"),
                    )

                    if not audio_url:
                        raise RuntimeError("TTS succeeded but no audio_url was returned.")

                    st.session_state[LAST_AUDIO_KEY] = audio_url
                    st.session_state[LAST_TEXT_KEY] = clean_text
                    st.session_state[LAST_RESULT_KEY] = result
                    st.session_state[LAST_ERROR_KEY] = ""

                    cached = "cached" if result.get("cached") else "generated"
                    st.success(f"Speech {cached} successfully.")

            except Exception as exc:
                st.session_state[LAST_ERROR_KEY] = str(exc)
                st.error(f"TTS API error: {exc}")
                st.caption("Make sure Terminal 1 is running: python sign_live_api.py")


# -----------------------------
# Persistent output area
# -----------------------------
last_error = st.session_state.get(LAST_ERROR_KEY, "")
last_audio_url = st.session_state.get(LAST_AUDIO_KEY, "")
last_text = st.session_state.get(LAST_TEXT_KEY, "")
last_result = st.session_state.get(LAST_RESULT_KEY, {}) or {}

if last_audio_url:
    st.divider()
    st.subheader("Last generated speech")
    st.caption("This stays here when you switch pages and come back, until you clear it or generate a new one.")

    st.write(f"**Text:** {last_text}")
    st.audio(last_audio_url)

    with st.expander("TTS details", expanded=False):
        st.json(last_result)

elif last_error:
    st.error(last_error)
