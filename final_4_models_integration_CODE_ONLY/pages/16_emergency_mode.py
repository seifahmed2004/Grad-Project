from __future__ import annotations

import json
from typing import Any, Dict

import requests
import streamlit as st
import streamlit.components.v1 as components


st.title("🚨 Emergency Mode")
st.caption("Instant emergency speech + optional model-based voice from the warmed TTS server.")


ENDPOINT_KEY = "emergency_tts_endpoint"
SELECTED_TEXT_KEY = "emergency_selected_text"
LAST_AUDIO_KEY = "emergency_last_audio_url"
LAST_TEXT_KEY = "emergency_last_text"
LAST_RESULT_KEY = "emergency_last_result"
LAST_ERROR_KEY = "emergency_last_error"


def init_state() -> None:
    st.session_state.setdefault(ENDPOINT_KEY, "http://127.0.0.1:8000/api/tts")
    st.session_state.setdefault(SELECTED_TEXT_KEY, "")
    st.session_state.setdefault(LAST_AUDIO_KEY, "")
    st.session_state.setdefault(LAST_TEXT_KEY, "")
    st.session_state.setdefault(LAST_RESULT_KEY, {})
    st.session_state.setdefault(LAST_ERROR_KEY, "")


def profile_value(keys, default=""):
    user = st.session_state.get("user", {}) or {}
    if isinstance(user, dict):
        for k in keys:
            if user.get(k) not in (None, ""):
                return user.get(k)

    for k in keys:
        if st.session_state.get(k) not in (None, ""):
            return st.session_state.get(k)

    return default


def absolute_audio_url(endpoint: str, audio_url: str | None) -> str:
    if not audio_url:
        return ""
    audio_url = str(audio_url)
    if audio_url.startswith("http://") or audio_url.startswith("https://"):
        return audio_url
    if audio_url.startswith("/"):
        return endpoint.replace("/api/tts", "").rstrip("/") + audio_url
    return audio_url


def tts_api(text: str, endpoint: str, gender: str, age: str) -> Dict[str, Any]:
    response = requests.post(
        endpoint,
        data={
            "text": text,
            "gender": str(gender or ""),
            "age": str(age or ""),
        },
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


def browser_speech(text: str, autoplay: bool = True):
    text_json = json.dumps(text)
    autoplay_js = "setTimeout(speakEmergency, 250);" if autoplay else ""

    components.html(
        f"""
        <div style="padding:14px;border-radius:14px;background:#111827;border:1px solid rgba(255,255,255,.12);">
          <button id="emergencySpeakBtn" style="
              width:100%;
              padding:15px;
              border-radius:12px;
              border:0;
              background:#ef4444;
              color:white;
              font-weight:900;
              cursor:pointer;
              font-size:16px;
          ">▶ Speak Emergency Phrase Instantly</button>
          <p id="emergencySpeechStatus" style="color:#cbd5e1;margin-top:10px;">
              Press the button if audio does not start automatically.
          </p>
        </div>
        <script>
        const emergencyText = {text_json};

        function pickVoice() {{
            const voices = window.speechSynthesis.getVoices() || [];
            return voices.find(v => (v.lang || "").toLowerCase().startsWith("en")) || voices[0] || null;
        }}

        function speakEmergency() {{
            const status = document.getElementById("emergencySpeechStatus");
            if (!("speechSynthesis" in window)) {{
                status.textContent = "Browser speech is not supported.";
                return;
            }}

            window.speechSynthesis.cancel();
            const msg = new SpeechSynthesisUtterance(emergencyText);
            msg.lang = "en-US";
            msg.rate = 1.02;
            msg.pitch = 1.0;

            const voice = pickVoice();
            if (voice) msg.voice = voice;

            msg.onstart = () => status.textContent = "Speaking now...";
            msg.onend = () => status.textContent = "Done.";
            msg.onerror = (e) => status.textContent = "Speech error: " + (e.error || "unknown");

            window.speechSynthesis.speak(msg);
        }}

        document.getElementById("emergencySpeakBtn").addEventListener("click", speakEmergency);
        {autoplay_js}
        </script>
        """,
        height=125,
    )


def generate_model_voice(text: str) -> None:
    gender = str(profile_value(["gender", "profile_gender", "predicted_gender"], ""))
    age = str(profile_value(["age", "profile_age", "predicted_age"], ""))

    data = tts_api(text, st.session_state[ENDPOINT_KEY], gender, age)

    if not data.get("ok"):
        st.session_state[LAST_ERROR_KEY] = str(data.get("error") or data.get("detail") or "TTS failed.")
        st.session_state[LAST_TEXT_KEY] = text
        st.session_state[LAST_AUDIO_KEY] = ""
        st.session_state[LAST_RESULT_KEY] = data
        return

    result = data.get("result", {})
    audio_url = absolute_audio_url(st.session_state[ENDPOINT_KEY], result.get("audio_url"))

    if not audio_url:
        st.session_state[LAST_ERROR_KEY] = "TTS succeeded but no audio URL was returned."
        st.session_state[LAST_TEXT_KEY] = text
        st.session_state[LAST_AUDIO_KEY] = ""
        st.session_state[LAST_RESULT_KEY] = data
        return

    st.session_state[LAST_TEXT_KEY] = text
    st.session_state[LAST_AUDIO_KEY] = audio_url
    st.session_state[LAST_RESULT_KEY] = result
    st.session_state[LAST_ERROR_KEY] = ""


init_state()

gender = profile_value(["gender", "profile_gender", "predicted_gender"], "Unknown")
age = profile_value(["age", "profile_age", "predicted_age"], "Unknown")

st.info(f"Voice profile → Gender: {gender or 'Unknown'} | Age: {age or 'Unknown'}")

with st.expander("Emergency TTS settings", expanded=False):
    st.text_input("TTS API endpoint", key=ENDPOINT_KEY)

    if st.button("Warm up model TTS", use_container_width=True):
        try:
            warmup_url = st.session_state[ENDPOINT_KEY].replace("/api/tts", "/api/tts/warmup")
            result = requests.get(warmup_url, timeout=180).json()
            if result.get("ok") or result.get("tts_warmed_up"):
                st.success("TTS warmup completed.")
            else:
                st.warning(result)
        except Exception as exc:
            st.error(f"Warmup failed: {exc}")

phrases = [
    ("I need help", "أحتاج مساعدة"),
    ("Call a doctor", "اتصل بالطبيب"),
    ("I am in pain", "أنا أتألم"),
    ("Where is the hospital?", "أين المستشفى؟"),
    ("Call my family", "اتصل بعائلتي"),
    ("I need water", "أحتاج ماء"),
    ("I cannot hear you", "لا أستطيع سماعك"),
    ("Please speak slowly", "من فضلك تحدث ببطء"),
]

display_lang = st.radio("Display language", ["English", "Arabic"], horizontal=True)

cols = st.columns(2)
for idx, (en, ar) in enumerate(phrases):
    label = ar if display_lang == "Arabic" else en
    with cols[idx % 2]:
        if st.button(label, key=f"emergency_phrase_{idx}", use_container_width=True, type="primary"):
            st.session_state[SELECTED_TEXT_KEY] = en
            st.session_state[LAST_TEXT_KEY] = en
            st.session_state[LAST_ERROR_KEY] = ""

custom = st.text_input("Custom emergency phrase", placeholder="Type a phrase to speak...", key="emergency_custom_phrase")
if st.button("Use custom phrase", use_container_width=True):
    if not custom.strip():
        st.warning("Write a phrase first.")
    else:
        st.session_state[SELECTED_TEXT_KEY] = custom.strip()
        st.session_state[LAST_TEXT_KEY] = custom.strip()
        st.session_state[LAST_ERROR_KEY] = ""

selected_text = st.session_state.get(SELECTED_TEXT_KEY, "")

if selected_text:
    st.divider()
    st.subheader("Selected emergency phrase")
    st.success(selected_text)

    # This is the instant reliable emergency voice.
    browser_speech(selected_text, autoplay=True)

    col_model, col_clear = st.columns(2)
    with col_model:
        if st.button("Generate model voice too", use_container_width=True):
            with st.spinner("Generating model voice from warmed TTS API..."):
                generate_model_voice(selected_text)

    with col_clear:
        if st.button("Clear saved emergency voice", use_container_width=True):
            st.session_state[SELECTED_TEXT_KEY] = ""
            st.session_state[LAST_AUDIO_KEY] = ""
            st.session_state[LAST_TEXT_KEY] = ""
            st.session_state[LAST_RESULT_KEY] = {}
            st.session_state[LAST_ERROR_KEY] = ""
            st.success("Saved emergency voice cleared.")

last_error = st.session_state.get(LAST_ERROR_KEY, "")
last_audio = st.session_state.get(LAST_AUDIO_KEY, "")
last_result = st.session_state.get(LAST_RESULT_KEY, {}) or {}

if last_error:
    st.error(last_error)

if last_audio:
    st.subheader("Last model-generated emergency voice")
    st.audio(last_audio)
    st.caption("This model voice stays here when you switch pages until you clear it or generate another one.")
    with st.expander("Model TTS details", expanded=False):
        st.json(last_result)
