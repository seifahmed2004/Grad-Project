from __future__ import annotations

from pathlib import Path
from datetime import datetime
import shutil
import json
import pickle
import re


ROOT_EXPECTED = Path("final_4_models_integration_CODE_ONLY")

SRC_STATE = Path(__file__).resolve().parent / "app" / "persistent_page_state.py"
DST_STATE = ROOT_EXPECTED / "app" / "persistent_page_state.py"

SRC_STT_PAGE = Path(__file__).resolve().parent / "pages" / "12_speech_to_text.py"
DST_STT_PAGE = ROOT_EXPECTED / "pages" / "12_speech_to_text.py"

SRC_EMERGENCY_PAGE = Path(__file__).resolve().parent / "pages" / "16_emergency_mode.py"
DST_EMERGENCY_PAGE = ROOT_EXPECTED / "pages" / "16_emergency_mode.py"

LIVE_PAGE = ROOT_EXPECTED / "pages" / "15_live_sign_translation.py"
API_FILE = ROOT_EXPECTED / "sign_live_api.py"

STATE_JSON = ROOT_EXPECTED / ".ishara_local_state" / "session_state_snapshot.json"
STATE_PICKLE = ROOT_EXPECTED / ".ishara_local_state" / "session_state_snapshot.pkl"


UNSAFE_EXACT_KEYS = {
    "stt_recorded_audio",
    "stt_uploaded_audio",
    "live_stt_upload",
    "live_stt_upload_widget",
    "live_stt_recorder_widget",
    "stt_recorded_audio_widget",
    "stt_uploaded_audio_widget",
    "clear_persistent_page_state_button",
}

UNSAFE_PREFIXES = ("_", "$", "FormSubmitter", "emergency_phrase_")
UNSAFE_PARTS = ("file_uploader", "uploaded_file", "upload_widget", "audio_input", "recorded_audio", "recorder_widget", "camera_input", "download_button", "form_submit")


def is_safe_key(key: str) -> bool:
    key = str(key)
    low = key.lower()
    if key in UNSAFE_EXACT_KEYS:
        return False
    if key.startswith(UNSAFE_PREFIXES):
        return False
    if key.startswith("btn_") or key.startswith("button_"):
        return False
    if low.endswith("_upload") or low.endswith("_record") or low.endswith("_recorder"):
        return False
    if any(part in low for part in UNSAFE_PARTS):
        return False
    return True


def backup(path: Path) -> None:
    if path.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst = path.with_suffix(path.suffix + f".before_stt_emergency_fix_{ts}.bak")
        shutil.copy2(path, dst)
        print(f"Backup: {dst}")


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing patch source: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    backup(dst)
    shutil.copy2(src, dst)
    print(f"Installed: {dst}")


def clean_json_state(path: Path) -> None:
    if not path.exists():
        return
    backup(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        state = payload.get("state", {})
        if isinstance(state, dict):
            cleaned = {k: v for k, v in state.items() if is_safe_key(k)}
            removed = sorted(set(state.keys()) - set(cleaned.keys()))
            payload["state"] = cleaned
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Cleaned JSON state. Removed unsafe keys: {removed}")
    except Exception as exc:
        print(f"Could not clean JSON state, deleting it: {exc}")
        try:
            path.unlink()
        except Exception:
            pass


def clean_pickle_state(path: Path) -> None:
    if not path.exists():
        return
    backup(path)
    try:
        with path.open("rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            cleaned = {k: v for k, v in data.items() if is_safe_key(str(k))}
            removed = sorted(set(map(str, data.keys())) - set(map(str, cleaned.keys())))
            with path.open("wb") as f:
                pickle.dump(cleaned, f)
            print(f"Cleaned pickle state. Removed unsafe keys: {removed}")
    except Exception as exc:
        print(f"Could not clean pickle state, deleting it: {exc}")
        try:
            path.unlink()
        except Exception:
            pass


def insert_before_main(text: str, code: str) -> str:
    if "FAST STT API PATCH" in text:
        return text

    m = re.search(r"\nif\s+__name__\s*==\s*['\"]__main__['\"]\s*:", text)
    if m:
        return text[:m.start()] + "\n\n" + code + "\n" + text[m.start():]

    return text.rstrip() + "\n\n" + code + "\n"


def patch_api(root: Path) -> None:
    api_path = root / API_FILE
    if not api_path.exists():
        print(f"Skipped API patch: missing {api_path}")
        return

    backup(api_path)
    text = api_path.read_text(encoding="utf-8")
    original = text
    text = insert_before_main(text, '# --- FAST STT API PATCH ---\nimport hashlib as _hashlib\nimport wave as _wave\nimport threading as _threading\nfrom typing import Optional as _Optional, Any as _Any, Dict as _Dict\n\nfrom fastapi import UploadFile as _UploadFile, File as _File\n\nSTT_API_CACHE: _Dict[str, _Dict[str, _Any]] = {}\nSTT_API_WARMED_UP = False\nSTT_API_WARMUP_ERROR: _Optional[str] = None\n\n\ndef _stt_sha(data: bytes) -> str:\n    return _hashlib.sha256(data).hexdigest()\n\n\ndef _stt_extract_text(raw: _Any) -> str:\n    if isinstance(raw, str):\n        return raw.strip()\n\n    if isinstance(raw, dict):\n        return str(\n            raw.get("text")\n            or raw.get("transcript")\n            or raw.get("clean_text")\n            or raw.get("prediction")\n            or raw.get("sentence")\n            or ""\n        ).strip()\n\n    return str(raw or "").strip()\n\n\ndef _run_stt_adapter(audio_path: Path) -> _Dict[str, _Any]:\n    import model_adapters.speech_to_text as speech_adapter\n\n    candidate_names = [\n        "transcribe_speech",\n        "speech_to_text",\n        "transcribe_audio",\n        "predict_speech",\n        "run_speech_to_text",\n        "predict",\n    ]\n\n    last_error = None\n    for name in candidate_names:\n        fn = getattr(speech_adapter, name, None)\n        if not callable(fn):\n            continue\n\n        attempts = [\n            lambda: fn(audio_path=str(audio_path)),\n            lambda: fn(audio_file=str(audio_path)),\n            lambda: fn(file_path=str(audio_path)),\n            lambda: fn(path=str(audio_path)),\n            lambda: fn(str(audio_path)),\n        ]\n\n        for attempt in attempts:\n            try:\n                raw = attempt()\n                return {\n                    "ok": True,\n                    "text": _stt_extract_text(raw),\n                    "raw_result": _json_safe(raw) if "_json_safe" in globals() else raw,\n                    "adapter_function": name,\n                }\n            except TypeError as exc:\n                last_error = exc\n                continue\n\n    return {\n        "ok": False,\n        "error": "No compatible speech-to-text adapter function found in model_adapters/speech_to_text.py",\n        "last_error": str(last_error) if last_error else None,\n    }\n\n\ndef _make_silent_wav(path: Path) -> None:\n    path.parent.mkdir(parents=True, exist_ok=True)\n    sample_rate = 16000\n    seconds = 1\n    samples = b"\\x00\\x00" * sample_rate * seconds\n\n    with _wave.open(str(path), "wb") as wav:\n        wav.setnchannels(1)\n        wav.setsampwidth(2)\n        wav.setframerate(sample_rate)\n        wav.writeframes(samples)\n\n\ndef _stt_warmup_blocking() -> _Dict[str, _Any]:\n    global STT_API_WARMED_UP, STT_API_WARMUP_ERROR\n\n    if STT_API_WARMED_UP:\n        return {\n            "ok": True,\n            "stt_warmed_up": True,\n            "stt_error": STT_API_WARMUP_ERROR,\n            "cached": True,\n        }\n\n    try:\n        warm_path = OUTPUTS_DIR / "stt_warmup_silence.wav"\n        _make_silent_wav(warm_path)\n\n        # This may return empty text; the important part is loading the model once.\n        _run_stt_adapter(warm_path)\n\n        STT_API_WARMED_UP = True\n        STT_API_WARMUP_ERROR = None\n        return {"ok": True, "stt_warmed_up": True, "message": "STT warmup completed."}\n\n    except Exception as exc:\n        STT_API_WARMED_UP = False\n        STT_API_WARMUP_ERROR = str(exc)\n        return {"ok": False, "stt_warmed_up": False, "error": str(exc)}\n\n\ndef _stt_warmup_background() -> None:\n    _thread = _threading.Thread(target=_stt_warmup_blocking, name="stt-warmup", daemon=True)\n    _thread.start()\n\n\n@app.on_event("startup")\ndef _startup_fast_stt_warmup():\n    _stt_warmup_background()\n\n\n@app.get("/api/stt/warmup")\ndef stt_warmup():\n    return _stt_warmup_blocking()\n\n\n@app.get("/api/stt/warmup-status")\ndef stt_warmup_status():\n    return {\n        "ok": STT_API_WARMED_UP,\n        "stt_warmed_up": STT_API_WARMED_UP,\n        "stt_error": STT_API_WARMUP_ERROR,\n        "cache_size": len(STT_API_CACHE),\n    }\n\n\n@app.post("/api/stt")\nasync def speech_to_text_api(file: _UploadFile = _File(...)):\n    try:\n        data = await file.read()\n        if not data:\n            raise ValueError("Uploaded audio is empty.")\n\n        digest = _stt_sha(data)\n        if digest in STT_API_CACHE:\n            cached = dict(STT_API_CACHE[digest])\n            cached["cached"] = True\n            return {"ok": True, "result": cached}\n\n        ext = Path(file.filename or "audio.wav").suffix.lower() or ".wav"\n        if ext not in [".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"]:\n            ext = ".wav"\n\n        in_dir = OUTPUTS_DIR / "stt_api_inputs"\n        in_dir.mkdir(parents=True, exist_ok=True)\n        audio_path = in_dir / f"stt_{digest[:24]}{ext}"\n        if not audio_path.exists():\n            audio_path.write_bytes(data)\n\n        raw = _run_stt_adapter(audio_path)\n\n        if not raw.get("ok"):\n            return {"ok": False, "error": raw.get("error", "STT failed."), "result": raw}\n\n        result = {\n            "text": raw.get("text", ""),\n            "audio_path": str(audio_path),\n            "adapter_function": raw.get("adapter_function"),\n            "raw": raw,\n            "cached": False,\n        }\n\n        STT_API_CACHE[digest] = result\n        return {"ok": True, "result": result}\n\n    except Exception as exc:\n        return _error_response(exc)\n# --- END FAST STT API PATCH ---\n')

    if text != original:
        api_path.write_text(text, encoding="utf-8")
        print("Patched sign_live_api.py with fast STT API endpoints.")
    else:
        print("sign_live_api.py already has fast STT API patch.")


def patch_live_page(root: Path) -> None:
    page_path = root / LIVE_PAGE
    if not page_path.exists():
        print(f"Skipped live page patch: missing {page_path}")
        return

    backup(page_path)
    text = page_path.read_text(encoding="utf-8")
    original = text

    marker = "# ===============================\n# Speech → Text section for hearing speaker"
    if marker in text:
        text = text[: text.index(marker)] + '# ===============================\n# Speech → Text section for hearing speaker\n# ===============================\n\nst.divider()\nst.header("🎙️ Speech to Text for the Deaf User")\nst.caption("If the hearing person wants to speak, record their voice here and convert it to text for the deaf user to read.")\n\nLIVE_STT_ENDPOINT_KEY = "live_stt_api_endpoint"\nLIVE_STT_AUDIO_PATH_KEY = "live_stt_audio_path"\nLIVE_STT_AUDIO_NAME_KEY = "live_stt_audio_name"\nLIVE_STT_TEXT_KEY = "live_stt_last_text"\nLIVE_STT_RAW_KEY = "live_stt_last_raw"\nLIVE_STT_ERROR_KEY = "live_stt_last_error"\n\nst.session_state.setdefault(LIVE_STT_ENDPOINT_KEY, "http://127.0.0.1:8000/api/stt")\nst.session_state.setdefault(LIVE_STT_AUDIO_PATH_KEY, "")\nst.session_state.setdefault(LIVE_STT_AUDIO_NAME_KEY, "")\nst.session_state.setdefault(LIVE_STT_TEXT_KEY, "")\nst.session_state.setdefault(LIVE_STT_RAW_KEY, {})\nst.session_state.setdefault(LIVE_STT_ERROR_KEY, "")\n\nLIVE_STT_DIR = PROJECT_ROOT / "outputs" / "live_stt_inputs"\nLIVE_STT_DIR.mkdir(parents=True, exist_ok=True)\n\n\ndef _live_hash_bytes(data: bytes) -> str:\n    import hashlib\n    return hashlib.sha256(data).hexdigest()[:24]\n\n\ndef _live_save_audio(data: bytes, name: str, prefix: str) -> Path:\n    suffix = Path(name or "audio.wav").suffix.lower() or ".wav"\n    if suffix not in [".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"]:\n        suffix = ".wav"\n\n    path = LIVE_STT_DIR / f"{prefix}_{_live_hash_bytes(data)}{suffix}"\n    if not path.exists():\n        path.write_bytes(data)\n    return path\n\n\ndef _live_extract_stt_text(data: Dict[str, Any]) -> str:\n    result = data.get("result", data)\n    if isinstance(result, dict):\n        return str(\n            result.get("text")\n            or result.get("transcript")\n            or result.get("clean_text")\n            or result.get("prediction")\n            or result.get("sentence")\n            or ""\n        ).strip()\n    return str(result or "").strip()\n\n\ndef _live_call_stt_api(endpoint: str, audio_path: Path) -> Dict[str, Any]:\n    import requests\n\n    with audio_path.open("rb") as f:\n        response = requests.post(\n            endpoint,\n            files={"file": (audio_path.name, f, "application/octet-stream")},\n            timeout=180,\n        )\n\n    try:\n        data = response.json()\n    except Exception:\n        data = {"ok": False, "error": response.text}\n\n    if not response.ok and "ok" not in data:\n        data["ok"] = False\n        data["error"] = data.get("error") or f"HTTP {response.status_code}"\n\n    return data\n\n\nwith st.expander("Speech to Text settings", expanded=False):\n    st.text_input("STT API endpoint", key=LIVE_STT_ENDPOINT_KEY)\n    if st.button("Warm up live STT model", use_container_width=True):\n        import requests\n        try:\n            warmup_url = st.session_state[LIVE_STT_ENDPOINT_KEY].replace("/api/stt", "/api/stt/warmup")\n            result = requests.get(warmup_url, timeout=180).json()\n            if result.get("ok") or result.get("stt_warmed_up"):\n                st.success("Live STT warmup completed.")\n            else:\n                st.warning(result)\n        except Exception as exc:\n            st.error(f"Warmup failed: {exc}")\n\ntry:\n    audio_value = st.audio_input("Record voice message", key="live_stt_recorder_widget")\nexcept Exception:\n    audio_value = None\n    st.warning("Your Streamlit version may not support st.audio_input. Use upload below instead.")\n\nuploaded_audio = st.file_uploader(\n    "Or upload an audio file",\n    type=["wav", "mp3", "m4a", "ogg", "flac", "webm"],\n    key="live_stt_upload_widget",\n)\n\nif audio_value is not None:\n    audio_data = audio_value.getvalue()\n    audio_name = getattr(audio_value, "name", "recording.wav") or "recording.wav"\n    audio_path = _live_save_audio(audio_data, audio_name, "recorded")\n    st.session_state[LIVE_STT_AUDIO_PATH_KEY] = str(audio_path)\n    st.session_state[LIVE_STT_AUDIO_NAME_KEY] = "Recorded voice message"\n\nelif uploaded_audio is not None:\n    audio_data = uploaded_audio.getvalue()\n    audio_path = _live_save_audio(audio_data, uploaded_audio.name, "uploaded")\n    st.session_state[LIVE_STT_AUDIO_PATH_KEY] = str(audio_path)\n    st.session_state[LIVE_STT_AUDIO_NAME_KEY] = uploaded_audio.name\n\ncurrent_live_stt_audio = st.session_state.get(LIVE_STT_AUDIO_PATH_KEY, "")\nif current_live_stt_audio and Path(current_live_stt_audio).exists():\n    st.audio(current_live_stt_audio)\n    st.caption(f"Current audio: {st.session_state.get(LIVE_STT_AUDIO_NAME_KEY, \'\')}")\n\nif st.button("Convert Speech to Text", type="primary", use_container_width=True, key="live_stt_convert_button"):\n    current_live_stt_audio = st.session_state.get(LIVE_STT_AUDIO_PATH_KEY, "")\n\n    if not current_live_stt_audio or not Path(current_live_stt_audio).exists():\n        st.warning("Record or upload audio first.")\n    else:\n        with st.spinner("Transcribing with fast warmed STT API..."):\n            stt_result = _live_call_stt_api(\n                st.session_state[LIVE_STT_ENDPOINT_KEY],\n                Path(current_live_stt_audio),\n            )\n\n        if not stt_result.get("ok"):\n            st.session_state[LIVE_STT_ERROR_KEY] = str(stt_result.get("error") or stt_result.get("detail") or "STT failed.")\n            st.session_state[LIVE_STT_RAW_KEY] = stt_result\n        else:\n            st.session_state[LIVE_STT_TEXT_KEY] = _live_extract_stt_text(stt_result)\n            st.session_state[LIVE_STT_RAW_KEY] = stt_result\n            st.session_state[LIVE_STT_ERROR_KEY] = ""\n\nif st.session_state.get(LIVE_STT_ERROR_KEY):\n    st.error(st.session_state[LIVE_STT_ERROR_KEY])\n\nif st.session_state.get(LIVE_STT_TEXT_KEY):\n    st.success("Speech converted to text.")\n    st.markdown(\n        f"""\n        <div style="padding:22px;border-radius:18px;background:#111827;border:1px solid rgba(255,255,255,.12);">\n          <div style="font-size:14px;color:#94a3b8;margin-bottom:8px;">Text for the deaf user to read</div>\n          <div style="font-size:34px;font-weight:900;color:#22c55e;line-height:1.35;">{st.session_state[LIVE_STT_TEXT_KEY] or \'---\'}</div>\n        </div>\n        """,\n        unsafe_allow_html=True,\n    )\n    st.caption("This STT result stays here when you switch pages, until you record/upload another audio or clear it.")\n\nwith st.expander("Raw Live STT result", expanded=False):\n    st.json(st.session_state.get(LIVE_STT_RAW_KEY, {}))\n\nif st.button("Clear live STT result", use_container_width=True, key="live_stt_clear_button"):\n    st.session_state[LIVE_STT_AUDIO_PATH_KEY] = ""\n    st.session_state[LIVE_STT_AUDIO_NAME_KEY] = ""\n    st.session_state[LIVE_STT_TEXT_KEY] = ""\n    st.session_state[LIVE_STT_RAW_KEY] = {}\n    st.session_state[LIVE_STT_ERROR_KEY] = ""\n    st.success("Live STT result cleared.")\n' + "\n"
    else:
        # fallback: replace from any Speech to Text section if it exists
        alt = "st.header(\"🎙️ Speech to Text for the Deaf User\")"
        idx = text.find(alt)
        if idx != -1:
            before = text.rfind("\n", 0, idx)
            text = text[:before] + "\n" + '# ===============================\n# Speech → Text section for hearing speaker\n# ===============================\n\nst.divider()\nst.header("🎙️ Speech to Text for the Deaf User")\nst.caption("If the hearing person wants to speak, record their voice here and convert it to text for the deaf user to read.")\n\nLIVE_STT_ENDPOINT_KEY = "live_stt_api_endpoint"\nLIVE_STT_AUDIO_PATH_KEY = "live_stt_audio_path"\nLIVE_STT_AUDIO_NAME_KEY = "live_stt_audio_name"\nLIVE_STT_TEXT_KEY = "live_stt_last_text"\nLIVE_STT_RAW_KEY = "live_stt_last_raw"\nLIVE_STT_ERROR_KEY = "live_stt_last_error"\n\nst.session_state.setdefault(LIVE_STT_ENDPOINT_KEY, "http://127.0.0.1:8000/api/stt")\nst.session_state.setdefault(LIVE_STT_AUDIO_PATH_KEY, "")\nst.session_state.setdefault(LIVE_STT_AUDIO_NAME_KEY, "")\nst.session_state.setdefault(LIVE_STT_TEXT_KEY, "")\nst.session_state.setdefault(LIVE_STT_RAW_KEY, {})\nst.session_state.setdefault(LIVE_STT_ERROR_KEY, "")\n\nLIVE_STT_DIR = PROJECT_ROOT / "outputs" / "live_stt_inputs"\nLIVE_STT_DIR.mkdir(parents=True, exist_ok=True)\n\n\ndef _live_hash_bytes(data: bytes) -> str:\n    import hashlib\n    return hashlib.sha256(data).hexdigest()[:24]\n\n\ndef _live_save_audio(data: bytes, name: str, prefix: str) -> Path:\n    suffix = Path(name or "audio.wav").suffix.lower() or ".wav"\n    if suffix not in [".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"]:\n        suffix = ".wav"\n\n    path = LIVE_STT_DIR / f"{prefix}_{_live_hash_bytes(data)}{suffix}"\n    if not path.exists():\n        path.write_bytes(data)\n    return path\n\n\ndef _live_extract_stt_text(data: Dict[str, Any]) -> str:\n    result = data.get("result", data)\n    if isinstance(result, dict):\n        return str(\n            result.get("text")\n            or result.get("transcript")\n            or result.get("clean_text")\n            or result.get("prediction")\n            or result.get("sentence")\n            or ""\n        ).strip()\n    return str(result or "").strip()\n\n\ndef _live_call_stt_api(endpoint: str, audio_path: Path) -> Dict[str, Any]:\n    import requests\n\n    with audio_path.open("rb") as f:\n        response = requests.post(\n            endpoint,\n            files={"file": (audio_path.name, f, "application/octet-stream")},\n            timeout=180,\n        )\n\n    try:\n        data = response.json()\n    except Exception:\n        data = {"ok": False, "error": response.text}\n\n    if not response.ok and "ok" not in data:\n        data["ok"] = False\n        data["error"] = data.get("error") or f"HTTP {response.status_code}"\n\n    return data\n\n\nwith st.expander("Speech to Text settings", expanded=False):\n    st.text_input("STT API endpoint", key=LIVE_STT_ENDPOINT_KEY)\n    if st.button("Warm up live STT model", use_container_width=True):\n        import requests\n        try:\n            warmup_url = st.session_state[LIVE_STT_ENDPOINT_KEY].replace("/api/stt", "/api/stt/warmup")\n            result = requests.get(warmup_url, timeout=180).json()\n            if result.get("ok") or result.get("stt_warmed_up"):\n                st.success("Live STT warmup completed.")\n            else:\n                st.warning(result)\n        except Exception as exc:\n            st.error(f"Warmup failed: {exc}")\n\ntry:\n    audio_value = st.audio_input("Record voice message", key="live_stt_recorder_widget")\nexcept Exception:\n    audio_value = None\n    st.warning("Your Streamlit version may not support st.audio_input. Use upload below instead.")\n\nuploaded_audio = st.file_uploader(\n    "Or upload an audio file",\n    type=["wav", "mp3", "m4a", "ogg", "flac", "webm"],\n    key="live_stt_upload_widget",\n)\n\nif audio_value is not None:\n    audio_data = audio_value.getvalue()\n    audio_name = getattr(audio_value, "name", "recording.wav") or "recording.wav"\n    audio_path = _live_save_audio(audio_data, audio_name, "recorded")\n    st.session_state[LIVE_STT_AUDIO_PATH_KEY] = str(audio_path)\n    st.session_state[LIVE_STT_AUDIO_NAME_KEY] = "Recorded voice message"\n\nelif uploaded_audio is not None:\n    audio_data = uploaded_audio.getvalue()\n    audio_path = _live_save_audio(audio_data, uploaded_audio.name, "uploaded")\n    st.session_state[LIVE_STT_AUDIO_PATH_KEY] = str(audio_path)\n    st.session_state[LIVE_STT_AUDIO_NAME_KEY] = uploaded_audio.name\n\ncurrent_live_stt_audio = st.session_state.get(LIVE_STT_AUDIO_PATH_KEY, "")\nif current_live_stt_audio and Path(current_live_stt_audio).exists():\n    st.audio(current_live_stt_audio)\n    st.caption(f"Current audio: {st.session_state.get(LIVE_STT_AUDIO_NAME_KEY, \'\')}")\n\nif st.button("Convert Speech to Text", type="primary", use_container_width=True, key="live_stt_convert_button"):\n    current_live_stt_audio = st.session_state.get(LIVE_STT_AUDIO_PATH_KEY, "")\n\n    if not current_live_stt_audio or not Path(current_live_stt_audio).exists():\n        st.warning("Record or upload audio first.")\n    else:\n        with st.spinner("Transcribing with fast warmed STT API..."):\n            stt_result = _live_call_stt_api(\n                st.session_state[LIVE_STT_ENDPOINT_KEY],\n                Path(current_live_stt_audio),\n            )\n\n        if not stt_result.get("ok"):\n            st.session_state[LIVE_STT_ERROR_KEY] = str(stt_result.get("error") or stt_result.get("detail") or "STT failed.")\n            st.session_state[LIVE_STT_RAW_KEY] = stt_result\n        else:\n            st.session_state[LIVE_STT_TEXT_KEY] = _live_extract_stt_text(stt_result)\n            st.session_state[LIVE_STT_RAW_KEY] = stt_result\n            st.session_state[LIVE_STT_ERROR_KEY] = ""\n\nif st.session_state.get(LIVE_STT_ERROR_KEY):\n    st.error(st.session_state[LIVE_STT_ERROR_KEY])\n\nif st.session_state.get(LIVE_STT_TEXT_KEY):\n    st.success("Speech converted to text.")\n    st.markdown(\n        f"""\n        <div style="padding:22px;border-radius:18px;background:#111827;border:1px solid rgba(255,255,255,.12);">\n          <div style="font-size:14px;color:#94a3b8;margin-bottom:8px;">Text for the deaf user to read</div>\n          <div style="font-size:34px;font-weight:900;color:#22c55e;line-height:1.35;">{st.session_state[LIVE_STT_TEXT_KEY] or \'---\'}</div>\n        </div>\n        """,\n        unsafe_allow_html=True,\n    )\n    st.caption("This STT result stays here when you switch pages, until you record/upload another audio or clear it.")\n\nwith st.expander("Raw Live STT result", expanded=False):\n    st.json(st.session_state.get(LIVE_STT_RAW_KEY, {}))\n\nif st.button("Clear live STT result", use_container_width=True, key="live_stt_clear_button"):\n    st.session_state[LIVE_STT_AUDIO_PATH_KEY] = ""\n    st.session_state[LIVE_STT_AUDIO_NAME_KEY] = ""\n    st.session_state[LIVE_STT_TEXT_KEY] = ""\n    st.session_state[LIVE_STT_RAW_KEY] = {}\n    st.session_state[LIVE_STT_ERROR_KEY] = ""\n    st.success("Live STT result cleared.")\n' + "\n"
        else:
            text = text.rstrip() + "\n\n" + '# ===============================\n# Speech → Text section for hearing speaker\n# ===============================\n\nst.divider()\nst.header("🎙️ Speech to Text for the Deaf User")\nst.caption("If the hearing person wants to speak, record their voice here and convert it to text for the deaf user to read.")\n\nLIVE_STT_ENDPOINT_KEY = "live_stt_api_endpoint"\nLIVE_STT_AUDIO_PATH_KEY = "live_stt_audio_path"\nLIVE_STT_AUDIO_NAME_KEY = "live_stt_audio_name"\nLIVE_STT_TEXT_KEY = "live_stt_last_text"\nLIVE_STT_RAW_KEY = "live_stt_last_raw"\nLIVE_STT_ERROR_KEY = "live_stt_last_error"\n\nst.session_state.setdefault(LIVE_STT_ENDPOINT_KEY, "http://127.0.0.1:8000/api/stt")\nst.session_state.setdefault(LIVE_STT_AUDIO_PATH_KEY, "")\nst.session_state.setdefault(LIVE_STT_AUDIO_NAME_KEY, "")\nst.session_state.setdefault(LIVE_STT_TEXT_KEY, "")\nst.session_state.setdefault(LIVE_STT_RAW_KEY, {})\nst.session_state.setdefault(LIVE_STT_ERROR_KEY, "")\n\nLIVE_STT_DIR = PROJECT_ROOT / "outputs" / "live_stt_inputs"\nLIVE_STT_DIR.mkdir(parents=True, exist_ok=True)\n\n\ndef _live_hash_bytes(data: bytes) -> str:\n    import hashlib\n    return hashlib.sha256(data).hexdigest()[:24]\n\n\ndef _live_save_audio(data: bytes, name: str, prefix: str) -> Path:\n    suffix = Path(name or "audio.wav").suffix.lower() or ".wav"\n    if suffix not in [".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"]:\n        suffix = ".wav"\n\n    path = LIVE_STT_DIR / f"{prefix}_{_live_hash_bytes(data)}{suffix}"\n    if not path.exists():\n        path.write_bytes(data)\n    return path\n\n\ndef _live_extract_stt_text(data: Dict[str, Any]) -> str:\n    result = data.get("result", data)\n    if isinstance(result, dict):\n        return str(\n            result.get("text")\n            or result.get("transcript")\n            or result.get("clean_text")\n            or result.get("prediction")\n            or result.get("sentence")\n            or ""\n        ).strip()\n    return str(result or "").strip()\n\n\ndef _live_call_stt_api(endpoint: str, audio_path: Path) -> Dict[str, Any]:\n    import requests\n\n    with audio_path.open("rb") as f:\n        response = requests.post(\n            endpoint,\n            files={"file": (audio_path.name, f, "application/octet-stream")},\n            timeout=180,\n        )\n\n    try:\n        data = response.json()\n    except Exception:\n        data = {"ok": False, "error": response.text}\n\n    if not response.ok and "ok" not in data:\n        data["ok"] = False\n        data["error"] = data.get("error") or f"HTTP {response.status_code}"\n\n    return data\n\n\nwith st.expander("Speech to Text settings", expanded=False):\n    st.text_input("STT API endpoint", key=LIVE_STT_ENDPOINT_KEY)\n    if st.button("Warm up live STT model", use_container_width=True):\n        import requests\n        try:\n            warmup_url = st.session_state[LIVE_STT_ENDPOINT_KEY].replace("/api/stt", "/api/stt/warmup")\n            result = requests.get(warmup_url, timeout=180).json()\n            if result.get("ok") or result.get("stt_warmed_up"):\n                st.success("Live STT warmup completed.")\n            else:\n                st.warning(result)\n        except Exception as exc:\n            st.error(f"Warmup failed: {exc}")\n\ntry:\n    audio_value = st.audio_input("Record voice message", key="live_stt_recorder_widget")\nexcept Exception:\n    audio_value = None\n    st.warning("Your Streamlit version may not support st.audio_input. Use upload below instead.")\n\nuploaded_audio = st.file_uploader(\n    "Or upload an audio file",\n    type=["wav", "mp3", "m4a", "ogg", "flac", "webm"],\n    key="live_stt_upload_widget",\n)\n\nif audio_value is not None:\n    audio_data = audio_value.getvalue()\n    audio_name = getattr(audio_value, "name", "recording.wav") or "recording.wav"\n    audio_path = _live_save_audio(audio_data, audio_name, "recorded")\n    st.session_state[LIVE_STT_AUDIO_PATH_KEY] = str(audio_path)\n    st.session_state[LIVE_STT_AUDIO_NAME_KEY] = "Recorded voice message"\n\nelif uploaded_audio is not None:\n    audio_data = uploaded_audio.getvalue()\n    audio_path = _live_save_audio(audio_data, uploaded_audio.name, "uploaded")\n    st.session_state[LIVE_STT_AUDIO_PATH_KEY] = str(audio_path)\n    st.session_state[LIVE_STT_AUDIO_NAME_KEY] = uploaded_audio.name\n\ncurrent_live_stt_audio = st.session_state.get(LIVE_STT_AUDIO_PATH_KEY, "")\nif current_live_stt_audio and Path(current_live_stt_audio).exists():\n    st.audio(current_live_stt_audio)\n    st.caption(f"Current audio: {st.session_state.get(LIVE_STT_AUDIO_NAME_KEY, \'\')}")\n\nif st.button("Convert Speech to Text", type="primary", use_container_width=True, key="live_stt_convert_button"):\n    current_live_stt_audio = st.session_state.get(LIVE_STT_AUDIO_PATH_KEY, "")\n\n    if not current_live_stt_audio or not Path(current_live_stt_audio).exists():\n        st.warning("Record or upload audio first.")\n    else:\n        with st.spinner("Transcribing with fast warmed STT API..."):\n            stt_result = _live_call_stt_api(\n                st.session_state[LIVE_STT_ENDPOINT_KEY],\n                Path(current_live_stt_audio),\n            )\n\n        if not stt_result.get("ok"):\n            st.session_state[LIVE_STT_ERROR_KEY] = str(stt_result.get("error") or stt_result.get("detail") or "STT failed.")\n            st.session_state[LIVE_STT_RAW_KEY] = stt_result\n        else:\n            st.session_state[LIVE_STT_TEXT_KEY] = _live_extract_stt_text(stt_result)\n            st.session_state[LIVE_STT_RAW_KEY] = stt_result\n            st.session_state[LIVE_STT_ERROR_KEY] = ""\n\nif st.session_state.get(LIVE_STT_ERROR_KEY):\n    st.error(st.session_state[LIVE_STT_ERROR_KEY])\n\nif st.session_state.get(LIVE_STT_TEXT_KEY):\n    st.success("Speech converted to text.")\n    st.markdown(\n        f"""\n        <div style="padding:22px;border-radius:18px;background:#111827;border:1px solid rgba(255,255,255,.12);">\n          <div style="font-size:14px;color:#94a3b8;margin-bottom:8px;">Text for the deaf user to read</div>\n          <div style="font-size:34px;font-weight:900;color:#22c55e;line-height:1.35;">{st.session_state[LIVE_STT_TEXT_KEY] or \'---\'}</div>\n        </div>\n        """,\n        unsafe_allow_html=True,\n    )\n    st.caption("This STT result stays here when you switch pages, until you record/upload another audio or clear it.")\n\nwith st.expander("Raw Live STT result", expanded=False):\n    st.json(st.session_state.get(LIVE_STT_RAW_KEY, {}))\n\nif st.button("Clear live STT result", use_container_width=True, key="live_stt_clear_button"):\n    st.session_state[LIVE_STT_AUDIO_PATH_KEY] = ""\n    st.session_state[LIVE_STT_AUDIO_NAME_KEY] = ""\n    st.session_state[LIVE_STT_TEXT_KEY] = ""\n    st.session_state[LIVE_STT_RAW_KEY] = {}\n    st.session_state[LIVE_STT_ERROR_KEY] = ""\n    st.success("Live STT result cleared.")\n' + "\n"

    if text != original:
        page_path.write_text(text, encoding="utf-8")
        print("Patched live page STT section to use fast STT API and persist result.")
    else:
        print("Live page already patched.")


def main() -> None:
    root = Path.cwd()
    inner = root / ROOT_EXPECTED
    if not inner.exists():
        raise SystemExit(
            "Run this script from:\n"
            r"C:\Users\lenovo\Downloads\final_4_models_integration_CODE_ONLY"
        )

    copy_file(SRC_STATE, root / DST_STATE)
    clean_json_state(root / STATE_JSON)
    clean_pickle_state(root / STATE_PICKLE)

    copy_file(SRC_STT_PAGE, root / DST_STT_PAGE)
    copy_file(SRC_EMERGENCY_PAGE, root / DST_EMERGENCY_PAGE)

    patch_api(root)
    patch_live_page(root)

    print("\nDONE ✅ STT fixed in Live + STT page, Emergency sound fixed, and persistence fixed.")
    print("\nRestart BOTH servers:")
    print("\nTerminal 1:")
    print("  conda activate grad_py310")
    print(r'  cd "C:\Users\lenovo\Downloads\final_4_models_integration_CODE_ONLY\final_4_models_integration_CODE_ONLY"')
    print("  python sign_live_api.py")
    print("\nWarm up once:")
    print("  http://127.0.0.1:8000/api/stt/warmup")
    print("  http://127.0.0.1:8000/api/tts/warmup")
    print("\nTerminal 2:")
    print("  conda activate grad_py310")
    print(r'  cd "C:\Users\lenovo\Downloads\final_4_models_integration_CODE_ONLY\final_4_models_integration_CODE_ONLY"')
    print("  python -m streamlit run streamlit_app.py --server.fileWatcherType none --server.runOnSave false")
    print("\nDo NOT git push if deployment should stay unchanged.")


if __name__ == "__main__":
    main()
