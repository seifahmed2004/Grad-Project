from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = PROJECT_ROOT / ".ishara_local_state"
STATE_JSON = STATE_DIR / "session_state_snapshot.json"
STATE_PICKLE = STATE_DIR / "session_state_snapshot.pkl"


# Streamlit does NOT allow manually setting session_state values for:
# st.button, st.download_button, st.file_uploader, st.camera_input, st.audio_input,
# and form submit buttons. We skip these keys completely.
_UNSAFE_EXACT_KEYS = {
    "stt_recorded_audio",
    "stt_uploaded_audio",
    "stt_recorded_audio_widget",
    "stt_uploaded_audio_widget",
    "live_stt_upload",
    "live_stt_upload_widget",
    "live_stt_recorder_widget",
    "live_stt_convert_button",
    "live_stt_clear_button",
    "emergency_phrase_0",
    "emergency_phrase_1",
    "emergency_phrase_2",
    "emergency_phrase_3",
    "emergency_phrase_4",
    "emergency_phrase_5",
    "emergency_phrase_6",
    "emergency_phrase_7",
    "clear_persistent_page_state_button",
}

_UNSAFE_PREFIXES = (
    "_",
    "$",
    "FormSubmitter",
    "emergency_phrase_",
)

_UNSAFE_KEY_PARTS = (
    "button",
    "btn",
    "file_uploader",
    "uploaded_file",
    "upload_widget",
    "audio_input",
    "recorded_audio",
    "recorder_widget",
    "camera_input",
    "download_button",
    "form_submit",
)


def _is_safe_key(key: str) -> bool:
    key = str(key)
    low = key.lower()

    if key in _UNSAFE_EXACT_KEYS:
        return False

    if key.startswith(_UNSAFE_PREFIXES):
        return False

    # Anything ending in _button is a Streamlit button key.
    if low.endswith("_button") or low.endswith("_btn"):
        return False

    # Upload/record widgets are not manually restorable.
    if low.endswith("_upload") or low.endswith("_record") or low.endswith("_recorder"):
        return False

    if any(part in low for part in _UNSAFE_KEY_PARTS):
        return False

    return True


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return {"__type__": "path", "value": str(value)}

    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]

    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if isinstance(k, (str, int, float, bool)):
                out[str(k)] = _json_safe(v)
        return out

    raise TypeError(type(value).__name__)


def _restore_json_safe(value: Any) -> Any:
    if isinstance(value, dict) and value.get("__type__") == "path":
        return value.get("value", "")

    if isinstance(value, list):
        return [_restore_json_safe(v) for v in value]

    if isinstance(value, dict):
        return {k: _restore_json_safe(v) for k, v in value.items()}

    return value


def _collect_json_state() -> dict[str, Any]:
    data: dict[str, Any] = {}

    for key in list(st.session_state.keys()):
        if not _is_safe_key(key):
            continue

        try:
            data[str(key)] = _json_safe(st.session_state.get(key))
        except Exception:
            continue

    return data


def restore_persistent_state() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    if STATE_JSON.exists():
        try:
            payload = json.loads(STATE_JSON.read_text(encoding="utf-8"))
            data = payload.get("state", {})
            if isinstance(data, dict):
                for key, value in data.items():
                    if _is_safe_key(key) and key not in st.session_state:
                        st.session_state[key] = _restore_json_safe(value)
        except Exception as exc:
            print("Persistent JSON restore skipped:", exc)

    if STATE_PICKLE.exists():
        try:
            with STATE_PICKLE.open("rb") as f:
                pickled = pickle.load(f)

            if isinstance(pickled, dict):
                for key, value in pickled.items():
                    key = str(key)
                    if _is_safe_key(key) and key not in st.session_state:
                        st.session_state[key] = value
        except Exception as exc:
            print("Persistent pickle restore skipped:", exc)

    # Protect safe keys from Streamlit multipage cleanup.
    for key in list(st.session_state.keys()):
        if not _is_safe_key(str(key)):
            continue
        try:
            st.session_state[key] = st.session_state[key]
        except Exception:
            pass


def save_persistent_state() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "state": _collect_json_state(),
    }

    try:
        STATE_JSON.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        print("Persistent JSON save skipped:", exc)

    pickled: dict[str, Any] = {}
    for key in list(st.session_state.keys()):
        key = str(key)

        if not _is_safe_key(key):
            continue

        try:
            value = st.session_state.get(key)
            type_name = type(value).__name__.lower()

            if any(bad in type_name for bad in ["model", "tensor", "uploadedfile", "file"]):
                continue

            pickle.dumps(value)
            pickled[key] = value
        except Exception:
            continue

    try:
        with STATE_PICKLE.open("wb") as f:
            pickle.dump(pickled, f)
    except Exception as exc:
        print("Persistent pickle save skipped:", exc)


def clear_persistent_state() -> None:
    for path in [STATE_JSON, STATE_PICKLE]:
        try:
            if path.exists():
                path.unlink()
        except Exception:
            pass


def state_keeper_status() -> dict[str, Any]:
    return {
        "state_dir": str(STATE_DIR),
        "json_exists": STATE_JSON.exists(),
        "pickle_exists": STATE_PICKLE.exists(),
        "json_path": str(STATE_JSON),
        "pickle_path": str(STATE_PICKLE),
        "session_keys": list(st.session_state.keys()),
    }
