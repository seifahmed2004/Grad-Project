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

SRC_EMERGENCY = Path(__file__).resolve().parent / "pages" / "16_emergency_mode.py"
DST_EMERGENCY = ROOT_EXPECTED / "pages" / "16_emergency_mode.py"

LIVE_PAGE = ROOT_EXPECTED / "pages" / "15_live_sign_translation.py"

STATE_JSON = ROOT_EXPECTED / ".ishara_local_state" / "session_state_snapshot.json"
STATE_PICKLE = ROOT_EXPECTED / ".ishara_local_state" / "session_state_snapshot.pkl"


UNSAFE_EXACT_KEYS = {
    "stt_recorded_audio",
    "stt_uploaded_audio",
    "stt_recorded_audio_widget",
    "stt_uploaded_audio_widget",
    "live_stt_upload",
    "live_stt_upload_widget",
    "live_stt_recorder_widget",
    "live_stt_convert_button",
    "live_stt_clear_button",
    "clear_persistent_page_state_button",
}

UNSAFE_PREFIXES = ("_", "$", "FormSubmitter", "emergency_phrase_")
UNSAFE_PARTS = ("button", "btn", "file_uploader", "uploaded_file", "upload_widget", "audio_input", "recorded_audio", "recorder_widget", "camera_input", "download_button", "form_submit")


def is_safe_key(key: str) -> bool:
    key = str(key)
    low = key.lower()

    if key in UNSAFE_EXACT_KEYS:
        return False
    if key.startswith(UNSAFE_PREFIXES):
        return False
    if low.endswith("_button") or low.endswith("_btn"):
        return False
    if low.endswith("_upload") or low.endswith("_record") or low.endswith("_recorder"):
        return False
    if any(part in low for part in UNSAFE_PARTS):
        return False

    return True


def backup(path: Path) -> None:
    if path.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst = path.with_suffix(path.suffix + f".before_live_emergency_button_fix_{ts}.bak")
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


def patch_live_button_keys(root: Path) -> None:
    page = root / LIVE_PAGE
    if not page.exists():
        print(f"Skipped live page key cleanup: missing {page}")
        return

    backup(page)
    text = page.read_text(encoding="utf-8")
    original = text

    # Removing explicit keys from buttons avoids Streamlit trying to restore them.
    text = text.replace(', key="live_stt_convert_button"', "")
    text = text.replace(", key='live_stt_convert_button'", "")
    text = text.replace(', key="live_stt_clear_button"', "")
    text = text.replace(", key='live_stt_clear_button'", "")

    if text != original:
        page.write_text(text, encoding="utf-8")
        print("Removed unsafe live STT button keys.")
    else:
        print("Live page button keys already OK.")


def main() -> None:
    root = Path.cwd()
    inner = root / ROOT_EXPECTED

    if not inner.exists():
        raise SystemExit(
            "Run this script from:\n"
            r"C:\Users\lenovo\Downloads\final_4_models_integration_CODE_ONLY"
        )

    copy_file(SRC_STATE, root / DST_STATE)
    copy_file(SRC_EMERGENCY, root / DST_EMERGENCY)

    clean_json_state(root / STATE_JSON)
    clean_pickle_state(root / STATE_PICKLE)
    patch_live_button_keys(root)

    print("\nDONE ✅ Emergency instant speech fixed and live STT button-state error fixed.")
    print("\nRestart Streamlit after applying.")
    print("\nAPI server should still be running for model TTS, but Emergency will speak instantly with browser speech even if the API is slow.")
    print("\nDo NOT git push if deployment should stay unchanged.")


if __name__ == "__main__":
    main()
