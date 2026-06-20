from __future__ import annotations

from pathlib import Path
from datetime import datetime
import shutil
import json
import pickle


ROOT_EXPECTED = Path("final_4_models_integration_CODE_ONLY")
SRC_STATE_KEEPER = Path(__file__).resolve().parent / "app" / "persistent_page_state.py"
DST_STATE_KEEPER = ROOT_EXPECTED / "app" / "persistent_page_state.py"
STATE_DIR = ROOT_EXPECTED / ".ishara_local_state"
STATE_JSON = STATE_DIR / "session_state_snapshot.json"
STATE_PICKLE = STATE_DIR / "session_state_snapshot.pkl"

UNSAFE_EXACT_KEYS = {
    "stt_recorded_audio",
    "stt_uploaded_audio",
    "clear_persistent_page_state_button",
}

UNSAFE_PREFIXES = (
    "_",
    "$",
    "FormSubmitter",
    "emergency_phrase_",
)

UNSAFE_KEY_PARTS = (
    "file_uploader",
    "uploaded_file",
    "camera_input",
    "audio_input",
    "recorded_audio",
    "download_button",
    "form_submit",
)


def is_safe_key(key: str) -> bool:
    key = str(key)

    if key in UNSAFE_EXACT_KEYS:
        return False

    if key.startswith(UNSAFE_PREFIXES):
        return False

    if key.startswith("btn_") or key.startswith("button_"):
        return False

    if any(part in key.lower() for part in UNSAFE_KEY_PARTS):
        return False

    return True


def backup(path: Path) -> None:
    if path.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = path.with_suffix(path.suffix + f".before_widget_state_fix_{ts}.bak")
        shutil.copy2(path, backup_path)
        print(f"Backup: {backup_path}")


def clean_json(path: Path) -> None:
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
            print(f"Cleaned JSON state. Removed keys: {removed}")
    except Exception as exc:
        print(f"Could not clean JSON state, deleting it instead: {exc}")
        try:
            path.unlink()
        except Exception:
            pass


def clean_pickle(path: Path) -> None:
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
            print(f"Cleaned pickle state. Removed keys: {removed}")
    except Exception as exc:
        print(f"Could not clean pickle state, deleting it instead: {exc}")
        try:
            path.unlink()
        except Exception:
            pass


def main() -> None:
    root = Path.cwd()
    inner = root / ROOT_EXPECTED

    if not inner.exists():
        raise SystemExit(
            "Run this script from:\n"
            r"C:\Users\lenovo\Downloads\final_4_models_integration_CODE_ONLY"
        )

    dst = root / DST_STATE_KEEPER
    dst.parent.mkdir(parents=True, exist_ok=True)

    backup(dst)
    shutil.copy2(SRC_STATE_KEEPER, dst)
    print(f"Installed fixed persistent state keeper: {dst}")

    clean_json(root / STATE_JSON)
    clean_pickle(root / STATE_PICKLE)

    print("\nDONE ✅ Widget-state restore error fixed.")
    print("\nNow these keys will NOT be restored manually:")
    print("- stt_recorded_audio")
    print("- stt_uploaded_audio")
    print("- emergency_phrase_* buttons")
    print("- file/audio/camera/button-like widget keys")
    print("\nYour text inputs, generated speech, transcript result, live sentence, profile values, predictions, etc. will still persist.")
    print("\nRestart Streamlit after this patch.")
    print("\nDo NOT git push if you want deployment unchanged.")


if __name__ == "__main__":
    main()
