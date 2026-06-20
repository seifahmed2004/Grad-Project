from __future__ import annotations

from pathlib import Path
from datetime import datetime
import shutil
import json
import pickle


ROOT_EXPECTED = Path("final_4_models_integration_CODE_ONLY")
SRC_PAGE = Path(__file__).resolve().parent / "pages" / "10_sign_language.py"
PAGES_DIR = ROOT_EXPECTED / "pages"

STATE_JSON = ROOT_EXPECTED / ".ishara_local_state" / "session_state_snapshot.json"
STATE_PICKLE = ROOT_EXPECTED / ".ishara_local_state" / "session_state_snapshot.pkl"


def backup(path: Path) -> None:
    if path.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst = path.with_suffix(path.suffix + f".before_persistent_sign_page_{ts}.bak")
        shutil.copy2(path, dst)
        print(f"Backup: {dst}")


def find_sign_page(root: Path) -> Path:
    pages_dir = root / PAGES_DIR

    candidates = [
        pages_dir / "10_sign_language.py",
        pages_dir / "09_sign_language.py",
        pages_dir / "01_sign_language.py",
    ]

    for path in candidates:
        if path.exists():
            return path

    # Fallback: search for any page containing sign language title.
    for path in pages_dir.glob("*.py"):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
            if "sign language to text" in text or "predict_sign_language" in text:
                return path
        except Exception:
            continue

    # Default target if page does not exist yet.
    return pages_dir / "10_sign_language.py"


def clean_old_widget_state(root: Path) -> None:
    # Remove uploader widget keys from saved persistent state if present.
    unsafe_parts = ["sign_video_upload_widget", "sign_upload", "file_uploader", "uploaded_file"]

    json_path = root / STATE_JSON
    if json_path.exists():
        backup(json_path)
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            state = payload.get("state", {})
            if isinstance(state, dict):
                cleaned = {
                    k: v for k, v in state.items()
                    if not any(part in str(k).lower() for part in unsafe_parts)
                }
                removed = sorted(set(state.keys()) - set(cleaned.keys()))
                payload["state"] = cleaned
                json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"Cleaned sign uploader keys from JSON state: {removed}")
        except Exception as exc:
            print(f"Could not clean JSON state: {exc}")

    pickle_path = root / STATE_PICKLE
    if pickle_path.exists():
        backup(pickle_path)
        try:
            with pickle_path.open("rb") as f:
                data = pickle.load(f)

            if isinstance(data, dict):
                cleaned = {
                    k: v for k, v in data.items()
                    if not any(part in str(k).lower() for part in unsafe_parts)
                }
                removed = sorted(set(map(str, data.keys())) - set(map(str, cleaned.keys())))
                with pickle_path.open("wb") as f:
                    pickle.dump(cleaned, f)
                print(f"Cleaned sign uploader keys from pickle state: {removed}")
        except Exception as exc:
            print(f"Could not clean pickle state: {exc}")


def main() -> None:
    root = Path.cwd()
    inner = root / ROOT_EXPECTED

    if not inner.exists():
        raise SystemExit(
            "Run this script from:\n"
            r"C:\Users\lenovo\Downloads\final_4_models_integration_CODE_ONLY"
        )

    target_page = find_sign_page(root)
    target_page.parent.mkdir(parents=True, exist_ok=True)

    backup(target_page)
    shutil.copy2(SRC_PAGE, target_page)
    print(f"Installed persistent Sign Language page: {target_page}")

    clean_old_widget_state(root)

    print("\nDONE ✅ Sign Language to Text input/output persistence enabled.")
    print("\nWhat changed:")
    print("- uploaded sign video is saved to outputs/sign_inputs")
    print("- video preview stays when switching pages")
    print("- last sign prediction stays when switching pages")
    print("- advanced settings stay too")
    print("- only clears when you click Clear saved or upload/run a new video")
    print("\nRestart Streamlit after applying.")
    print("\nDo NOT git push if deployment should stay unchanged.")


if __name__ == "__main__":
    main()
