from __future__ import annotations

from pathlib import Path
from datetime import datetime
import shutil


ROOT_EXPECTED = Path("final_4_models_integration_CODE_ONLY")
SRC_PAGE = Path(__file__).resolve().parent / "pages" / "11_text_to_speech.py"
DST_PAGE = ROOT_EXPECTED / "pages" / "11_text_to_speech.py"


def backup(path: Path) -> None:
    if path.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = path.with_suffix(path.suffix + f".before_fast_persistent_tts_{ts}.bak")
        shutil.copy2(path, backup_path)
        print(f"Backup: {backup_path}")


def main() -> None:
    root = Path.cwd()
    inner = root / ROOT_EXPECTED

    if not inner.exists():
        raise SystemExit(
            "Run this script from:\n"
            r"C:\Users\lenovo\Downloads\final_4_models_integration_CODE_ONLY"
        )

    dst = root / DST_PAGE
    dst.parent.mkdir(parents=True, exist_ok=True)

    backup(dst)
    shutil.copy2(SRC_PAGE, dst)

    print(f"Patched: {dst}")
    print("\nDONE ✅ Text to Speech now uses the same fast TTS API as Live.")
    print("\nWhat changed:")
    print("- no direct slow Kokoro load inside the Streamlit page")
    print("- calls http://127.0.0.1:8000/api/tts")
    print("- keeps the text input after switching pages")
    print("- keeps the last generated voice/audio after switching pages")
    print("- only clears when you press Clear saved speech or generate a new audio")
    print("\nRun Terminal 1:")
    print("  conda activate grad_py310")
    print(r'  cd "C:\Users\lenovo\Downloads\final_4_models_integration_CODE_ONLY\final_4_models_integration_CODE_ONLY"')
    print("  python sign_live_api.py")
    print("\nOptional warmup before demo:")
    print("  http://127.0.0.1:8000/api/tts/warmup")
    print("\nRun Terminal 2:")
    print("  conda activate grad_py310")
    print(r'  cd "C:\Users\lenovo\Downloads\final_4_models_integration_CODE_ONLY\final_4_models_integration_CODE_ONLY"')
    print("  python -m streamlit run streamlit_app.py --server.fileWatcherType none --server.runOnSave false")
    print("\nDo NOT git push if you want the deployed app unchanged.")


if __name__ == "__main__":
    main()
