from __future__ import annotations

import re
import shutil
from pathlib import Path
from datetime import datetime


ROOT_EXPECTED = Path("final_4_models_integration_CODE_ONLY")


def backup(path: Path) -> None:
    if path.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst = path.with_suffix(path.suffix + f".before_rollback_emergency_fast_tts_{ts}.bak")
        shutil.copy2(path, dst)
        print(f"Backup: {dst}")


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing patch file: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    backup(dst)
    shutil.copy2(src, dst)
    print(f"Restored: {dst}")


def remove_page_block(text: str, filename: str) -> str:
    pattern = re.compile(
        r"\n?\s*st\.Page\(\s*\n"
        r"\s*PAGES_DIR\s*/\s*[\"']" + re.escape(filename) + r"[\"']\s*,\s*\n"
        r"(?:.|\n)*?"
        r"\s*\),\s*",
        re.MULTILINE,
    )
    return pattern.sub("", text)


def insert_after_page(text: str, target_filename: str, block: str) -> str:
    pattern = re.compile(
        r"(\s*st\.Page\(\s*\n"
        r"\s*PAGES_DIR\s*/\s*[\"']" + re.escape(target_filename) + r"[\"']\s*,\s*\n"
        r"(?:.|\n)*?"
        r"\s*\),)"
    )
    match = pattern.search(text)
    if not match:
        return text
    return text[: match.end()] + "\n" + block + text[match.end() :]


def patch_navigation(app_path: Path) -> None:
    if not app_path.exists():
        print(f"Skipped navigation patch: missing {app_path}")
        return

    backup(app_path)
    text = app_path.read_text(encoding="utf-8")
    original = text

    for filename in ["17_communication_history.py", "18_admin_dashboard.py"]:
        text = remove_page_block(text, filename)

    emergency_block = '''                st.Page(
                    PAGES_DIR / "16_emergency_mode.py",
                    title="Emergency Mode",
                    icon=":material/emergency:",
                ),'''

    if "16_emergency_mode.py" not in text:
        text = insert_after_page(text, "15_live_sign_translation.py", emergency_block)
        if "16_emergency_mode.py" not in text:
            text = insert_after_page(text, "14_gender_detection.py", emergency_block)

    text = text.replace('title="Live Communication"', 'title="Live Sign Translation"')
    text = text.replace("title='Live Communication'", "title='Live Sign Translation'")

    if text != original:
        app_path.write_text(text, encoding="utf-8")
        print("Navigation patched: Emergency Mode added, advanced pages removed.")
    else:
        print("Navigation already OK.")


def main() -> None:
    root = Path.cwd()
    inner = root / ROOT_EXPECTED
    if not inner.exists():
        raise SystemExit(
            "Run this script from:\n"
            r"C:\Users\lenovo\Downloads\final_4_models_integration_CODE_ONLY"
        )

    patch_root = Path(__file__).resolve().parent

    copy_file(patch_root / "pages" / "15_live_sign_translation.py", inner / "pages" / "15_live_sign_translation.py")
    copy_file(patch_root / "pages" / "16_emergency_mode.py", inner / "pages" / "16_emergency_mode.py")
    copy_file(patch_root / "sign_live_api.py", inner / "sign_live_api.py")

    patch_navigation(inner / "streamlit_app.py")

    print("\nDONE OK - Restored rollback live mode + Emergency + fast warmed TTS API.")
    print("\nRun Terminal 1:")
    print("  conda activate grad_py310")
    print(r'  cd "C:\Users\lenovo\Downloads\final_4_models_integration_CODE_ONLY\final_4_models_integration_CODE_ONLY"')
    print("  python sign_live_api.py")
    print("\nWhen API opens, check warmup:")
    print("  http://127.0.0.1:8000/api/sign-v2/warmup-status")
    print("  http://127.0.0.1:8000/api/tts/warmup")
    print("\nRun Terminal 2:")
    print("  conda activate grad_py310")
    print(r'  cd "C:\Users\lenovo\Downloads\final_4_models_integration_CODE_ONLY\final_4_models_integration_CODE_ONLY"')
    print("  python -m streamlit run streamlit_app.py --server.fileWatcherType none --server.runOnSave false")
    print("\nOpen:")
    print("  http://localhost:8501/live_sign_translation")
    print("\nIMPORTANT: Do NOT git push if you want the cloud deployment to stay unchanged.")


if __name__ == "__main__":
    main()
