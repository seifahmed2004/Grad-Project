from __future__ import annotations

import shutil
from pathlib import Path
from datetime import datetime


def backup_file(path: Path) -> None:
    if path.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = path.with_suffix(path.suffix + f".before_two_server_restore_{ts}.bak")
        shutil.copy2(path, backup)
        print(f"Backup: {backup}")


def main() -> None:
    root = Path.cwd()
    expected = root / "final_4_models_integration_CODE_ONLY"
    if not expected.exists():
        raise SystemExit(
            "Run this script from:\n"
            "C:\\Users\\lenovo\\Downloads\\final_4_models_integration_CODE_ONLY"
        )

    patch_root = Path(__file__).resolve().parent

    targets = [
        (
            patch_root / "pages" / "15_live_sign_translation.py",
            expected / "pages" / "15_live_sign_translation.py",
        ),
        (
            patch_root / "sign_live_api.py",
            expected / "sign_live_api.py",
        ),
    ]

    for src, dst in targets:
        if not src.exists():
            raise FileNotFoundError(f"Missing patch source: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        backup_file(dst)
        shutil.copy2(src, dst)
        print(f"Restored: {dst}")

    print("\nDONE ✅ Two-server local live mode restored.")
    print("\nRun locally with ONLY these two terminals:\n")
    print("Terminal 1:")
    print('  conda activate grad_py310')
    print(r'  cd "C:\Users\lenovo\Downloads\final_4_models_integration_CODE_ONLY\final_4_models_integration_CODE_ONLY"')
    print("  python sign_live_api.py")
    print("\nTerminal 2:")
    print("  conda activate grad_py310")
    print(r'  cd "C:\Users\lenovo\Downloads\final_4_models_integration_CODE_ONLY\final_4_models_integration_CODE_ONLY"')
    print("  python -m streamlit run streamlit_app.py --server.fileWatcherType none --server.runOnSave false")
    print("\nOpen:")
    print("  http://localhost:8501/live_sign_translation")
    print("\nIMPORTANT: Do NOT git push if you want Streamlit Cloud deployment to stay unchanged.")


if __name__ == "__main__":
    main()
