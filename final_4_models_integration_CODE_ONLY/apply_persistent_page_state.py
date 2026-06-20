from __future__ import annotations

from pathlib import Path
from datetime import datetime
import shutil
import re


ROOT_EXPECTED = Path("final_4_models_integration_CODE_ONLY")
APP_FILE = ROOT_EXPECTED / "streamlit_app.py"
SRC_STATE_KEEPER = Path(__file__).resolve().parent / "app" / "persistent_page_state.py"
DST_STATE_KEEPER = ROOT_EXPECTED / "app" / "persistent_page_state.py"


def backup(path: Path) -> None:
    if path.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = path.with_suffix(path.suffix + f".before_persistent_state_{ts}.bak")
        shutil.copy2(path, backup_path)
        print(f"Backup: {backup_path}")


def find_set_page_config_end(text: str) -> int | None:
    start = text.find("st.set_page_config(")
    if start == -1:
        return None

    pos = start
    depth = 0
    while pos < len(text):
        ch = text[pos]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                line_end = text.find("\n", pos)
                return len(text) if line_end == -1 else line_end + 1
        pos += 1

    return None


def patch_streamlit_app(app_path: Path) -> None:
    if not app_path.exists():
        raise FileNotFoundError(f"Missing streamlit_app.py: {app_path}")

    backup(app_path)
    text = app_path.read_text(encoding="utf-8")
    original = text

    import_line = "from app.persistent_page_state import restore_persistent_state, save_persistent_state\n"
    if "from app.persistent_page_state import restore_persistent_state, save_persistent_state" not in text:
        lines = text.splitlines(True)
        insert_idx = 0
        for i, line in enumerate(lines[:120]):
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                insert_idx = i + 1
        lines.insert(insert_idx, import_line)
        text = "".join(lines)

    if "restore_persistent_state()" not in text:
        end = find_set_page_config_end(text)
        if end is not None:
            text = text[:end] + "\nrestore_persistent_state()\n" + text[end:]
        else:
            lines = text.splitlines(True)
            insert_idx = 0
            for i, line in enumerate(lines[:120]):
                stripped = line.strip()
                if stripped.startswith("import ") or stripped.startswith("from "):
                    insert_idx = i + 1
            lines.insert(insert_idx, "\nrestore_persistent_state()\n")
            text = "".join(lines)

    if "save_persistent_state()" not in text:
        pattern = re.compile(r"^([ \t]*)selected_page\.run\(\)\s*$", re.MULTILINE)
        match = pattern.search(text)
        if match:
            indent = match.group(1)
            replacement = (
                f"{indent}try:\n"
                f"{indent}    selected_page.run()\n"
                f"{indent}finally:\n"
                f"{indent}    save_persistent_state()"
            )
            text = pattern.sub(replacement, text, count=1)
        else:
            text += "\n\n# Persistent page state final save\nsave_persistent_state()\n"

    if text != original:
        app_path.write_text(text, encoding="utf-8")
        print("Patched streamlit_app.py with persistent state hooks.")
    else:
        print("streamlit_app.py already patched.")


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
    print(f"Installed: {dst}")

    patch_streamlit_app(root / APP_FILE)

    print("\nDONE - Page state persistence enabled.")
    print("\nIt preserves text inputs, selectboxes, checkboxes, sliders, generated sentences, predictions, and session values while switching pages.")
    print("File upload fields cannot be restored by browsers, but saved file paths/results can persist if your page stores them.")
    print("\nRun locally:")
    print("  conda activate grad_py310")
    print(r'  cd "C:\Users\lenovo\Downloads\final_4_models_integration_CODE_ONLY\final_4_models_integration_CODE_ONLY"')
    print("  python -m streamlit run streamlit_app.py --server.fileWatcherType none --server.runOnSave false")
    print("\nDo NOT git push if you want the deployed app unchanged.")


if __name__ == "__main__":
    main()
