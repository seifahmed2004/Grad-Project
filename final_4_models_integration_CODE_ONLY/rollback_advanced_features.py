from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STREAMLIT_APP = ROOT / "streamlit_app.py"

ADVANCED_PAGE_FILES = [
    ROOT / "pages" / "16_emergency_mode.py",
    ROOT / "pages" / "17_communication_history.py",
    ROOT / "pages" / "18_admin_dashboard.py",
]
ADVANCED_EXTRA_FILES = [
    ROOT / "app" / "history_store.py",
    ROOT / "apply_feature_navigation_patch.py",
]


def _remove_page_block(text: str, filename: str) -> str:
    """Remove a Streamlit st.Page(...) block that references a specific page file."""
    # Handles the common formatted block created by the previous patch.
    pattern = re.compile(
        r"\n?\s*st\.Page\(\s*\n"
        r"\s*PAGES_DIR\s*/\s*[\"']" + re.escape(filename) + r"[\"']\s*,\s*\n"
        r"(?:.|\n)*?"
        r"\s*\),\s*",
        re.MULTILINE,
    )
    return pattern.sub("", text)


def patch_navigation() -> None:
    if not STREAMLIT_APP.exists():
        print("streamlit_app.py not found. Run this script from the inner project folder.")
        return

    text = STREAMLIT_APP.read_text(encoding="utf-8")
    original = text

    backup = STREAMLIT_APP.with_suffix(".py.backup_before_rollback_advanced")
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")

    # Remove the advanced dashboard/history/emergency pages from navigation.
    for filename in [
        "16_emergency_mode.py",
        "17_communication_history.py",
        "18_admin_dashboard.py",
    ]:
        text = _remove_page_block(text, filename)

    # Put the live page title back to the previous simple wording if it was changed.
    text = text.replace('title="Live Communication"', 'title="Live Sign Translation"')
    text = text.replace("title='Live Communication'", "title='Live Sign Translation'")

    # If the advanced patch inserted Live Communication but the old title block uses exact text,
    # this also keeps the Live page in sidebar, it only removes the extras.
    if text != original:
        STREAMLIT_APP.write_text(text, encoding="utf-8")
        print("Navigation rolled back. Backup saved to:", backup)
    else:
        print("Navigation already looks clean. Backup saved to:", backup)


def delete_advanced_files() -> None:
    for path in ADVANCED_PAGE_FILES + ADVANCED_EXTRA_FILES:
        try:
            if path.exists():
                path.unlink()
                print("Deleted:", path.relative_to(ROOT))
        except Exception as exc:
            print("Could not delete", path, "->", exc)


def main() -> None:
    patch_navigation()
    delete_advanced_files()
    print("\nDone. Restored the live page/API to the previous clean live version.")
    print("Now restart Streamlit and sign_live_api.py.")


if __name__ == "__main__":
    main()
