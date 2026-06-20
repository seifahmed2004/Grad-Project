from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STREAMLIT_APP = ROOT / "streamlit_app.py"


def _remove_page_block(text: str, filename: str) -> str:
    pattern = re.compile(
        r"\n?\s*st\.Page\(\s*\n"
        r"\s*PAGES_DIR\s*/\s*[\"']" + re.escape(filename) + r"[\"']\s*,\s*\n"
        r"(?:.|\n)*?"
        r"\s*\),\s*",
        re.MULTILINE,
    )
    return pattern.sub("", text)


def _insert_after_block(text: str, target_filename: str, block: str) -> str:
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


def patch_streamlit_app() -> None:
    if not STREAMLIT_APP.exists():
        raise FileNotFoundError("Run this script from the inner project folder that contains streamlit_app.py")

    text = STREAMLIT_APP.read_text(encoding="utf-8")
    original = text

    backup = STREAMLIT_APP.with_suffix(".py.backup_before_speed_patch")
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")

    # Remove advanced pages we do not want in this clean app version.
    for filename in ["17_communication_history.py", "18_admin_dashboard.py"]:
        text = _remove_page_block(text, filename)

    # Import silent background preloader.
    if "from app.fast_preload import start_fast_preload" not in text:
        anchor = "from app.ui import load_global_css, render_sidebar_user_card"
        if anchor in text:
            text = text.replace(anchor, anchor + "\nfrom app.fast_preload import start_fast_preload", 1)
        else:
            # Fallback after PROJECT_ROOT setup.
            text = text.replace("restore_authenticated_user()", "from app.fast_preload import start_fast_preload\n\nrestore_authenticated_user()", 1)

    # Start preloading once the user profile is complete, without blocking page navigation.
    preload_call = """        try:\n            start_fast_preload()\n        except Exception:\n            pass\n\n"""
    if "start_fast_preload()" in text and preload_call not in text:
        marker = "    else:\n        pages = {"
        if marker in text:
            text = text.replace(marker, "    else:\n" + preload_call + "        pages = {", 1)

    # Add Emergency Mode after Gender Detection if missing.
    emergency_block = '''                st.Page(
                    PAGES_DIR / "16_emergency_mode.py",
                    title="Emergency Mode",
                    icon=":material/emergency:",
                ),'''
    if "16_emergency_mode.py" not in text:
        text = _insert_after_block(text, "14_gender_detection.py", emergency_block)

    # Add Feedback in Project section before Log out if missing.
    feedback_block = '''                st.Page(
                    PAGES_DIR / "17_feedback.py",
                    title="Feedback",
                    icon=":material/rate_review:",
                ),
'''
    if "17_feedback.py" not in text:
        logout_block = '''                st.Page(
                    logout_page,
                    title="Log out",
                    icon=":material/logout:",
                ),'''
        if logout_block in text:
            # Add before the last logout block; simplest safe behavior: first Project logout after About.
            idx = text.rfind(logout_block)
            text = text[:idx] + feedback_block + text[idx:]

    # Keep Live page title clean if another patch renamed it.
    text = text.replace('title="Live Communication"', 'title="Live Sign Translation"')
    text = text.replace("title='Live Communication'", "title='Live Sign Translation'")

    STREAMLIT_APP.write_text(text, encoding="utf-8")
    if text != original:
        print("streamlit_app.py updated successfully.")
        print("Backup:", backup)
    else:
        print("No navigation/app changes were needed.")


if __name__ == "__main__":
    patch_streamlit_app()
