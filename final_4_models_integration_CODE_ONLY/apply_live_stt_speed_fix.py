from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LIVE_PAGE = ROOT / "pages" / "15_live_sign_translation.py"
STT_ADAPTER = ROOT / "model_adapters" / "speech_to_text.py"
FAST_PRELOAD = ROOT / "app" / "fast_preload.py"


def patch_live_page() -> None:
    if not LIVE_PAGE.exists():
        print("skip: pages/15_live_sign_translation.py not found")
        return

    text = LIVE_PAGE.read_text(encoding="utf-8")
    original = text

    backup = LIVE_PAGE.with_suffix(".py.backup_before_live_stt_fix")
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")

    # Start STT warm-up silently when the Live page is opened.
    warmup_block = '''\n# Fast STT warm-up: makes the first recording transcription much faster.\ntry:\n    from app.live_stt_fast import start_stt_warmup\n    start_stt_warmup()\nexcept Exception:\n    pass\n'''
    if "from app.live_stt_fast import start_stt_warmup" not in text:
        # Insert after PROJECT_ROOT/sys.path setup if available, otherwise after imports.
        marker = "if str(PROJECT_ROOT) not in sys.path:\n    sys.path.insert(0, str(PROJECT_ROOT))\n"
        if marker in text:
            text = text.replace(marker, marker + warmup_block, 1)
        else:
            # fallback: after streamlit import
            text = text.replace("import streamlit as st\n", "import streamlit as st\n" + warmup_block + "\n", 1)

    new_function = '''def _run_speech_to_text(audio_path: Path) -> Dict[str, Any]:\n    """\n    Fast and robust STT runner for the Live page.\n    Fixes the old bug where audio_path was passed as str into an adapter that calls .exists().\n    """\n    try:\n        from app.live_stt_fast import run_speech_to_text_fast, start_stt_warmup\n\n        # Keep the warm-up thread alive; this call is instant after the first time.\n        start_stt_warmup()\n        return run_speech_to_text_fast(Path(audio_path))\n\n    except Exception as exc:\n        import traceback\n        return {"ok": False, "error": str(exc), "traceback": traceback.format_exc(limit=8)}\n\n'''

    pattern = re.compile(
        r"def _run_speech_to_text\(audio_path:\s*Path\)\s*->\s*Dict\[str,\s*Any\]:\n"
        r"(?:.|\n)*?\n(?=if run_stt:)",
        re.MULTILINE,
    )
    if pattern.search(text):
        text = pattern.sub(new_function, text, count=1)
    else:
        print("warning: _run_speech_to_text block not found in live page; no function replacement done")

    LIVE_PAGE.write_text(text, encoding="utf-8")
    print("patched:", LIVE_PAGE if text != original else "live page already patched")
    print("backup:", backup)


def patch_stt_adapter() -> None:
    if not STT_ADAPTER.exists():
        print("skip: model_adapters/speech_to_text.py not found")
        return

    text = STT_ADAPTER.read_text(encoding="utf-8")
    original = text

    backup = STT_ADAPTER.with_suffix(".py.backup_before_path_fix")
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")

    if "from pathlib import Path" not in text:
        # place after __future__ if present, otherwise at top
        if "from __future__ import annotations" in text:
            text = text.replace("from __future__ import annotations\n", "from __future__ import annotations\n\nfrom pathlib import Path\n", 1)
        else:
            text = "from pathlib import Path\n" + text

    # Add audio_path conversion at the beginning of transcribe_audio if not already done.
    if "audio_path = Path(audio_path)" not in text:
        pattern = re.compile(r"(def\s+transcribe_audio\s*\([^\)]*audio_path[^\)]*\)\s*[^:]*:\n)")
        match = pattern.search(text)
        if match:
            insert = match.group(1) + "    audio_path = Path(audio_path)\n"
            text = text[:match.start()] + insert + text[match.end():]
        else:
            print("warning: def transcribe_audio(...audio_path...) not found; adapter not modified")

    STT_ADAPTER.write_text(text, encoding="utf-8")
    print("patched:", STT_ADAPTER if text != original else "STT adapter already patched")
    print("backup:", backup)


def patch_fast_preload() -> None:
    if not FAST_PRELOAD.exists():
        print("skip: app/fast_preload.py not found")
        return

    text = FAST_PRELOAD.read_text(encoding="utf-8")
    original = text

    backup = FAST_PRELOAD.with_suffix(".py.backup_before_live_stt_fix")
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")

    new_function = '''def _warm_stt() -> str:\n    try:\n        from app.live_stt_fast import warm_stt_once\n        result = warm_stt_once()\n        return "ok" if result.get("ok") else f"partial: {result.get('error')}"\n    except Exception as exc:\n        return f"skip: {exc}"\n\n'''

    pattern = re.compile(r"def _warm_stt\(\)\s*->\s*str:\n(?:.|\n)*?\n(?=def _warm_image_module)", re.MULTILINE)
    if pattern.search(text):
        text = pattern.sub(new_function, text, count=1)
    elif "def _warm_stt()" in text:
        print("warning: could not safely replace _warm_stt in app/fast_preload.py")

    FAST_PRELOAD.write_text(text, encoding="utf-8")
    print("patched:", FAST_PRELOAD if text != original else "fast_preload already patched")
    print("backup:", backup)


if __name__ == "__main__":
    patch_live_page()
    patch_stt_adapter()
    patch_fast_preload()
    print("\nDone. Restart Streamlit after this patch.")
