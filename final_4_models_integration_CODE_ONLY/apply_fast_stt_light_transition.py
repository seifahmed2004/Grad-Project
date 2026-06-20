from __future__ import annotations

from pathlib import Path
from datetime import datetime
import shutil
import re


ROOT_EXPECTED = Path("final_4_models_integration_CODE_ONLY")
INNER_APP = ROOT_EXPECTED / "streamlit_app.py"

SRC_STT_HELPER = Path(__file__).resolve().parent / "app" / "live_stt_fast.py"
DST_STT_HELPER = ROOT_EXPECTED / "app" / "live_stt_fast.py"

SRC_STT_PAGE = Path(__file__).resolve().parent / "pages" / "12_speech_to_text.py"
DST_STT_PAGE = ROOT_EXPECTED / "pages" / "12_speech_to_text.py"

STT_ADAPTER = ROOT_EXPECTED / "model_adapters" / "speech_to_text.py"


def backup(path: Path) -> None:
    if path.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst = path.with_suffix(path.suffix + f".before_fast_stt_light_transition_{ts}.bak")
        shutil.copy2(path, dst)
        print(f"Backup: {dst}")


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing patch source: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    backup(dst)
    shutil.copy2(src, dst)
    print(f"Installed: {dst}")


def patch_stt_adapter(root: Path) -> None:
    adapter = root / STT_ADAPTER
    if not adapter.exists():
        print(f"Skipped adapter patch: missing {adapter}")
        return

    text = adapter.read_text(encoding="utf-8")
    original = text
    backup(adapter)

    if "from pathlib import Path" not in text:
        if "from __future__ import annotations" in text:
            text = text.replace(
                "from __future__ import annotations\n",
                "from __future__ import annotations\n\nfrom pathlib import Path\n",
                1,
            )
        else:
            text = "from pathlib import Path\n" + text

    if "audio_path = Path(audio_path)" not in text:
        pattern = re.compile(r"(def\s+transcribe_audio\s*\([^\)]*audio_path[^\)]*\)\s*[^:]*:\n)")
        match = pattern.search(text)
        if match:
            text = text[:match.end()] + "    audio_path = Path(audio_path)\n" + text[match.end():]

    if text != original:
        adapter.write_text(text, encoding="utf-8")
        print(f"Patched: {adapter}")
    else:
        print("STT adapter already OK.")


def lighten_transitions(app_path: Path) -> None:
    if not app_path.exists():
        print(f"Skipped transition patch: missing {app_path}")
        return

    text = app_path.read_text(encoding="utf-8")
    original = text
    backup(app_path)

    # Premium transition hook: make it lighter.
    text = re.sub(
        r"inject_premium_page_transitions\s*\([^\)]*\)",
        'inject_premium_page_transitions(effect="vapor", intensity="soft", duration_ms=460)',
        text,
        count=1,
        flags=re.DOTALL,
    )

    # If smooth transition exists, leave it but make sure premium is not epic.
    text = text.replace("intensity=\"epic\"", "intensity=\"soft\"")
    text = text.replace("intensity='epic'", "intensity='soft'")
    text = text.replace("duration_ms=1080", "duration_ms=460")
    text = text.replace("duration_ms=1200", "duration_ms=520")
    text = text.replace("duration_ms=900", "duration_ms=460")

    if text != original:
        app_path.write_text(text, encoding="utf-8")
        print("Transitions made lighter in streamlit_app.py.")
    else:
        print("No transition hook found or already light.")


def patch_premium_css(root: Path) -> None:
    transition_file = root / ROOT_EXPECTED / "app" / "premium_page_transition.py"
    if not transition_file.exists():
        return

    text = transition_file.read_text(encoding="utf-8")
    original = text
    backup(transition_file)

    # Reduce overlay strength and remove heavy button rerun pulse feeling.
    text = text.replace('if intensity == "soft":\n        blur = 14\n        mist_opacity = 0.52\n        scale_from = 0.992\n        y_from = 16',
                        'if intensity == "soft":\n        blur = 7\n        mist_opacity = 0.24\n        scale_from = 0.997\n        y_from = 6')
    text = text.replace("startRerunPulse();", "// startRerunPulse disabled for lighter local demo;")
    text = text.replace("const TRANSITION_MS = 650;", "const TRANSITION_MS = 300;")
    text = text.replace("const COOLDOWN_MS = 900;", "const COOLDOWN_MS = 420;")

    if text != original:
        transition_file.write_text(text, encoding="utf-8")
        print("Premium transition CSS/JS made lighter.")


def main() -> None:
    root = Path.cwd()
    inner = root / ROOT_EXPECTED

    if not inner.exists():
        raise SystemExit(
            "Run this script from:\n"
            r"C:\Users\lenovo\Downloads\final_4_models_integration_CODE_ONLY"
        )

    copy_file(SRC_STT_HELPER, root / DST_STT_HELPER)
    copy_file(SRC_STT_PAGE, root / DST_STT_PAGE)
    patch_stt_adapter(root)

    lighten_transitions(root / INNER_APP)
    patch_premium_css(root)

    print("\nDONE ✅ Fast persistent STT page installed and transitions made lighter.")
    print("\nWhat changed:")
    print("- Speech to Text uses warmed app.live_stt_fast helper.")
    print("- Audio + transcript stay when switching pages.")
    print("- Repeated same audio uses cache and returns instantly.")
    print("- Page transition intensity reduced to soft and shorter duration.")
    print("\nRun:")
    print("  conda activate grad_py310")
    print(r'  cd "C:\Users\lenovo\Downloads\final_4_models_integration_CODE_ONLY\final_4_models_integration_CODE_ONLY"')
    print("  python -m streamlit run streamlit_app.py --server.fileWatcherType none --server.runOnSave false")
    print("\nDo NOT git push if you want deployment unchanged.")


if __name__ == "__main__":
    main()
