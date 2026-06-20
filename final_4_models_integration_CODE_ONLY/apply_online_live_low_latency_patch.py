from __future__ import annotations

from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
APP = ROOT / "final_4_models_integration_CODE_ONLY"
if not APP.exists():
    # When script is executed from a copied patch folder inside project root
    if (ROOT.parent / "final_4_models_integration_CODE_ONLY").exists():
        APP = ROOT.parent / "final_4_models_integration_CODE_ONLY"
    else:
        raise SystemExit("Could not find final_4_models_integration_CODE_ONLY folder. Run from the repo root.")

SRC_PAGE = ROOT / "pages" / "15_live_sign_translation.py"
DST_PAGE = APP / "pages" / "15_live_sign_translation.py"
if not SRC_PAGE.exists():
    # Script may be copied directly to root with pages folder next to it
    SRC_PAGE = Path(__file__).resolve().parent / "pages" / "15_live_sign_translation.py"
if not SRC_PAGE.exists():
    raise SystemExit("Patch page file not found: pages/15_live_sign_translation.py")

DST_PAGE.parent.mkdir(parents=True, exist_ok=True)
if DST_PAGE.exists():
    backup = DST_PAGE.with_suffix(".py.backup_before_low_latency_live")
    shutil.copy2(DST_PAGE, backup)
shutil.copy2(SRC_PAGE, DST_PAGE)
print(f"Updated: {DST_PAGE}")

# Fix requirements cleanly without concatenating package names.
REQ = APP / "requirements.txt"
REQ.parent.mkdir(parents=True, exist_ok=True)
lines = []
if REQ.exists():
    raw = REQ.read_text(encoding="utf-8", errors="ignore")
    raw = raw.replace("opencv-python-headlesspillow", "opencv-python-headless\nPillow")
    raw = raw.replace("opencv-python\n", "opencv-python-headless\n")
    raw = raw.replace("opencv-contrib-python\n", "opencv-python-headless\n")
    lines = [line.strip() for line in raw.splitlines() if line.strip()]

required = [
    "opencv-python-headless",
    "Pillow",
    "streamlit-webrtc>=0.72,<1",
    "av>=12",
    "mediapipe==0.10.14",
]
# Remove conflicting OpenCV packages and duplicate mediapipe/av/webrtc entries.
clean = []
for line in lines:
    low = line.lower()
    if low in {"opencv-python", "opencv-contrib-python"}:
        continue
    if low.startswith("streamlit-webrtc") or low == "av" or low.startswith("av>") or low.startswith("mediapipe"):
        continue
    if low == "opencv-python-headlesspillow":
        continue
    clean.append(line)
for pkg in required:
    if not any(x.lower() == pkg.lower() for x in clean):
        clean.append(pkg)
REQ.write_text("\n".join(clean) + "\n", encoding="utf-8")
print(f"Updated: {REQ}")

PKG = APP / "packages.txt"
pkg_lines = []
if PKG.exists():
    pkg_lines = [line.strip() for line in PKG.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
for pkg in ["libgl1", "libglib2.0-0", "ffmpeg"]:
    if pkg not in pkg_lines:
        pkg_lines.append(pkg)
PKG.write_text("\n".join(pkg_lines) + "\n", encoding="utf-8")
print(f"Updated: {PKG}")

print("Done. Commit and push these files, then reboot Streamlit Cloud.")
