from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP = ROOT / "final_4_models_integration_CODE_ONLY"
if not APP.exists():
    APP = ROOT

src_page = ROOT / "pages" / "15_live_sign_translation.py"
dst_page = APP / "pages" / "15_live_sign_translation.py"
dst_page.parent.mkdir(parents=True, exist_ok=True)
dst_page.write_text(src_page.read_text(encoding="utf-8"), encoding="utf-8")
print(f"Updated: {dst_page}")

req = APP / "requirements.txt"
text = req.read_text(encoding="utf-8") if req.exists() else ""
text = text.replace("opencv-python-headlesspillow", "opencv-python-headless\nPillow")
text = text.replace("opencv-python-headlessPillow", "opencv-python-headless\nPillow")

remove_prefixes = (
    "opencv-python",
    "opencv-contrib-python",
    "opencv-python-headless",
    "mediapipe",
    "numpy",
    "Pillow",
    "pillow",
    "streamlit-webrtc",
    "av",
)
lines = []
for line in text.splitlines():
    stripped = line.strip()
    if not stripped:
        continue
    low = stripped.lower()
    if any(low.startswith(p.lower()) for p in remove_prefixes):
        continue
    lines.append(stripped)

for item in [
    "numpy==1.26.4",
    "opencv-python-headless==4.10.0.84",
    "Pillow>=10",
    "mediapipe==0.10.14",
    "streamlit-webrtc>=0.72,<1",
    "av>=12",
]:
    if item not in lines:
        lines.append(item)

req.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Updated: {req}")

packages = APP / "packages.txt"
packages.write_text("libgl1\nlibglib2.0-0\nffmpeg\n", encoding="utf-8")
print(f"Updated: {packages}")

print("\nNow run:")
print("git add final_4_models_integration_CODE_ONLY/pages/15_live_sign_translation.py final_4_models_integration_CODE_ONLY/requirements.txt final_4_models_integration_CODE_ONLY/packages.txt")
print('git commit -m "Restore auto live translation inside Streamlit"')
print("git push")
