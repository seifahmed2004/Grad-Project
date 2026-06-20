from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQ = ROOT / "requirements.txt"

REQUIRED_LINES = [
    "streamlit-webrtc>=0.72,<1",
    "av>=12",
]

if REQ.exists():
    text = REQ.read_text(encoding="utf-8", errors="ignore")
    lower = text.lower()
    add = []
    for line in REQUIRED_LINES:
        pkg = line.split(">=")[0].lower()
        if pkg not in lower:
            add.append(line)
    if add:
        REQ.write_text(text.rstrip() + "\n" + "\n".join(add) + "\n", encoding="utf-8")
        print("Updated requirements.txt with:", ", ".join(add))
    else:
        print("requirements.txt already has WebRTC dependencies.")
else:
    REQ.write_text("streamlit>=1.40,<2\n" + "\n".join(REQUIRED_LINES) + "\n", encoding="utf-8")
    print("Created requirements.txt with Streamlit WebRTC dependencies.")

print("Online live no-API patch is ready.")
print("Re-run Streamlit: python -m streamlit run streamlit_app.py --server.fileWatcherType none --server.runOnSave false")
