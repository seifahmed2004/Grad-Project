from pathlib import Path

REQUIRED_LINES = [
    "streamlit-webrtc>=0.72,<1",
    "av>=12",
]

ROOT = Path.cwd()

candidate_paths = [
    ROOT / "requirements.txt",
    ROOT / "final_4_models_integration_CODE_ONLY" / "requirements.txt",
    ROOT / "final_4_models_integration_CODE_ONLY" / "final_4_models_integration_CODE_ONLY" / "requirements.txt",
]

existing = [p for p in candidate_paths if p.exists()]

# If no requirements file exists, create one next to the common Streamlit entrypoint.
if not existing:
    for p in candidate_paths:
        if (p.parent / "streamlit_app.py").exists() or p.parent == ROOT:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("", encoding="utf-8")
            existing.append(p)
            break

if not existing:
    p = ROOT / "requirements.txt"
    p.write_text("", encoding="utf-8")
    existing.append(p)

for req_path in existing:
    text = req_path.read_text(encoding="utf-8", errors="ignore")
    lines = [line.strip() for line in text.splitlines()]
    normalized = {line.lower().split("==")[0].split(">=")[0].split("<")[0].strip() for line in lines if line.strip() and not line.strip().startswith("#")}

    additions = []
    for required in REQUIRED_LINES:
        package_name = required.lower().split(">=")[0].split("==")[0].split("<")[0].strip()
        if package_name not in normalized:
            additions.append(required)

    if additions:
        new_text = text.rstrip() + "\n" + "\n".join(additions) + "\n"
        req_path.write_text(new_text, encoding="utf-8")
        print(f"Updated: {req_path}")
        for item in additions:
            print(f"  + {item}")
    else:
        print(f"Already OK: {req_path}")

print("\nDone. Commit and push the updated requirements.txt file(s) to GitHub, then reboot/redeploy Streamlit Cloud.")
