# Online Live Low Latency Patch

This patch replaces `pages/15_live_sign_translation.py` with a Streamlit Cloud-friendly auto live page:

- Auto prediction loop inside Streamlit, no FastAPI/localhost.
- Lower-latency WebRTC constraints.
- Stores recent frames in a processor and runs sign prediction every tick.
- Keeps live controls: Start Auto, Stop, Undo, Clear, Speak Sentence, STT.
- Fixes requirements for `streamlit-webrtc`, `av`, `mediapipe`, and OpenCV headless.

Run from repo root:

```powershell
python apply_online_live_low_latency_patch.py
```

Then:

```powershell
git add final_4_models_integration_CODE_ONLY/pages/15_live_sign_translation.py
git add final_4_models_integration_CODE_ONLY/requirements.txt
git add final_4_models_integration_CODE_ONLY/packages.txt
git commit -m "Improve online live camera latency"
git push
```

Then reboot the Streamlit app.
