# Rollback + Emergency + Fast TTS Patch

This patch restores the same rollback step:
- Removes advanced extra features from the Live page
- Restores the clean two-server Live Sign Translation page
- Adds Emergency Mode
- Speeds up Live Speak Sentence by using warmed-up Kokoro TTS inside `sign_live_api.py`
- Adds TTS output cache for repeated phrases
- Adds TTS warmup endpoint: `http://127.0.0.1:8000/api/tts/warmup`

Apply from the outer project folder:

```powershell
cd "C:\Users\lenovo\Downloads\final_4_models_integration_CODE_ONLY"
python .\rollback_emergency_fast_tts_patch\apply_rollback_emergency_fast_tts.py
```

Do not git push if you want the deployed Streamlit app to stay unchanged.
