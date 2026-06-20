# Fast Persistent Text to Speech Page Patch

This patch changes the Text to Speech page to use the same FastAPI TTS endpoint used by Live:

`http://127.0.0.1:8000/api/tts`

It also keeps:
- input text
- last generated audio
- TTS result details

when switching between pages.

Apply:

```powershell
cd "C:\Users\lenovo\Downloads\final_4_models_integration_CODE_ONLY"
python .\fast_persistent_tts_page_patch\apply_fast_persistent_tts_page.py
```

Do not git push if you want cloud deployment unchanged.
