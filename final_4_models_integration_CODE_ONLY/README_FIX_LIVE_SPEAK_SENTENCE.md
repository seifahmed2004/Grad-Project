# Fix Live Speak Sentence

This patch replaces the server-side Kokoro call in `pages/15_live_sign_translation.py` with a browser-native speech synthesis button.

Why: Streamlit Cloud server-side TTS can be slow or blocked by audio file generation/playback. Browser speech synthesis is instant and runs directly on the user's device.

Run from the Git repo root:

```powershell
python .\fix_live_speak_sentence_patch\apply_fix_live_speak_sentence.py
```

Then commit and push:

```powershell
git add final_4_models_integration_CODE_ONLY/pages/15_live_sign_translation.py
git commit -m "Fix live speak sentence button"
git push
```
