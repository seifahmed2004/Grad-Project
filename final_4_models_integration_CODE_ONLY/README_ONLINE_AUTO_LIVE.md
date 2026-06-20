# Online Auto Live Streamlit Patch

Restores an online/deployable live sign page without `sign_live_api.py`.

It uses Streamlit WebRTC, 1280x720 camera constraints, automatic chunk prediction, local-style live settings, Undo/Clear/Speak, and STT.

Run from repo root:

```powershell
python apply_online_auto_live_patch.py
git add final_4_models_integration_CODE_ONLY/pages/15_live_sign_translation.py final_4_models_integration_CODE_ONLY/requirements.txt final_4_models_integration_CODE_ONLY/packages.txt
git commit -m "Restore auto live translation inside Streamlit"
git push
```

Then reboot the Streamlit Cloud app.
