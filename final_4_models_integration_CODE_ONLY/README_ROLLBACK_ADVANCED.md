# Rollback Advanced Features Patch

This patch returns the live page one step back to the previous clean live version:

- Keeps Live Sign Translation
- Keeps Speak Sentence with age/gender-aware TTS
- Keeps the simple STT section
- Removes the Model Performance panel from the live page
- Removes Camera Guidance UI from the live page
- Removes the colorful advanced style and returns to the simpler previous Streamlit style
- Removes Emergency / History / Admin pages from navigation

## Install

Copy these files into the inner project folder:

```text
C:\Users\lenovo\Downloads\final_4_models_integration_CODE_ONLY\final_4_models_integration_CODE_ONLY
```

Files included:

```text
pages/15_live_sign_translation.py
sign_live_api.py
rollback_advanced_features.py
```

Then run:

```powershell
conda activate grad_py310
cd "C:\Users\lenovo\Downloads\final_4_models_integration_CODE_ONLY\final_4_models_integration_CODE_ONLY"
python rollback_advanced_features.py
```

Restart:

```powershell
python sign_live_api.py
python -m streamlit run streamlit_app.py --server.fileWatcherType none --server.runOnSave false
```
