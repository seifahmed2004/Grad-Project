# Live Sign Translation Full Features Patch

Copy into the inner Streamlit project folder.

Files:

- `sign_live_api.py` → project root
- `pages/15_live_sign_translation.py` → `pages/15_live_sign_translation.py`

Requirements:

- Updated `tts_inference.py` with gender + age voice selection.
- Updated `model_adapters/sign_language.py`.
- Existing `model_adapters/speech_to_text.py`.

Run two terminals:

```powershell
conda activate grad_py310
cd "C:\Users\lenovo\Downloads\final_4_models_integration_CODE_ONLY\final_4_models_integration_CODE_ONLY"
python sign_live_api.py
```

```powershell
conda activate grad_py310
cd "C:\Users\lenovo\Downloads\final_4_models_integration_CODE_ONLY\final_4_models_integration_CODE_ONLY"
python -m streamlit run streamlit_app.py
```

Open Streamlit with localhost or 127.0.0.1 for camera access.
