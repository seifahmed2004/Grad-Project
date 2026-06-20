# Live STT Speed + Path Error Fix

This patch fixes the Live page Speech-to-Text error:

`'str' object has no attribute 'exists'`

It also warms the STT model silently with a short generated WAV so the first real recording is faster.

## Install

Copy this patch content into the inner project folder:

`final_4_models_integration_CODE_ONLY/final_4_models_integration_CODE_ONLY`

Then run:

```powershell
conda activate grad_py310
cd "C:\Users\lenovo\Downloads\final_4_models_integration_CODE_ONLY\final_4_models_integration_CODE_ONLY"
python apply_live_stt_speed_fix.py
```

Restart Streamlit:

```powershell
python -m streamlit run streamlit_app.py --server.fileWatcherType none --server.runOnSave false
```

## What changed

- `pages/15_live_sign_translation.py` now uses a robust Path-first STT runner.
- `model_adapters/speech_to_text.py` becomes safe even if a string path is passed.
- `app/live_stt_fast.py` warms STT in the background and reuses the warmed model.
- `app/fast_preload.py` uses the same fast STT warm-up helper.
