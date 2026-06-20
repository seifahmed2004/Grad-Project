# Ishara Online Live No-API Patch

This patch replaces the Live Sign Translation page so it no longer depends on `sign_live_api.py` or `http://127.0.0.1:8000`.

It uses `streamlit-webrtc` inside Streamlit to capture camera frames and runs the sign model directly through `model_adapters.sign_language.predict_single_sign`.

## Files

- `pages/15_live_sign_translation.py`
- `apply_online_live_no_api_patch.py`

## Install / apply

Copy the patch files into your project root, then run:

```powershell
conda activate grad_py310
cd "C:\Users\lenovo\Downloads\final_4_models_integration_CODE_ONLY\final_4_models_integration_CODE_ONLY"
python apply_online_live_no_api_patch.py
```

Then run Streamlit only:

```powershell
python -m streamlit run streamlit_app.py --server.fileWatcherType none --server.runOnSave false
```

For Streamlit Cloud, commit these files and make sure `requirements.txt` includes:

```text
streamlit-webrtc>=0.72,<1
av>=12
```

## Important

You no longer need to run:

```powershell
python sign_live_api.py
```

The camera page should be opened through HTTPS on Streamlit Cloud, or through `localhost` when running locally.
