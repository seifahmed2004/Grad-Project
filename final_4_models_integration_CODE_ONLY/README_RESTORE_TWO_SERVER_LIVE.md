# Restore Two-Server Local Live Patch

This patch restores the local live sign translation mode used before deployment:

- Streamlit server: `localhost:8501`
- FastAPI live sign server: `127.0.0.1:8000`
- No `localhost:5500`
- No Streamlit WebRTC live page
- Restores the old live page UI/buttons and calls FastAPI endpoint:
  `http://127.0.0.1:8000/api/sign-v2/single`

## Apply

Run from:

```powershell
cd "C:\Users\lenovo\Downloads\final_4_models_integration_CODE_ONLY"
python .\restore_two_server_live_patch\apply_restore_two_server_live.py
```

## Run

Terminal 1:

```powershell
conda activate grad_py310
cd "C:\Users\lenovo\Downloads\final_4_models_integration_CODE_ONLY\final_4_models_integration_CODE_ONLY"
python sign_live_api.py
```

Terminal 2:

```powershell
conda activate grad_py310
cd "C:\Users\lenovo\Downloads\final_4_models_integration_CODE_ONLY\final_4_models_integration_CODE_ONLY"
python -m streamlit run streamlit_app.py --server.fileWatcherType none --server.runOnSave false
```

Open:

```text
http://localhost:8501/live_sign_translation
```

Do not push this branch if you want the cloud deployment to stay unchanged.
