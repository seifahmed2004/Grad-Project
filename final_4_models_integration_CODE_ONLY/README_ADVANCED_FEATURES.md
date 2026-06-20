# Ishara Advanced Features Patch

Copy these files into the inner project folder that contains `streamlit_app.py`.

New/updated files:

- `sign_live_api.py`
- `app/history_store.py`
- `pages/15_live_sign_translation.py`
- `pages/16_emergency_mode.py`
- `pages/17_communication_history.py`
- `pages/18_admin_dashboard.py`
- `.streamlit/config.toml`
- `apply_feature_navigation_patch.py`

After copying, run:

```powershell
conda activate grad_py310
cd "C:\Users\lenovo\Downloads\final_4_models_integration_CODE_ONLY\final_4_models_integration_CODE_ONLY"
python apply_feature_navigation_patch.py
```

Then start the live API:

```powershell
python sign_live_api.py
```

In another terminal start Streamlit:

```powershell
python -m streamlit run streamlit_app.py --server.fileWatcherType none --server.runOnSave false
```

Open with localhost or 127.0.0.1 for camera/microphone access.

Admin Dashboard password: `admin` by default. To change it:

```powershell
set ISHARA_ADMIN_KEY=your_password
```

Privacy note: communication history stores text and technical metadata. It does not store camera photos/videos or microphone audio in history unless the user explicitly consents.
