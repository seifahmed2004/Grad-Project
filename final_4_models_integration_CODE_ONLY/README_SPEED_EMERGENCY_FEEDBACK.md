# Ishara Speed + Emergency + Feedback Patch

This patch does four things:

1. Makes page navigation faster by disabling Streamlit file watching.
2. Starts a silent background preload after login for:
   - Text to Speech
   - Speech to Text
   - Age Prediction
   - Gender Detection
   - Sign model
3. Replaces the slow model pages with fast-loading pages that only import heavy model code after pressing the action button.
4. Adds:
   - Emergency Mode
   - Feedback page

## Install

Copy all files into the inner project folder:

```powershell
C:\Users\lenovo\Downloads\final_4_models_integration_CODE_ONLY\final_4_models_integration_CODE_ONLY
```

Then run:

```powershell
conda activate grad_py310
cd "C:\Users\lenovo\Downloads\final_4_models_integration_CODE_ONLY\final_4_models_integration_CODE_ONLY"
python apply_speed_emergency_feedback_patch.py
```

Run Streamlit like this:

```powershell
python -m streamlit run streamlit_app.py --server.fileWatcherType none --server.runOnSave false
```

For live camera/TTS, run the live API in a second terminal:

```powershell
python sign_live_api.py
```

Open with localhost / 127.0.0.1 for camera access.
