# Persistent Sign Language Page Patch

Fixes:
- Uploaded sign video disappears when switching pages
- Sign prediction output disappears when switching pages

What it does:
- Saves uploaded sign videos to `outputs/sign_inputs`
- Stores video path/name in `st.session_state`
- Shows saved video preview after switching pages
- Stores the last sign prediction result in `st.session_state`
- Keeps advanced settings persistent

Apply:

```powershell
cd "C:\Users\lenovo\Downloads\final_4_models_integration_CODE_ONLY"
python .\persistent_sign_page_patch\apply_persistent_sign_page.py
```

Restart Streamlit after applying.
