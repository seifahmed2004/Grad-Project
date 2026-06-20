# Persistent Page State Patch

Keeps most page values when switching between Streamlit pages.

Preserves:
- text inputs
- selectbox / radio / checkbox / slider values
- generated live sentence if stored in `st.session_state`
- profile values
- predictions/results stored in `st.session_state`

Limitation:
- Browser file upload fields cannot be restored for security reasons.
- But if the app stores uploaded file paths/results in `st.session_state`, those can persist.

Apply:

```powershell
cd "C:\Users\lenovo\Downloads\final_4_models_integration_CODE_ONLY"
python .\persistent_page_state_patch\apply_persistent_page_state.py
```

Do not git push if you want the deployed cloud app unchanged.
