# Fix Widget State Restore Patch

Fixes Streamlit errors like:

- Values for the widget with key `stt_recorded_audio` cannot be set using st.session_state
- Values for the widget with key `emergency_phrase_0` cannot be set using st.session_state

Cause:
The persistent page state patch was restoring widget keys that Streamlit does not allow manual assignment for:
buttons, audio_input, file_uploader, camera_input, etc.

This patch:
- replaces `app/persistent_page_state.py` with a safer version
- removes unsafe keys from saved local state files
- keeps safe values persistent:
  - text inputs
  - generated TTS audio URL
  - STT transcription result
  - live sentence if stored in state
  - profile values
  - predictions

Apply:

```powershell
cd "C:\Users\lenovo\Downloads\final_4_models_integration_CODE_ONLY"
python .\fix_widget_state_restore_patch\apply_fix_widget_state_restore.py
```

Restart Streamlit after applying.
