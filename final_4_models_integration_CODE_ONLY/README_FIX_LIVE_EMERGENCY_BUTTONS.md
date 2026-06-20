# Fix Live STT Button State + Emergency Voice Patch

Fixes:
- StreamlitValueAssignmentNotAllowedError for `live_stt_convert_button`
- Emergency Mode hanging with no sound

What it does:
- Replaces the persistent state keeper with a safer version that never restores button/audio/upload keys
- Cleans old saved local state files
- Removes unsafe explicit button keys in the live STT section
- Replaces Emergency Mode with instant browser speech + optional model voice

Apply:

```powershell
cd "C:\Users\lenovo\Downloads\final_4_models_integration_CODE_ONLY"
python .\fix_live_emergency_buttons_patch\apply_fix_live_emergency_buttons.py
```

Restart Streamlit after applying.
