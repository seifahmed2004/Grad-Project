# Streamlit WebRTC requirements fix

This patch adds the missing Python packages needed by the online live camera page:

```txt
streamlit-webrtc>=0.72,<1
av>=12
```

## Run locally from your project folder

```powershell
conda activate grad_py310
cd "C:\Users\lenovo\Downloads\final_4_models_integration_CODE_ONLY\final_4_models_integration_CODE_ONLY"
python apply_streamlit_webrtc_requirements_fix.py
```

Then commit and push the changed `requirements.txt` file(s) to GitHub and reboot/redeploy the app on Streamlit Community Cloud.
