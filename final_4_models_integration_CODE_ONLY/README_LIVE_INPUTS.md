# Ishara Live Input Update

Copy the included folders into the existing inner project folder and replace the
matching files.

This update adds:

- First-login age and gender prediction from one camera/uploaded face photo.
- User review and confirmation before saving predictions to the profile.
- Live camera or upload on Age Prediction.
- Live camera or upload on Gender Detection.
- Live microphone or upload on Speech to Text.
- Live webcam clip recording or upload on Sign Language to Text.
- A shared cached AgeGenderInterface so the age and gender models are loaded once.

Install:

```powershell
python -m pip install -r requirements_live_inputs.txt
```

Run:

```powershell
python -m streamlit run streamlit_app.py
```

For remote deployment, browser camera and microphone access require HTTPS.
