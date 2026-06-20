# Ishara Streamlit Website Upgrade

This folder adds:

- Login and signup pages.
- Secure salted password hashing using Python PBKDF2.
- A local SQLite user database.
- A profile page that stores the user's full name, age, and gender.
- Dynamic Streamlit navigation visible only after login.
- One independent sidebar page for each model, in this order:
  1. Sign Language to Text
  2. Text to Speech
  3. Speech to Text
  4. Age Prediction
  5. Gender Detection
- A redesigned home page, profile page, responsive styling, status cards,
  validation, loading messages, result cards, and download controls.
- A separate adapter file for every model.

## Important repository status

At the time this package was created, the GitHub repository page was still empty
because the previous push had been rejected by GitHub's large-file limit.
Therefore, the exact names and signatures of the current backend inference
functions could not be inspected.

The website and account system are complete. The five files in
`model_adapters/` are the only places that still need to be connected to the
existing inference code.

## Copy into the existing project

Copy these items to the root of the current project:

- `streamlit_app.py`
- `app/`
- `assets/`
- `model_adapters/`
- `pages/`
- `.streamlit/`

Do not delete the existing:

- `backend/`
- `models/`
- model checkpoints
- preprocessing or inference files

Merge `requirements_website_upgrade.txt` into the existing `requirements.txt`.

Add the contents of `.gitignore_additions.txt` to the existing `.gitignore`.

## Connect each model

Edit only these adapter files:

- `model_adapters/sign_language.py`
- `model_adapters/text_to_speech.py`
- `model_adapters/speech_to_text.py`
- `model_adapters/age_prediction.py`
- `model_adapters/gender_detection.py`

Each adapter contains the required input and output structure.

Example:

```python
from backend.age_prediction.inference import predict

def predict_age(image_path):
    raw_result = predict(str(image_path))
    return {
        "age": raw_result["age"],
        "age_group": raw_result["age_group"],
        "confidence": raw_result.get("confidence"),
    }
```

Do not load large models on every button click. Put the model loader in the
backend and decorate it with `@st.cache_resource`, or cache a thin adapter
loader.

## Run

From the project root:

```powershell
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Local database

The application creates:

```text
data/ishara_users.db
```

This is suitable for the local graduation-project demo. Keep the database out
of Git because it contains user account information.

For a public production deployment, replace SQLite with a managed database and
use HTTPS, reset-password emails, email verification, rate limiting, and secure
server-side secrets.

## Expected model files

The website labels use these known checkpoint names:

```text
models/asl_landmark_best_v3.pth
models/best_tts_acoustic.pth
models/vocoder_best.pt
models/best_speech_model.pth
models/best_age_efficientnet_b4_finetuned.pth
models/best_gender_utkface.pth
models/yolov8n-face-lindevs.pt
```
