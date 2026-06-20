# Streamlit Sign Model Patch

This patch updates ONLY the Sign Language to Text integration in the Streamlit website.
It does not modify Text-to-Speech, Speech-to-Text, Age Prediction, or Gender Detection.

Copy files to the project root:

```text
model_adapters/sign_language.py
pages/10_sign_language.py
backend/language_decoder.py
resources/sentence_corpus.txt
```

Required existing files:

```text
interfaces/sign_language_interface.py
models/asl_landmark_best_v3.pth
models/label2id.json  # if your interface expects it
```

Run:

```powershell
conda activate grad_py310
python -m streamlit run streamlit_app.py
```

Open:

```text
http://127.0.0.1:8501
```
