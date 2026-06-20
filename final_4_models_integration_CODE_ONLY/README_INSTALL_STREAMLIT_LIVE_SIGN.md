# Streamlit Live Sign Translation Patch

Copy these files into the INNER project folder:

```text
streamlit_app.py
sign_live_api.py
pages/15_live_sign_translation.py
model_adapters/sign_language.py
backend/language_decoder.py
resources/sentence_corpus.txt
```

This keeps the important live.js settings and logic while adding the page to Streamlit.

Run Terminal 1:

```powershell
conda activate grad_py310
cd C:\Users\lenovo\Downloads\final_4_models_integration_CODE_ONLY\final_4_models_integration_CODE_ONLY
python sign_live_api.py
```

Run Terminal 2:

```powershell
conda activate grad_py310
cd C:\Users\lenovo\Downloads\final_4_models_integration_CODE_ONLY\final_4_models_integration_CODE_ONLY
python -m streamlit run streamlit_app.py
```

Open http://127.0.0.1:8501 and choose Live Sign Translation from the sidebar.
