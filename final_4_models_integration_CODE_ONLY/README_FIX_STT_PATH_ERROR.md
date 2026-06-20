# Fix STT Path Error

Fixes:

```text
'str' object has no attribute 'exists'
```

Cause: the STT adapter expects `audio_path` as a `Path`, but the page passed it as a string.

Copy:

```text
pages/12_speech_to_text.py
```

into your inner project folder:

```text
C:\Users\lenovo\Downloads\final_4_models_integration_CODE_ONLY\final_4_models_integration_CODE_ONLY\pages\12_speech_to_text.py
```

Then restart Streamlit.
