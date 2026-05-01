# Multi-Model MVP Integration

This package integrates four model interfaces:
- Sign Language to Text
- Text to Speech
- Speech to Text
- Age Prediction

## Folder structure
- `interfaces/` all interface files
- `models/` place model checkpoints here
- `outputs/` generated outputs
- `samples/` optional test files
- `main.py` integration entry point

## Required model files
Put these files inside `models/`:
- `asl_rgb_200_last.pth`
- `best.pt`
- `best_speech_model.pth`
- `best_age_resnet18.pth`

## Run commands

### 1) Sign Language -> Text -> Speech
```bash
python main.py sign-tts --video samples/sample_video.mp4
```

### 2) Sign Language -> Text only
```bash
python main.py sign-text --video samples/sample_video.mp4
```

### 3) Text -> Speech only
```bash
python main.py text-tts --text "hello world"
```

### 4) Speech -> Text
```bash
python main.py speech-text --audio samples/sample_audio.wav
```

### 5) Age Prediction
```bash
python main.py age --image samples/sample_face.jpg
```

## Setup
```bash
pip install -r requirements.txt
```

## Notes
- Best MVP demo flow remains: `Sign Language -> Text -> Speech`
- `Speech -> Text` and `Age Prediction` are added as separate runnable integrated components.
- If you want to use CPU explicitly, add `--device cpu`
## Sign Language Inference Module

This feature supports sign-language video input and converts it into predicted text output for the MVP demo.


## Unified MVP Workflow

Current implemented system flow:

1. Sign Language Video -> Text -> Speech
2. Speech Audio -> Text
3. Face Image -> Age Prediction

All modules are accessible through the unified main.py entry point.