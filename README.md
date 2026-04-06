# Grad-Project

Multi-model MVP integration project including:
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