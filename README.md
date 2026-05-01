# Grad-Project

A multi-model AI graduation project that integrates four components into one unified MVP system:

- **Sign Language to Text**
- **Text to Speech**
- **Speech to Text**
- **Age Prediction**

The project is organized as an **integration-ready Python system** where each model is wrapped in a clean interface and all components are controlled through a single entry point: `main.py`.

---

## Project Overview

This project aims to support accessibility and communication workflows using multiple AI models.

It combines:
- video-based sign language recognition
- text-to-speech generation
- speech-to-text transcription
- face-based age prediction

The repository focuses on the **MVP integration phase**, where separately trained models are turned into reusable interfaces and connected inside one runnable project.

---

## Main MVP Pipeline

The main MVP workflow used for demonstration is:

**Sign Language → Text → Speech**

### Flow
1. A sign language video is given as input
2. The sign recognition model predicts the corresponding text / gloss
3. The predicted text is passed to the TTS model
4. A speech audio file is generated as output

This provides a complete end-to-end demo pipeline for the project.

---

## Integrated Models

### 1) Sign Language to Text
- **Input:** video file
- **Output:** predicted gloss / text
- **Model file:** `asl_rgb_200_last.pth`
- **Interface:** `interfaces/sign_language_interface.py`

### 2) Text to Speech
- **Input:** text
- **Output:** generated `.wav` file
- **Model file:** `best.pt`
- **Interface:** `interfaces/tts_interface.py`

### 3) Speech to Text
- **Input:** audio file
- **Output:** predicted transcription
- **Model file:** `best_speech_model.pth`
- **Interface:** `interfaces/speech_interface.py`

### 4) Age Prediction
- **Input:** face image
- **Output:** predicted age and age group
- **Model file:** `best_age_resnet18.pth`
- **Interface:** `interfaces/age_interface.py`

---

## Project Structure

```text
Grad-Project/
├── interfaces/
│   ├── __init__.py
│   ├── sign_language_interface.py
│   ├── tts_interface.py
│   ├── speech_interface.py
│   └── age_interface.py
├── models/
│   ├── PUT_MODELS_HERE.txt
│   ├── asl_rgb_200_last.pth
│   ├── best.pt
│   ├── best_speech_model.pth
│   └── best_age_resnet18.pth
├── outputs/
├── main.py
├── requirements.txt
└── README.md
Setup

Install dependencies using:

pip install -r requirements.txt
Required Model Files

Model weight files are not included in this repository because of GitHub file size limits.

Before running the project, place the following files manually inside the models/ folder:

asl_rgb_200_last.pth
best.pt
best_speech_model.pth
best_age_resnet18.pth
Run Commands
Sign Language → Text → Speech
python main.py sign-tts --video "PATH_TO_VIDEO"
Sign Language → Text only
python main.py sign-text --video "PATH_TO_VIDEO"
Text → Speech only
python main.py text-tts --text "hello world"
Speech → Text
python main.py speech-text --audio "PATH_TO_AUDIO"
Age Prediction
python main.py age --image "PATH_TO_IMAGE"
Optional: force CPU
python main.py sign-tts --video "PATH_TO_VIDEO" --device cpu
python main.py sign-text --video "PATH_TO_VIDEO" --device cpu
python main.py text-tts --text "hello world" --device cpu
python main.py speech-text --audio "PATH_TO_AUDIO" --device cpu
python main.py age --image "PATH_TO_IMAGE" --device cpu
Outputs

Generated outputs are saved inside the outputs/ folder.

Examples:

sign_result.json
tts_result.json
speech_result.json
age_result.json
generated .wav files

This makes testing and demonstration easier.

Integration Design

Each model is wrapped in its own interface file and includes:

model loading
preprocessing
inference
postprocessing
structured output

This makes the system:

modular
reusable
easier to test
easier to demonstrate

All workflows are executed through:

main.py

So the project behaves as one integrated system rather than separate notebooks.

Why This Repository Is Integrated

This repository is considered an integrated MVP because:

all models are placed in one unified project
all models are runnable from one main entry point
the interfaces follow one consistent design
the project includes a real end-to-end pipeline:
Sign Language → Text → Speech

Additional models are also available as integrated runnable components:

Speech → Text
Age Prediction
Text → Speech
Sign → Text
Notes
The main MVP demo flow is:
Sign Language → Text → Speech
For Windows paths, always place file paths between quotes " "
Make sure all required model files are placed inside the models/ folder before running
Output quality depends on the provided model checkpoints and input quality
Limitations
Model weights are not uploaded to GitHub because of file size limits
Some models may require more memory depending on the local environment
TTS output quality depends on checkpoint quality and inference stability
Speech recognition performance depends on audio quality and recording conditions
Sign recognition performance depends on video clarity and class coverage
Graduation Project Context

This repository represents the integration phase of the graduation project MVP.

It focuses on:

converting trained models into reusable interfaces
organizing them into one clean structure
enabling execution from one main file
demonstrating at least one complete AI pipeline for presentation and submission
Recommended Environment

Recommended environment for running the project:

VS Code
Python virtual environment
local models/ folder containing the required checkpoints
Summary

This repository contains:

the integrated codebase
interface wrappers for all 4 models
a unified execution entry point
an MVP-ready workflow for demonstration

It is intended to be the main integration and demo environment for the graduation project.