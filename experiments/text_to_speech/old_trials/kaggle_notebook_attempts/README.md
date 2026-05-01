# Kaggle Text-to-Speech Notebook Attempts

## Purpose
This folder contains earlier Kaggle-based Text-to-Speech attempts developed before moving the training pipeline to the HPC environment.

## Summary
These notebooks were used to test different from-scratch TTS approaches, including Tacotron-style acoustic modeling, transformer-based ideas, Griffin-Lim audio reconstruction, and early HiFi-GAN experiments.

## Result
The early models produced mel-spectrograms and some waveform outputs, but the generated speech quality was limited. The main issue was robotic or buzzy audio caused by Griffin-Lim waveform reconstruction and limited Kaggle session time.

## Limitation
Kaggle sessions were limited in runtime, which made long TTS training difficult. Some runs stopped before the model fully converged, and checkpoint persistence was harder to manage.

## Improvement After These Attempts
The project was later moved to the HPC GPU environment using SLURM scripts. The improved version includes checkpoint saving, automatic resume, longer training, and a from-scratch HiFi-GAN vocoder to improve audio quality.

## Evidence
This folder includes previous notebooks and scripts used during experimentation. Large checkpoints and generated audio files are not committed to GitHub.