# Griffin-Lim Text-to-Speech Trial

## Purpose
This folder contains an earlier Text-to-Speech training attempt using a Tacotron-style acoustic model with Griffin-Lim waveform reconstruction.

## Result
The model was able to generate mel-spectrograms and audio files, but the final audio sounded robotic and buzzy.

## Reason
The robotic sound was mainly caused by the Griffin-Lim vocoder. Griffin-Lim is a classical signal-processing method that estimates waveform phase from a mel-spectrogram. It does not learn natural speech waveform patterns, so even when the predicted mel-spectrogram is structured, the reconstructed audio can still sound robotic.

## Evidence
Included files:
- Training script
- HPC training script
- Run script
- Training summary plot
- Final attention/mel plot

Large files such as checkpoints and WAV outputs are intentionally not committed to GitHub.