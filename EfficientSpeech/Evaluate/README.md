# Evaluate

This directory contains the EfficientSpeech inference, synthesis, and model conversion utilities.

## Structure

- `synthesize.py`: Text and phoneme processing utilities used for speech synthesis.
- `demo.py`: EfficientSpeech inference and demonstration script.
- `convert.py`: Model conversion utilities.

## Validation

Validation is integrated into the PyTorch Lightning training pipeline.

The reorganized project has successfully completed a one-epoch training and validation test using LJSpeech.

The validation pipeline reports mel, pitch, energy, duration, and total validation losses.

Generated audio files, model checkpoints, and evaluation outputs are excluded from Git.
