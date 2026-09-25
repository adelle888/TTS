# Model

This directory contains the EfficientSpeech model implementation and its supporting modules.

## Structure

- `model.py`: Main EfficientSpeech model implemented with PyTorch Lightning.
- `layers/`: EfficientSpeech neural network and acoustic model components.
- `text/`: Text normalization and phoneme processing utilities.
- `audio/`: Audio processing and STFT utilities.

## Model Pipeline

The main processing pipeline is:

    Text
      |
      v
    Phoneme Sequence
      |
      v
    Phoneme Encoder
      |
      v
    Pitch / Energy / Duration Prediction
      |
      v
    Mel Decoder
      |
      v
    Mel-Spectrogram

HiFi-GAN is used as the vocoder for waveform generation.
