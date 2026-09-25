Model

This directory contains the FastSpeech2 model implementation and its acoustic processing components.

Directory Structure
Model/
├── model/
│   ├── fastspeech2.py
│   ├── loss.py
│   ├── modules.py
│   └── optimizer.py
├── transformer/
├── text/
├── audio/
└── README.md
1. FastSpeech2

The main FastSpeech2 implementation is located at:

Model/model/fastspeech2.py

The verified model contains:

35,159,361 parameters

The main model pipeline is:

Phoneme Sequence
       ↓
Encoder
       ↓
Variance Adaptor
       ↓
Decoder
       ↓
Linear Projection
       ↓
Mel Spectrogram
       ↓
PostNet
       ↓
Refined Mel Spectrogram
2. Transformer

The Transformer components are located under:

Model/transformer/

They implement the encoder, decoder, attention layers and related modules used by FastSpeech2.

3. Variance Adaptor

The FastSpeech2 variance adaptor models three important speech characteristics:

Duration
Pitch
Energy

Its simplified processing flow is:

Encoder Output
      ↓
Duration Predictor
      ↓
Length Regulator
      ↓
Pitch Predictor
      ↓
Energy Predictor
      ↓
Decoder

During training, the duration, pitch and energy features generated during dataset preprocessing are used as supervision.

4. Loss

The FastSpeech2 loss implementation is located at:

Model/model/loss.py

The training and validation pipelines report:

Total Loss
Mel Loss
Mel PostNet Loss
Pitch Loss
Energy Loss
Duration Loss
5. Text Processing

Text processing modules are located under:

Model/text/

They provide text normalization, symbol definitions, pronunciation processing and text-to-sequence conversion.

6. Audio Processing

Audio processing modules are located under:

Model/audio/

These modules provide STFT and other acoustic processing utilities used by the preprocessing and synthesis pipelines.

7. Vocoder

HiFi-GAN is used to convert predicted mel spectrograms into waveforms.

The vocoder implementation is kept as a shared module at:

hifigan/

The model pipeline is therefore:

Text
 ↓
FastSpeech2
 ↓
Mel Spectrogram
 ↓
HiFi-GAN
 ↓
Waveform

Pretrained HiFi-GAN checkpoints are not committed to the repository and should be placed locally under hifigan/ when required.

8. Model Configuration

The verified LJSpeech model configuration is:

Training/config/LJSpeech/model.yaml
Verified Status

The reorganized model implementation has been successfully imported and used in the final 100-step training and validation smoke test.