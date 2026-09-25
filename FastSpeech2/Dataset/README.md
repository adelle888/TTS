Dataset

This directory contains the dataset preparation and preprocessing code used for the FastSpeech2 reproduction.

Directory Structure
Dataset/
├── dataset.py
├── prepare_align.py
├── preprocess.py
├── preprocessor/
│   ├── aishell3.py
│   ├── libritts.py
│   ├── ljspeech.py
│   └── preprocessor.py
└── README.md

The current reproduction has been verified using the LJSpeech dataset.

1. LJSpeech Dataset

The verified dataset configuration is:

Dataset: LJSpeech
Language: English
Number of utterances: 13,100
Sampling rate: 22,050 Hz

The raw dataset is not included in this repository.

After downloading LJSpeech, place it under:

raw_data/LJSpeech-1.1/
├── metadata.csv
└── wavs/
2. Prepare Alignment Data

FastSpeech2 requires phoneme-level duration information.

Prepare the LJSpeech corpus for forced alignment with:

python Dataset/prepare_align.py \
  Training/config/LJSpeech/preprocess.yaml

The preprocessing configuration is located at:

Training/config/LJSpeech/preprocess.yaml
3. Montreal Forced Aligner

Montreal Forced Aligner (MFA) is used to generate phoneme-level alignments.

The reproduction environment used:

MFA version: 3.4.2
Acoustic model: english_us_arpa

A separate MFA environment is recommended:

conda create -n mfa -c conda-forge montreal-forced-aligner -y
conda activate mfa

Check the installation:

mfa version

Download the required English models:

mfa model download dictionary english_us_arpa
mfa model download acoustic english_us_arpa

Run forced alignment:

mfa align \
  raw_data/LJSpeech/LJSpeech \
  english_us_arpa \
  english_us_arpa \
  preprocessed_data/LJSpeech/TextGrid \
  --clean

The verified local dataset contains 13,100 TextGrid files.

The number of alignment files can be checked with:

find preprocessed_data/LJSpeech/TextGrid \
  -name "*.TextGrid" | wc -l
4. Feature Preprocessing

Return to the FastSpeech2 environment and run:

conda activate fastspeech2_test

python Dataset/preprocess.py \
  Training/config/LJSpeech/preprocess.yaml

The preprocessing pipeline generates:

preprocessed_data/LJSpeech/
├── duration/
├── energy/
├── mel/
├── pitch/
├── TextGrid/
├── speakers.json
├── stats.json
├── train.txt
└── val.txt

The main features are:

Mel spectrogram: acoustic target for FastSpeech2.
Pitch: pitch information used by the variance adaptor.
Energy: speech energy information.
Duration: phoneme duration obtained from forced alignment.
5. Dataset Loading

Dataset loading is implemented in:

Dataset/dataset.py

The overall data pipeline is:

LJSpeech
    ↓
Dataset/prepare_align.py
    ↓
MFA Forced Alignment
    ↓
TextGrid
    ↓
Dataset/preprocess.py
    ↓
Mel / Pitch / Energy / Duration
    ↓
train.txt / val.txt
    ↓
Dataset/dataset.py
    ↓
PyTorch DataLoader
6. Repository Policy

Raw and generated datasets are not committed to Git:

raw_data/
preprocessed_data/

Users should download and preprocess the dataset locally before full training.

Verified Status

The complete LJSpeech dataset preprocessing and loading pipeline has been successfully used for FastSpeech2 training and validation.