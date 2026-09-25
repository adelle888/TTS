# Dataset

This directory contains the dataset preparation and preprocessing pipeline for EfficientSpeech.

## Structure

- `datamodule.py`: Dataset and DataLoader implementation for training and validation.
- `prepare_align.py`: Prepares LJSpeech audio and text files for forced alignment.
- `preprocess.py`: Runs acoustic feature preprocessing.
- `preprocessor/`: Contains dataset-specific preprocessing implementations.

## Dataset

The current reproduction uses the LJSpeech dataset.

Expected original dataset directory:

    raw_data/LJSpeech-1.1/

Prepared alignment input:

    raw_data/LJSpeech/

Preprocessed features:

    preprocessed_data/LJSpeech/

The preprocessing pipeline generates mel-spectrogram, pitch, energy, duration, training metadata, validation metadata, and statistical information.

The raw and generated dataset files are excluded from Git.
