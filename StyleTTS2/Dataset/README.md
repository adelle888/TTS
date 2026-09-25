# Dataset

This directory contains the dataset loading and preprocessing components used by the StyleTTS2 reproduction pipeline.

## Contents

- `meldataset.py`: audio loading, mel-spectrogram extraction, dataset definition, and DataLoader construction.
- `Data/train_list.txt`: training metadata for LJSpeech.
- `Data/val_list.txt`: validation metadata for LJSpeech.
- `Data/OOD_texts.txt`: out-of-distribution text samples used during training.

## Dataset

The reproduction uses the LJSpeech dataset.

The audio directory is specified by `root_path` in the training configuration:

```yaml
data_params:
  root_path: "/path/to/LJSpeech-1.1/wavs"

The raw LJSpeech audio files are not included in this repository.
Verification
The dataset pipeline has been verified by successfully constructing a DataLoader and loading training batches.
The test configuration produced:
- 12,500 training samples
- 100 validation samples
- 80-bin mel-spectrogram features
