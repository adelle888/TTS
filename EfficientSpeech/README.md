# EfficientSpeech Reproduction

This repository contains a reorganized and reproducible implementation of **EfficientSpeech: An On-Device Text to Speech Model**.

The project has been reorganized into four main modules:

- `Dataset/` — dataset preparation and preprocessing
- `Model/` — EfficientSpeech model implementation
- `Training/` — model training and validation
- `Evaluate/` — inference, synthesis, and model conversion

The current reproduction has been tested with the **LJSpeech** dataset and successfully completes data loading, model training, validation, and checkpoint generation.

---

## 1. Project Structure

```text
EfficientSpeech/
├── Dataset/
│   ├── datamodule.py
│   ├── prepare_align.py
│   ├── preprocess.py
│   └── preprocessor/
│
├── Model/
│   ├── model.py
│   ├── layers/
│   ├── text/
│   └── audio/
│
├── Training/
│   ├── train.py
│   └── config/
│       ├── LJSpeech/
│       │   └── preprocess.yaml
│       └── isip-preprocess.yaml
│
├── Evaluate/
│   ├── synthesize.py
│   ├── demo.py
│   └── convert.py
│
├── hifigan/
├── lexicon/
├── utils/
├── raw_data/
├── preprocessed_data/
├── lightning_logs/
├── requirements.txt
├── requirements_reproduce.txt
└── README.md
```

The directories `raw_data/`, `preprocessed_data/`, and `lightning_logs/` contain generated or local data and are excluded from Git.

---

## 2. Environment

The reproduction was tested with the following main environment:

```text
Python 3.10
PyTorch 2.7.0+cu128
TorchVision 0.22.0+cu128
TorchAudio 2.7.0+cu128
Lightning 2.6.6
TorchMetrics 1.9.0
NumPy 1.23.5
Librosa 0.11.0
SciPy 1.15.3
PyWorld 0.3.5
TGT 1.5
Einops 0.8.2
ONNX 1.17.0
ONNX Runtime 1.23.2
```

The verified GPU environment uses:

```text
NVIDIA GeForce RTX 5090
CUDA 12.8
```

The complete reproduction environment is recorded in:

```text
requirements_reproduce.txt
```

Activate the reproduction environment before running the project:

```bash
conda activate efficientspeech_test
```

---

## 3. Dataset

The current reproduction uses the **LJSpeech** dataset.

The expected original dataset structure is:

```text
raw_data/LJSpeech-1.1/
├── metadata.csv
└── wavs/
    ├── LJ001-0001.wav
    ├── LJ001-0002.wav
    └── ...
```

LJSpeech contains 13,100 utterances.

The dataset path is configured in:

```text
Training/config/LJSpeech/preprocess.yaml
```

The main paths are:

```yaml
path:
  corpus_path: "./raw_data/LJSpeech-1.1"
  lexicon_path: "lexicon/librispeech-lexicon.txt"
  raw_path: "./raw_data/LJSpeech"
  preprocessed_path: "./preprocessed_data/LJSpeech"
```

---

## 4. Dataset Preparation

Run all commands from the EfficientSpeech project root directory.

Prepare the LJSpeech wav and text files using:

```bash
python -m Dataset.prepare_align Training/config/LJSpeech/preprocess.yaml
```

After preparation, the expected structure is:

```text
raw_data/LJSpeech/
└── LJSpeech/
    ├── LJ001-0001.wav
    ├── LJ001-0001.lab
    ├── LJ001-0002.wav
    ├── LJ001-0002.lab
    └── ...
```

For the verified reproduction:

```text
WAV files: 13,100
LAB files: 13,100
```

---

## 5. Forced Alignment

EfficientSpeech preprocessing requires phoneme-level alignment information.

The aligned TextGrid files should be placed under:

```text
preprocessed_data/LJSpeech/TextGrid/LJSpeech/
```

Each utterance should have a corresponding TextGrid file, for example:

```text
preprocessed_data/LJSpeech/TextGrid/LJSpeech/LJ001-0001.TextGrid
```

The verified reproduction contains:

```text
13,100 TextGrid files
```

These alignment files are used to obtain phoneme durations during preprocessing.

---

## 6. Acoustic Feature Preprocessing

After the TextGrid alignment files are available, run:

```bash
python -m Dataset.preprocess Training/config/LJSpeech/preprocess.yaml
```

The preprocessing stage extracts:

- mel-spectrogram
- pitch
- energy
- phoneme duration
- speaker information
- dataset statistics
- training metadata
- validation metadata

The generated structure includes:

```text
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
```

The verified preprocessing result contains:

```text
Mel files:       13,100
Pitch files:     13,100
Energy files:    13,100
Duration files:  13,100

Training samples:   12,588
Validation samples:    512
```

---

## 7. DataLoader Verification

Before training, the processed dataset can be checked by loading one batch.

The verified reproduction produced:

```text
Train samples: 12588
Val samples: 512

phoneme:  [8, 90]
pitch:    [8, 90]
energy:   [8, 90]
duration: [8, 90]
mel:      [8, 701, 80]

DataLoader OK
```

This confirms that the preprocessed acoustic features can be correctly loaded by the training pipeline.

---

## 8. Model

The main EfficientSpeech implementation is located in:

```text
Model/model.py
```

Supporting model components are located under:

```text
Model/layers/
```

The simplified acoustic modeling pipeline is:

```text
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
```

HiFi-GAN is used as the vocoder for waveform generation.

In the verified default configuration, the model summary reports:

```text
EfficientSpeech trainable parameters: 266 K
HiFi-GAN non-trainable parameters:    926 K
Total parameters:                     ~1.2 M
```

---

## 9. Training

Training is implemented with PyTorch Lightning.

Run the following command from the project root directory:

```bash
python -m Training.train \
  --accelerator gpu \
  --devices 1 \
  --precision 16 \
  --max_epochs 1 \
  --batch-size 64 \
  --num_workers 4 \
  --preprocess-config Training/config/LJSpeech/preprocess.yaml \
  --verbose
```

For newer Lightning versions, `16-mixed` can also be used instead of the legacy `16` precision argument.

The verified one-epoch reproduction completed:

```text
Epochs:          1
Steps per epoch: 197
Batch size:      64
Training time:   approximately 6.4 seconds
```

on an NVIDIA GeForce RTX 5090.

---

## 10. Validation

Validation is integrated into the PyTorch Lightning training pipeline.

The current implementation performs validation every epoch.

The verified one-epoch run produced approximately:

```text
val_loss:      58.30
val_mel:        4.93
val_pitch:      1.19
val_energy:     1.19
val_duration:   4.22
```

These values are provided only as a smoke-test reference. A one-epoch run is intended to verify that the training and validation pipeline works correctly, not to reproduce final paper-level speech quality.

---

## 11. Checkpoint

PyTorch Lightning automatically saves model checkpoints under:

```text
lightning_logs/version_x/checkpoints/
```

For the verified reproduction:

```text
lightning_logs/version_3/checkpoints/epoch=0-step=197.ckpt
```

The generated checkpoint was successfully loaded with PyTorch and contained:

```text
Epoch:              0
Global step:        197
State dict entries: 263
Lightning version:  2.6.6
```

This confirms that the training state was successfully saved.

---

## 12. Import Verification

After reorganizing the original project into Dataset, Model, Training, and Evaluate modules, the main imports can be checked using:

```bash
python - <<'PY'
print("Testing Dataset...")
from Dataset.datamodule import LJSpeechDataModule
from Dataset.preprocessor.preprocessor import Preprocessor
print("Dataset OK")

print("Testing Model...")
from Model.model import EfficientSpeech
from Model.text import text_to_sequence
from Model.layers import PhonemeEncoder, MelDecoder, Phoneme2Mel
print("Model OK")

print("Testing Evaluate...")
from Evaluate.synthesize import get_lexicon_and_g2p, text2phoneme
print("Evaluate OK")

print("Testing Training...")
import Training.train
print("Training OK")

print("All imports OK!")
PY
```

Expected result:

```text
Dataset OK
Model OK
Evaluate OK
Training OK
All imports OK!
```

---

## 13. Reproduction Status

The following stages have been verified:

- Environment setup
- LJSpeech dataset preparation
- Forced-alignment data loading
- Acoustic feature preprocessing
- DataLoader
- EfficientSpeech model import
- Training
- Validation
- Checkpoint generation
- Checkpoint loading
- Reorganized module imports

The current repository therefore provides a working **training and validation reproduction pipeline** for EfficientSpeech.

---

## 14. Notes

The current reproduction focuses on verifying the training and validation pipeline.

Large generated files are not stored in Git, including:

```text
raw_data/
preprocessed_data/
lightning_logs/
*.ckpt
*.wav
```

Users should prepare the dataset and generated preprocessing files locally before training.

The original EfficientSpeech implementation has been reorganized into modular Dataset, Model, Training, and Evaluate directories to make the reproduction workflow easier to understand and maintain.
