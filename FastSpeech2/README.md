FastSpeech2 Reproduction

This repository provides a reorganized and verified reproduction of FastSpeech2, based on the original implementation by ming024/FastSpeech2.

The project is organized into four main components:

FastSpeech2/
├── Dataset/                 # Dataset preparation and preprocessing
│   ├── dataset.py
│   ├── prepare_align.py
│   ├── preprocess.py
│   └── preprocessor/
│
├── Model/                   # FastSpeech2 model implementation
│   ├── model/
│   ├── transformer/
│   ├── text/
│   └── audio/
│
├── Training/                # Training scripts and configurations
│   ├── train.py
│   └── config/
│
├── Evaluate/                # Validation and synthesis
│   ├── evaluate.py
│   └── synthesize.py
│
├── utils/
├── hifigan/
├── lexicon/
├── requirements.txt
├── requirements_reproduce.txt
└── README.md
1. Environment

The reorganized code has been successfully verified with:

Python 3.10.21
PyTorch 2.7.0+cu128
CUDA 12.8
GPU: NVIDIA GeForce RTX 5090

A complete snapshot of the verified Python environment is provided in:

requirements_reproduce.txt

Install the required packages using either the original dependency list:

pip install -r requirements.txt

or the verified reproduction environment:

pip install -r requirements_reproduce.txt
2. Dataset

The current reproduction has been verified using LJSpeech.

Raw datasets are not included in this repository.

Place LJSpeech under:

raw_data/LJSpeech-1.1/
├── metadata.csv
└── wavs/

Dataset preparation, MFA alignment, and preprocessing instructions are provided in:

Dataset/README.md

The overall dataset pipeline is:

LJSpeech
   ↓
Prepare Alignment Data
   ↓
Montreal Forced Aligner
   ↓
TextGrid
   ↓
Feature Preprocessing
   ↓
Mel / Pitch / Energy / Duration
   ↓
Training Dataset
3. Model

The FastSpeech2 implementation is located under:

Model/

The verified model contains:

35,159,361 parameters

Main model pipeline:

Text / Phoneme Sequence
          ↓
        Encoder
          ↓
    Variance Adaptor
          ↓
        Decoder
          ↓
    Mel Spectrogram
          ↓
        PostNet
          ↓
Refined Mel Spectrogram

See Model/README.md for details.

4. Training

For normal training:

python Training/train.py \
  -p Training/config/LJSpeech/preprocess.yaml \
  -m Training/config/LJSpeech/model.yaml \
  -t Training/config/LJSpeech/train.yaml

A short 100-step smoke-test configuration is also provided:

python Training/train.py \
  -p Training/config/LJSpeech/preprocess.yaml \
  -m Training/config/LJSpeech/model.yaml \
  -t Training/config/LJSpeech/train_test.yaml

The smoke test is intended to verify that the complete training pipeline works correctly. It is not intended to reproduce the final performance reported in the original paper.

See Training/README.md for details.

5. Evaluation

After the 100-step smoke test, the saved checkpoint can be evaluated using:

python Evaluate/evaluate.py \
  --restore_step 100 \
  -p Training/config/LJSpeech/preprocess.yaml \
  -m Training/config/LJSpeech/model.yaml \
  -t Training/config/LJSpeech/train_test.yaml

The final verified validation result was:

Validation Step 100
Total Loss:        8.8164
Mel Loss:          3.6938
Mel PostNet Loss:  2.3898
Pitch Loss:        1.2753
Energy Loss:       1.0990
Duration Loss:     0.3585

See Evaluate/README.md for details.

6. Verified Reproduction

The reorganized project has been tested through the following pipeline:

Dataset
   ↓
Model
   ↓
Training
   ↓
Checkpoint
   ↓
Evaluate

Verified components:

Dataset loading          ✓
Model construction       ✓
Forward propagation      ✓
Loss calculation         ✓
Backward propagation     ✓
Optimizer update         ✓
Checkpoint saving        ✓
Validation               ✓
Independent evaluation   ✓

The final independent evaluation result is identical to the validation result obtained during the corresponding training run.

7. Repository Policy

Large datasets, generated features, checkpoints, logs, and local experiment outputs are not committed to the repository.

Examples include:

raw_data/
preprocessed_data/
output/
*.pth.tar
*.tar.gz
*.zip

These files should be downloaded or generated locally when required.

Acknowledgement

This reproduction is based on the original FastSpeech2 implementation:

ming024/FastSpeech2

Please refer to the original project and FastSpeech2 paper for the complete model description and full experimental settings.