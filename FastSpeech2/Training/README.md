Training

This directory contains the FastSpeech2 training entry point and training configurations.

Directory Structure
Training/
├── train.py
├── config/
│   ├── LJSpeech/
│   │   ├── preprocess.yaml
│   │   ├── model.yaml
│   │   ├── train.yaml
│   │   └── train_test.yaml
│   └── ...
└── README.md
1. Verified Environment

The final reorganized FastSpeech2 code has been verified with:

Python: 3.10.21
PyTorch: 2.7.0+cu128
CUDA: 12.8
GPU: NVIDIA GeForce RTX 5090

Activate the reproduction environment:

conda activate fastspeech2_test

A complete package snapshot of the verified environment is provided in:

requirements_reproduce.txt
2. Configuration

FastSpeech2 uses three main configuration files:

Training/config/LJSpeech/preprocess.yaml
Training/config/LJSpeech/model.yaml
Training/config/LJSpeech/train.yaml

Their purposes are:

preprocess.yaml: dataset paths and acoustic preprocessing parameters.
model.yaml: FastSpeech2 architecture and vocoder configuration.
train.yaml: optimizer, checkpoint, logging and training settings.
3. Full Training

After preparing the LJSpeech dataset, start normal training from the repository root:

python Training/train.py \
  -p Training/config/LJSpeech/preprocess.yaml \
  -m Training/config/LJSpeech/model.yaml \
  -t Training/config/LJSpeech/train.yaml

The standard configuration uses a full training schedule.

4. Smoke Test

A short configuration is provided to verify that the complete training pipeline works correctly:

Training/config/LJSpeech/train_test.yaml

The smoke-test configuration uses:

step:
  total_step: 100
  log_step: 10
  synth_step: 50
  val_step: 50
  save_step: 50

Run the smoke test with:

python Training/train.py \
  -p Training/config/LJSpeech/preprocess.yaml \
  -m Training/config/LJSpeech/model.yaml \
  -t Training/config/LJSpeech/train_test.yaml
5. Training Pipeline
Dataset/dataset.py
        ↓
PyTorch DataLoader
        ↓
Model/model/FastSpeech2
        ↓
Forward Propagation
        ↓
FastSpeech2 Loss
        ↓
Backward Propagation
        ↓
Gradient Clipping
        ↓
Optimizer Update
        ↓
Checkpoint + Validation
6. Final Verified Smoke Test

After reorganizing the source code into the Dataset, Model, Training, and Evaluate directories, the complete training pipeline was run again from scratch.

Model parameters:

35,159,361

Example results from the final verified run:

Step 10/100
Total Loss: 16.6480

Step 50/100
Total Loss: 12.2434

Step 100/100
Total Loss: 10.0023

Validation results:

Validation Step 50
Total Loss: 9.8709

Validation Step 100
Total Loss: 8.8164

The short smoke test is intended to verify code execution and reproducibility. It is not intended to reproduce the final performance reported in the original FastSpeech2 paper.

7. Checkpoints

The smoke-test checkpoints are generated under:

output/ckpt/LJSpeech_test/

The final verified run generated:

50.pth.tar
100.pth.tar

Checkpoints and training outputs are not committed to Git.

Verified Status

The reorganized code successfully completed:

Dataset loading       ✓
Model construction    ✓
Forward propagation   ✓
Loss calculation      ✓
Backward propagation  ✓
Optimizer update      ✓
Validation            ✓
Checkpoint saving     ✓