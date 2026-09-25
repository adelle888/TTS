# Training

This directory contains the EfficientSpeech training entry point and configuration files.

## Structure

- `train.py`: Main training script.
- `config/`: Configuration files used by the reproduction pipeline.

## Training Command

Run training from the EfficientSpeech project root directory:

    python -m Training.train \
      --accelerator gpu \
      --devices 1 \
      --precision 16 \
      --max_epochs 1 \
      --batch-size 64 \
      --num_workers 4 \
      --preprocess-config Training/config/LJSpeech/preprocess.yaml \
      --verbose

## Verified Training Test

The reorganized project has been tested with:

- Dataset: LJSpeech
- Training samples: 12,588
- Validation samples: 512
- GPU: NVIDIA GeForce RTX 5090
- Epochs: 1
- Training steps: 197

The one-epoch training and validation test completed successfully.

A PyTorch Lightning checkpoint was generated under:

    lightning_logs/version_3/checkpoints/epoch=0-step=197.ckpt

The generated checkpoint was successfully loaded with PyTorch.

Training logs and checkpoints are excluded from Git.
