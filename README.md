# Unified TTS Reproduction

A unified PyTorch reproduction workspace for training and validating three text-to-speech (TTS) models:

- FastSpeech2
- EfficientSpeech
- StyleTTS2

The three models share a unified project structure with separate model, dataset, training, validation, and configuration modules.

The goal of this repository is to provide a minimal reproducible training and validation pipeline rather than reproducing every experiment and evaluation metric from the original papers.

---

## 1. Supported Models

| Model | Training | Validation | Dataset |
|---|---|---|---|
| FastSpeech2 | Yes | Yes | LJSpeech |
| EfficientSpeech | Yes | Yes | LJSpeech |
| StyleTTS2 | Yes | Yes | LJSpeech |

All three models have been tested with short training runs and validation.

---

## 2. Project Structure

```text
TTS/
├── common/
│   ├── text.py
│   └── utils.py
│
├── configs/
│   ├── fastspeech2.yaml
│   ├── efficientspeech.yaml
│   └── styletts2.yaml
│
├── datasets/
│   ├── fastspeech2.py
│   └── efficientspeech.py
│
├── models/
│   ├── fastspeech2.py
│   ├── efficientspeech.py
│   └── styletts2.py
│
├── training/
│   ├── fastspeech2.py
│   ├── efficientspeech.py
│   └── styletts2.py
│
├── validation/
│   ├── fastspeech2.py
│   ├── efficientspeech.py
│   └── styletts2.py
│
├── FastSpeech2/
├── EfficientSpeech/
├── StyleTTS2/
│
├── outputs/
├── requirements.txt
└── README.md
```

The top-level `models`, `datasets`, `training`, `validation`, and `configs` directories provide the unified interface.

The original model repositories are retained for model-specific components, preprocessing utilities, and pretrained components required by the unified implementation.

---

## 3. Environment

The reproduction environment was tested with:

```text
Python: 3.10
PyTorch: 2.7.0
CUDA: 12.8
GPU: NVIDIA GeForce RTX 5090
```

Create a Conda environment:

```bash
conda create -n tts_test python=3.10 -y
conda activate tts_test
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Check PyTorch and CUDA:

```bash
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

---

## 4. Dataset

The current reproduction uses the LJSpeech dataset.

Large datasets and preprocessed features are not stored in this Git repository.

Expected model-specific preprocessed data directories include:

```text
FastSpeech2/preprocessed_data/LJSpeech/
EfficientSpeech/preprocessed_data/LJSpeech/
```

StyleTTS2 uses its own dataset configuration and data list format.

Before training, check the paths in:

```text
configs/fastspeech2.yaml
configs/efficientspeech.yaml
configs/styletts2.yaml
```

and modify local dataset paths when necessary.

---

## 5. Pretrained Components

Some StyleTTS2 components require pretrained weights.

The configuration expects components including:

```text
StyleTTS2/Model/Utils/ASR/
StyleTTS2/Model/Utils/JDC/
StyleTTS2/Model/Utils/PLBERT/
```

Check `configs/styletts2.yaml` and update these paths according to your local environment.

Large pretrained weights are not intended to be committed to this repository.

---

## 6. FastSpeech2

### Training

Run a short smoke test:

```bash
python -m training.fastspeech2 \
    --steps 10 \
    --batch-size 2 \
    --save-interval 10
```

A checkpoint will be written under:

```text
outputs/fastspeech2/checkpoints/
```

### Validation

```bash
python -m validation.fastspeech2 \
    --checkpoint outputs/fastspeech2/checkpoints/step_10.pt \
    --batch-size 2 \
    --max-batches 2
```

Successful execution should end with:

```text
[OK] Validation finished
```

---

## 7. EfficientSpeech

### Training

Run a short smoke test:

```bash
python -m training.efficientspeech \
    --steps 10 \
    --batch-size 2 \
    --save-interval 10
```

Checkpoint directory:

```text
outputs/efficientspeech/checkpoints/
```

### Validation

```bash
python -m validation.efficientspeech \
    --checkpoint outputs/efficientspeech/checkpoints/step_10.pt \
    --batch-size 2 \
    --max-batches 2
```

Successful execution should end with:

```text
[OK] EfficientSpeech validation finished
```

---

## 8. StyleTTS2

StyleTTS2 requires additional pretrained components such as the ASR aligner, pitch extractor, and PLBERT.

Make sure the paths in:

```text
configs/styletts2.yaml
```

are valid before training.

### Training

Run a short smoke test:

```bash
python -m training.styletts2 \
    --steps 10 \
    --batch-size 2 \
    --save-interval 10
```

Checkpoint directory:

```text
outputs/styletts2/checkpoints/
```

### Validation

```bash
python -m validation.styletts2 \
    --checkpoint outputs/styletts2/checkpoints/step_10.pt \
    --batch-size 2 \
    --max-batches 2
```

Successful execution should end with:

```text
[OK] StyleTTS2 validation finished
```

---

## 9. Reproduction Status

The following minimal training-validation pipelines have been verified:

```text
FastSpeech2
    model initialization
        -> dataset loading
        -> forward
        -> loss
        -> training
        -> checkpoint
        -> validation

EfficientSpeech
    model initialization
        -> dataset loading
        -> forward
        -> loss
        -> training
        -> checkpoint
        -> validation

StyleTTS2
    model initialization
        -> dataset loading
        -> text alignment
        -> text encoding
        -> pitch extraction
        -> style encoding
        -> waveform decoding
        -> STFT loss
        -> training
        -> checkpoint
        -> validation
```

These are minimal reproduction/smoke-test runs. A 10-step run is intended to verify that the complete training and validation pipeline works; it is not intended to reproduce final paper-level model quality.

---

## 10. Outputs

Generated training artifacts are stored under:

```text
outputs/
├── fastspeech2/
│   └── checkpoints/
├── efficientspeech/
│   └── checkpoints/
└── styletts2/
    └── checkpoints/
```

Training outputs, datasets, checkpoints, caches, and large pretrained files should not be committed to Git.

---

## 11. Quick Start

After preparing the datasets and pretrained components:

```bash
conda create -n tts_test python=3.10 -y
conda activate tts_test

pip install -r requirements.txt
```

Then test one model, for example FastSpeech2:

```bash
python -m training.fastspeech2 \
    --steps 10 \
    --batch-size 2 \
    --save-interval 10

python -m validation.fastspeech2 \
    --checkpoint outputs/fastspeech2/checkpoints/step_10.pt \
    --batch-size 2 \
    --max-batches 2
```

The same unified workflow is provided for EfficientSpeech and StyleTTS2.

---

## 12. Notes

This repository focuses on a unified minimal reproduction pipeline.

The short training runs are used to verify:

- dataset loading
- model initialization
- forward propagation
- loss computation
- backward propagation
- optimizer updates
- checkpoint saving/loading
- validation

For full-scale training, increase the number of training steps and configure the dataset and model parameters accordingly.
