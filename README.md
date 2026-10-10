# TTS Reproduction Repository

A unified repository for reproducing and validating multiple Text-to-Speech (TTS) models.

## Supported Models

- FastSpeech2
- EfficientSpeech
- StyleTTS2

The purpose of this repository is to reorganize different TTS projects into a common structure and provide a minimal reproducible workflow:

```text
Dataset → Model → Training → Checkpoint → Validation
```

The current goal is functional reproduction rather than reproducing every experiment and metric reported in the original papers.

---

## Repository Structure

```text
TTS/
├── common/
│   └── utils.py
│
├── configs/
│   ├── fastspeech2.yaml
│   ├── efficientspeech.yaml
│   └── styletts2.yaml
│
├── datasets/
│   ├── __init__.py
│   ├── fastspeech2.py
│   ├── efficientspeech.py
│   └── styletts2.py
│
├── models/
│   ├── fastspeech2.py
│   ├── efficientspeech.py
│   ├── styletts2.py
│   └── styletts2_modules/
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
├── requirements.txt
├── .gitignore
└── README.md
```

Each model follows the same organization:

- `datasets/` — dataset loading
- `models/` — model implementation
- `configs/` — model and dataset configuration
- `training/` — training scripts
- `validation/` — validation scripts
- `common/` — shared utilities

---

## Environment

The repository has been tested with:

```text
Python 3.10
PyTorch 2.7.0
CUDA 12.8
NVIDIA GeForce RTX 5090
```

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## Dataset

The current experiments use the LJSpeech dataset.

Audio files are stored locally and are not included in this repository.

Expected audio path:

```text
data/LJSpeech-1.1/wavs
```

StyleTTS2 metadata is currently read from the local upstream StyleTTS2 data files:

```text
StyleTTS2/Dataset/Data/train_list.txt
StyleTTS2/Dataset/Data/val_list.txt
StyleTTS2/Dataset/Data/OOD_texts.txt
```

Dataset paths can be changed in the corresponding YAML files under `configs/`.

---

## FastSpeech2

The unified FastSpeech2 pipeline includes:

- dataset loading
- text encoding
- duration prediction
- pitch prediction
- energy prediction
- variance adaptation
- mel-spectrogram prediction
- PostNet
- loss computation
- training
- checkpoint generation
- validation

The minimal training and validation workflow has been successfully tested.

---

## EfficientSpeech

The unified EfficientSpeech implementation includes:

- dataset loading
- acoustic model
- duration prediction
- pitch prediction
- energy prediction
- mel-spectrogram prediction
- training
- checkpoint generation
- validation

The minimal training and validation workflow has been successfully tested.

---

## StyleTTS2

StyleTTS2 requires more components than FastSpeech2 and EfficientSpeech.

The required source modules have been migrated into:

```text
models/styletts2_modules/
```

The unified implementation contains the components required for the current reproduction workflow, including:

- text encoder
- style encoder
- decoder
- text aligner
- pitch extractor
- PLBERT
- diffusion modules
- discriminators
- STFT loss

The model implementation itself was verified without relying on imports from the original StyleTTS2 source tree.

---

## StyleTTS2 Pretrained Weights

Several pretrained components are required locally.

Expected paths include:

```text
models/styletts2_modules/Utils/ASR/epoch_00080.pth
models/styletts2_modules/Utils/JDC/bst.t7
models/styletts2_modules/Utils/PLBERT/step_1000000.t7
```

These large pretrained weight files are excluded from Git.

StyleTTS2 may also use:

```text
microsoft/wavlm-base-plus
```

which must be available locally or downloaded through the corresponding model library.

---

## StyleTTS2 Training Example

A minimal 10-step training test can be run with:

```bash
python training/styletts2.py \
    --steps 10 \
    --batch-size 2 \
    --num-workers 0 \
    --log-interval 1 \
    --save-interval 10
```

A successful test produced:

```text
Step 10 | STFT Loss 1.048278
```

and generated:

```text
outputs/styletts2/checkpoints/step_10.pt
```

---

## StyleTTS2 Validation Example

```bash
python validation/styletts2.py \
    --checkpoint outputs/styletts2/checkpoints/step_10.pt \
    --batch-size 2 \
    --num-workers 0 \
    --max-batches 2
```

Example result:

```text
STFT / Mel Loss : 1.013630
Batches         : 2
Checkpoint      : step 10
[OK] StyleTTS2 validation finished
```

The exact loss may vary between runs.

---

## Reproduction Status

| Model | Dataset | Model | Training | Checkpoint | Validation |
|---|---|---|---|---|---|
| FastSpeech2 | Yes | Yes | Yes | Yes | Yes |
| EfficientSpeech | Yes | Yes | Yes | Yes | Yes |
| StyleTTS2 | Yes | Yes | Yes | Yes | Yes |

All three models have completed the required minimal training and validation workflow.

---

## Git Policy

Large datasets, generated checkpoints, training outputs, and local upstream repositories are not tracked by Git.

Examples include:

```text
data/
outputs/
FastSpeech2/
EfficientSpeech/
StyleTTS2/
```

Large pretrained StyleTTS2 weights are also excluded.

This keeps the repository focused on the unified implementation and experiment configuration.

---

## Scope

A successful minimal reproduction in this repository means that a model can:

1. load its dataset;
2. construct the model;
3. execute forward/loss computation;
4. perform training updates;
5. save a checkpoint;
6. load the checkpoint;
7. complete validation.

This does not imply full paper-level reproduction, full convergence, MOS reproduction, or reproduction of all ablation experiments.

---

## Current Status

The unified implementations of FastSpeech2, EfficientSpeech, and StyleTTS2 are functional.

The repository now provides a common foundation for subsequent TTS reproduction, model comparison, training experiments, and further development.

## StyleTTS2 Reproduction Setup

This repository provides a minimal StyleTTS2 training and validation workflow.

### External assets and dataset

Pretrained weights and datasets are not included in this repository.

Set the following environment variables before running:

    export PYTHONNOUSERSITE=1
    export STYLETTS2_ASSETS=/path/to/tts_pretrained
    export STYLETTS2_DATA=/path/to/styletts2_manifests

The pretrained assets directory should contain:

    ASR/config.yml
    ASR/epoch_00080.pth
    JDC/bst.t7
    PLBERT/config.yml
    PLBERT/step_1000000.t7

The manifest directory should contain:

    train_list.txt
    val_list.txt
    OOD_texts.txt

LJSpeech audio files should be placed at the path configured by
`data_params.root_path` in `configs/styletts2.yaml`.

### Training

    python -m training.styletts2 --steps 10 --batch-size 2 --num-workers 0 --log-interval 1 --save-interval 10

### Validation

    python -m validation.styletts2 --checkpoint outputs/styletts2/checkpoints/step_10.pt --batch-size 2 --num-workers 0 --max-batches 2

### Minimal reproduction result

- GPU: NVIDIA GeForce RTX 5090
- Training: 10 steps
- Validation: 2 batches
- STFT / Mel Loss: 0.951130

These results confirm execution of the minimal workflow, not
full model convergence or final speech quality.

### Attribution

StyleTTS2 was developed by Yinghao Li and collaborators.
Original repository: https://github.com/yl4579/StyleTTS2

The original StyleTTS2 implementation is distributed under the MIT License.
Retain the original copyright and license notice when redistributing
derived source code.
