# TTS Reproduction Repository

A unified repository for reproducing, training, and validating multiple Text-to-Speech (TTS) models.

## Supported Models

- FastSpeech2
- EfficientSpeech
- StyleTTS2

The purpose of this repository is to reorganize different TTS projects into a common structure and provide a minimal reproducible workflow:

```text
Dataset → Preprocessing → Model → Training → Checkpoint → Validation
```

The current goal is functional reproduction rather than reproducing every experiment and metric reported in the original papers.

The repository provides unified dataset interfaces, model implementations, configuration files, training scripts, and validation scripts.

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
├── preprocess/
│   ├── fastspeech2/
│   │   └── Dataset/
│   │
│   └── efficientspeech/
│       └── Dataset/
│           └── preprocessor/
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
├── data/                       # Local datasets and features (Git ignored)
├── outputs/                    # Generated checkpoints (Git ignored)
│
├── requirements.txt
├── .gitignore
└── README.md
```

Each model follows the same organization:

- `preprocess/` — dataset preparation, alignment processing, and acoustic feature extraction
- `datasets/` — dataset loading and batch construction
- `models/` — model implementation
- `configs/` — model, preprocessing, and training configuration
- `training/` — training scripts
- `validation/` — validation scripts
- `common/` — shared utilities

The unified structure separates preprocessing, model implementation, training, and validation to simplify debugging and further development.

---

## Environment

The repository has been tested with:

```text
Python 3.10
PyTorch 2.7.0
CUDA 12.8
NVIDIA GeForce RTX 5090
```

The experiments were conducted using the Conda environment:

```text
tts_reproduce
```

Install dependencies with:

```bash
pip install -r requirements.txt
```

The preprocessing pipelines also require audio-processing dependencies, including:

- librosa
- NumPy
- SciPy
- pyworld
- PyYAML

Phoneme alignment is required for the alignment-dependent preprocessing pipelines of FastSpeech2 and EfficientSpeech.

Existing Montreal Forced Aligner (MFA) TextGrid results can be reused.

---

## Dataset and Preprocessing

The current experiments use the LJSpeech dataset.

Audio files and generated acoustic features are stored locally and are not included in this repository.

### LJSpeech Dataset

Expected dataset structure:

```text
data/
└── LJSpeech-1.1/
    ├── metadata.csv
    └── wavs/
        ├── LJ001-0001.wav
        ├── LJ001-0002.wav
        └── ...
```

The dataset contains speech recordings and corresponding text transcriptions.

Dataset paths can be modified in the corresponding YAML configuration files under `configs/`.

### Preprocessing Pipeline

FastSpeech2 and EfficientSpeech use an acoustic feature extraction pipeline:

```text
LJSpeech Audio + Transcription
              |
              v
      Text Normalization
              |
              v
      Phoneme Conversion
              |
              v
      Phoneme Alignment
         (MFA/TextGrid)
              |
              v
     Acoustic Feature Extraction
              |
       +------+------+------+
       |      |      |      |
       v      v      v      v
      Mel   Pitch  Energy Duration
       |      |      |      |
       +------+------+------+
              |
              v
      Preprocessed Dataset
              |
              v
       PyTorch DataLoader
              |
              v
        Model Training
```

The generated features include:

- Mel-spectrogram
- Pitch
- Energy
- Phoneme duration
- Text and phoneme sequences
- Dataset statistics
- Training and validation metadata

The preprocessing results are stored in model-specific directories.

For example:

```text
data/
├── fastspeech2/
│   ├── raw/
│   └── preprocessed/
│
└── efficientspeech/
    ├── raw/
    └── preprocessed/
```

A typical preprocessed directory contains:

```text
preprocessed/
├── mel/
├── pitch/
├── energy/
├── duration/
├── TextGrid/
├── train.txt
├── val.txt
├── stats.json
└── speakers.json
```

The `TextGrid` directory contains phoneme alignment information.

The preprocessing scripts use the dataset paths and audio parameters defined in the corresponding YAML files.

### MFA Alignment

FastSpeech2 and EfficientSpeech require phoneme duration information during training.

Phoneme durations are obtained from alignment files generated by Montreal Forced Aligner (MFA).

The expected alignment structure is:

```text
preprocessed/
└── TextGrid/
    └── LJSpeech/
        ├── LJ001-0001.TextGrid
        ├── LJ001-0002.TextGrid
        └── ...
```

The alignment files must be available before running the final feature extraction stage.

During the three-sample smoke tests, existing MFA alignment files were reused instead of running MFA again.

The alignment results are local data artifacts and are not included in Git.

---

## FastSpeech2

FastSpeech2 is a non-autoregressive TTS model that predicts acoustic features from text and phoneme representations.

The unified FastSpeech2 implementation includes:

- Dataset loading
- Text and phoneme encoding
- Encoder and decoder
- Duration prediction
- Pitch prediction
- Energy prediction
- Variance adaptation
- Mel-spectrogram prediction
- PostNet
- Loss computation
- Training
- Checkpoint generation
- Validation

### FastSpeech2 Configuration

Configuration file:

```text
configs/fastspeech2.yaml
```

The configuration contains dataset paths, audio preprocessing parameters, model parameters, and training settings.

Before preprocessing or training, ensure that the configured paths point to the correct local dataset directories.

### FastSpeech2 Preprocessing

Prepare the dataset:

```bash
python -m preprocess.fastspeech2.Dataset.prepare_align
```

After the required MFA TextGrid files are available, run feature extraction:

```bash
python -m preprocess.fastspeech2.Dataset.preprocess
```

The preprocessing pipeline generates Mel, Pitch, Energy, and Duration features, together with dataset metadata and statistics.

### FastSpeech2 Training

Example training command:

```bash
python -m training.fastspeech2 \
    --steps 1000 \
    --batch-size 2 \
    --num-workers 0
```

For a minimal one-step execution test:

```bash
python -m training.fastspeech2 \
    --steps 1 \
    --batch-size 1 \
    --num-workers 0 \
    --save-interval 1
```

Training checkpoints are saved under the configured output directory.

Example:

```text
outputs/fastspeech2/checkpoints/step_1.pt
```

### FastSpeech2 Validation

Example validation command:

```bash
python -m validation.fastspeech2 \
    --checkpoint outputs/fastspeech2/checkpoints/step_1.pt \
    --batch-size 1 \
    --num-workers 0 \
    --max-batches 1
```

Validation loads the saved checkpoint and computes losses on the validation dataset.

### FastSpeech2 Smoke Test Results

The FastSpeech2 pipeline was tested using three LJSpeech samples:

- Training samples: 2
- Validation samples: 1
- GPU: NVIDIA GeForce RTX 5090
- Training steps: 1
- Checkpoint step: 1

Verified operations:

1. Dataset preparation
2. MFA alignment reuse
3. Acoustic feature extraction
4. Dataset loading and DataLoader construction
5. Model forward propagation
6. Loss computation
7. Backpropagation and optimizer update
8. Checkpoint saving
9. Checkpoint loading
10. Validation

Observed result:

```text
Training: PASS
Checkpoint Saving: PASS
Checkpoint Loading: PASS
Validation: PASS

Validation Loss: 18.174057
Checkpoint Step: 1
```

The complete minimal training and validation workflow executed successfully.

The reported loss is from a one-step smoke test and is not a converged model result.

---

## EfficientSpeech

EfficientSpeech is a lightweight non-autoregressive TTS architecture designed for efficient acoustic modeling.

The unified EfficientSpeech implementation includes:

- Dataset loading
- Text and phoneme processing
- Acoustic model
- Duration prediction
- Pitch prediction
- Energy prediction
- Mel-spectrogram prediction
- Loss computation
- Training
- Checkpoint generation
- Validation

### EfficientSpeech Configuration

Configuration file:

```text
configs/efficientspeech.yaml
```

The configuration defines preprocessing parameters, model architecture, and training settings.

The optimizer reads its parameters from the `train` configuration section:

```yaml
train:
  lr: 0.001
  weight_decay: 0.00001
  max_epochs: 1
```

The training script uses:

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=cfg["train"]["lr"],
    weight_decay=cfg["train"]["weight_decay"],
)
```

This ensures that optimizer settings are consistent with the YAML configuration.

### EfficientSpeech Preprocessing

Prepare the dataset:

```bash
python -m preprocess.efficientspeech.Dataset.prepare_align
```

After the required MFA TextGrid files are available, run feature extraction:

```bash
python -m preprocess.efficientspeech.Dataset.preprocess
```

The preprocessing pipeline generates:

- Mel-spectrogram
- Pitch
- Energy
- Duration
- Training and validation metadata
- Dataset statistics

The generated features are used by the unified EfficientSpeech dataset implementation.

### EfficientSpeech Training

Example training command:

```bash
python -m training.efficientspeech \
    --steps 1000 \
    --batch-size 2 \
    --num-workers 0
```

For a minimal one-step execution test:

```bash
python -m training.efficientspeech \
    --steps 1 \
    --batch-size 1 \
    --num-workers 0 \
    --save-interval 1
```

Example checkpoint path:

```text
outputs/efficientspeech/checkpoints/step_1.pt
```

### EfficientSpeech Validation

```bash
python -m validation.efficientspeech \
    --checkpoint outputs/efficientspeech/checkpoints/step_1.pt \
    --batch-size 1 \
    --num-workers 0 \
    --max-batches 1
```

### EfficientSpeech Smoke Test Results

The EfficientSpeech pipeline was tested using three LJSpeech samples:

- Training samples: 2
- Validation samples: 1
- GPU: NVIDIA GeForce RTX 5090
- Training steps: 1
- Batch size: 1
- Checkpoint step: 1

Preprocessing results:

```text
train.txt: PASS
val.txt: PASS
speakers.json: PASS
stats.json: PASS

Mel features: 3
Pitch features: 3
Energy features: 3
Duration features: 3

Preprocessing: PASS
```

Training results:

```text
Model: EfficientSpeech
Device: CUDA
GPU: NVIDIA GeForce RTX 5090

Training Samples: 2
Batch Size: 1
Training Steps: 1

Training-Reported Parameters: 266,355

Step 1:
    Total Loss: 298.1150
    Mel Loss: 23.6031
    Pitch Loss: 1.2362
    Energy Loss: 2.7527
    Duration Loss: 54.1060

Training: PASS
Checkpoint Saving: PASS
```

Validation results:

```text
Validation Samples: 1
Validation Batches: 1
Checkpoint Step: 1

Validation-Reported Parameters: 266,417

Total Loss: 335.695984
Mel Loss: 26.669079
Pitch Loss: 0.913599
Energy Loss: 1.370631
Duration Loss: 64.436714

Checkpoint Loading: PASS
Validation: PASS
```

The following workflow was successfully verified:

```text
LJSpeech
   |
   v
Preprocessing
   |
   v
Dataset / DataLoader
   |
   v
EfficientSpeech
   |
   v
Forward Propagation
   |
   v
Loss Computation
   |
   v
Backpropagation
   |
   v
Optimizer Update
   |
   v
Checkpoint Saving
   |
   v
Checkpoint Loading
   |
   v
Validation
```

The training and validation scripts reported different parameter counts:

```text
Training:   266,355
Validation: 266,417
Difference:      62
```

This discrepancy requires further investigation to determine whether it originates from parameter-count reporting or differences in model initialization.

The checkpoint was successfully loaded during the smoke test, but the parameter-count discrepancy has not yet been resolved.

The validation loss reflects execution after only one training step and should not be interpreted as final model performance.

---

## StyleTTS2

StyleTTS2 requires more components than FastSpeech2 and EfficientSpeech.

The required source modules have been migrated into:

```text
models/styletts2_modules/
```

The unified implementation contains the components required for the current reproduction workflow, including:

- Text encoder
- Style encoder
- Decoder
- Text aligner
- Pitch extractor
- PLBERT
- Diffusion modules
- Discriminators
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

## StyleTTS2 Reproduction Setup

This repository provides a minimal StyleTTS2 training and validation workflow.

### External Assets and Dataset

Pretrained weights and datasets are not included in this repository.

Set the following environment variables before running:

```bash
export PYTHONNOUSERSITE=1
export STYLETTS2_ASSETS=/path/to/tts_pretrained
export STYLETTS2_DATA=/path/to/styletts2_manifests
```

The pretrained assets directory should contain:

```text
ASR/config.yml
ASR/epoch_00080.pth
JDC/bst.t7
PLBERT/config.yml
PLBERT/step_1000000.t7
```

The manifest directory should contain:

```text
train_list.txt
val_list.txt
OOD_texts.txt
```

LJSpeech audio files should be placed at the path configured by `data_params.root_path` in `configs/styletts2.yaml`.

### Training

```bash
python -m training.styletts2 \
    --steps 10 \
    --batch-size 2 \
    --num-workers 0 \
    --log-interval 1 \
    --save-interval 10
```

### Validation

```bash
python -m validation.styletts2 \
    --checkpoint outputs/styletts2/checkpoints/step_10.pt \
    --batch-size 2 \
    --num-workers 0 \
    --max-batches 2
```

### Minimal Reproduction Result

- GPU: NVIDIA GeForce RTX 5090
- Training: 10 steps
- Validation: 2 batches
- STFT / Mel Loss: 0.951130

These results confirm execution of the minimal workflow, not full model convergence or final speech quality.

### Attribution

StyleTTS2 was developed by Yinghao Li and collaborators.

Original repository: https://github.com/yl4579/StyleTTS2

The original StyleTTS2 implementation is distributed under the MIT License.

Retain the original copyright and license notice when redistributing derived source code.

---

## Reproduction Status

The repository currently supports minimal training and validation workflows for all three TTS models.

| Component | FastSpeech2 | EfficientSpeech | StyleTTS2 |
|---|---|---|---|
| Dataset loading | PASS | PASS | PASS |
| Model construction | PASS | PASS | PASS |
| Unified training entry | PASS | PASS | PASS |
| Forward propagation | PASS | PASS | PASS |
| Loss computation | PASS | PASS | PASS |
| Backpropagation | PASS | PASS | PASS |
| Checkpoint saving | PASS | PASS | PASS |
| Checkpoint loading | PASS | PASS | PASS |
| Validation | PASS | PASS | PASS |
| Three-sample preprocessing test | PASS | PASS | Not tested in this integration |
| Full convergence | Not tested | Not tested | Not tested |

### Summary of Verified Experiments

| Metric | FastSpeech2 | EfficientSpeech | StyleTTS2 |
|---|---|---|---|
| GPU | RTX 5090 | RTX 5090 | RTX 5090 |
| Training steps | 1 | 1 | 10 |
| Validation batches | 1 | 1 | 2 |
| Checkpoint | step 1 | step 1 | step 10 |
| Validation loss | 18.174057 | 335.695984 | 0.951130 |
| Minimal workflow | PASS | PASS | PASS |

**Note:** The reported losses are model-specific and are not directly comparable. They use different architectures, loss definitions, and test settings.

The StyleTTS2 value in this table comes from the previously recorded reproduction setup.

The FastSpeech2 and EfficientSpeech results were obtained from three-sample LJSpeech smoke tests.

These experiments verify that the implementations can execute the required training and validation operations.

They do not establish convergence or reproduction of the original papers' reported performance.

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

Generated files such as the following should not be committed:

```text
*.pt
*.pth
*.t7
*.pyc
__pycache__/
*.bak
```

The exact exclusions are controlled by `.gitignore`.

This keeps the repository focused on the unified implementation, preprocessing scripts, experiment configurations, and documentation.

---

## Scope

A successful minimal reproduction in this repository means that a model can:

1. Load its dataset.
2. Execute the required preprocessing pipeline, where applicable.
3. Construct the model.
4. Execute forward propagation and loss computation.
5. Perform training updates.
6. Save a checkpoint.
7. Load the checkpoint.
8. Complete validation.

This does not imply:

- Full paper-level reproduction
- Full-dataset training convergence
- Reproduction of original MOS results
- Reproduction of all objective evaluation metrics
- Reproduction of all ablation experiments
- Production-ready speech synthesis quality

The reported smoke-test losses are primarily used to verify that the training and validation pipelines execute successfully.

---

## Current Status

The unified implementations of FastSpeech2, EfficientSpeech, and StyleTTS2 have completed their recorded minimal training and validation workflows.

FastSpeech2 and EfficientSpeech additionally passed three-sample LJSpeech preprocessing tests, including Mel, Pitch, Energy, and Duration feature generation.

The repository now provides a common foundation for:

- Full-dataset TTS training
- Model comparison
- Training pipeline development
- Acoustic feature analysis
- Checkpoint management
- Further model reproduction and optimization

The next stage is full-dataset training, convergence evaluation, and speech-quality assessment.