# StyleTTS2 Reproduction

This repository provides a reorganized and reproducible implementation of **StyleTTS2**, with the project separated into four main components:

- **Dataset**
- **Model**
- **Training**
- **Evaluate**

The purpose of this repository is to provide a clear training and validation pipeline that can be reproduced in a new environment.

---

## 1. Project Structure

```text
StyleTTS2/
├── Dataset/
│   ├── Data/
│   │   ├── train_list.txt
│   │   ├── val_list.txt
│   │   └── OOD_texts.txt
│   ├── meldataset.py
│   ├── __init__.py
│   └── README.md
│
├── Model/
│   ├── Modules/
│   ├── Utils/
│   │   ├── ASR/
│   │   ├── JDC/
│   │   └── PLBERT/
│   ├── losses.py
│   ├── models.py
│   ├── optimizers.py
│   ├── text_utils.py
│   ├── __init__.py
│   └── README.md
│
├── Training/
│   ├── Configs/
│   │   ├── config.yml
│   │   ├── config_test.yml
│   │   ├── config_ft.yml
│   │   └── config_libritts.yml
│   ├── train_first.py
│   ├── train_first_smoke.py
│   ├── train_second.py
│   ├── train_finetune.py
│   ├── train_finetune_accelerate.py
│   ├── __init__.py
│   └── README.md
│
├── Evaluate/
│   ├── validate.py
│   ├── Inference_LJSpeech.ipynb
│   ├── Inference_LibriTTS.ipynb
│   ├── __init__.py
│   └── README.md
│
├── Models/
├── requirements.txt
├── requirements_reproduce.txt
├── LICENSE
└── README.md

The four main directories correspond to the major stages of the reproduction pipeline:
Dataset
   ↓
Model
   ↓
Training
   ↓
Evaluate

2. Environment
The reproduction environment was tested with:
Python:       3.10
PyTorch:      2.7.0+cu128
Torchaudio:   2.7.0+cu128
CUDA:         12.8
GPU:          NVIDIA GeForce RTX 5090
librosa:      0.11.0
transformers: 5.17.0
accelerate:   1.15.0

Create a new environment:
conda create -n styletts2_test python=3.10 -y
conda activate styletts2_test

Install the required dependencies:
pip install -r requirements_reproduce.txt

The original dependency list is also retained in:
requirements.txt

3. Dataset
The reproduction uses the LJSpeech dataset.
The dataset directory should contain the original LJSpeech audio files:
LJSpeech-1.1/
├── metadata.csv
└── wavs/
    ├── LJ001-0001.wav
    ├── LJ001-0002.wav
    └── ...

The raw dataset is not included in this repository.
Dataset metadata used by StyleTTS2 is located in:
Dataset/Data/
├── train_list.txt
├── val_list.txt
└── OOD_texts.txt

The reproduction dataset contains:
Training samples:   12,500
Validation samples: 100

The dataset path is configured through the root_path field in the YAML configuration.
Example:
data_params:
  root_path: "./raw_data/LJSpeech-1.1/wavs"

Place the LJSpeech dataset under `raw_data/LJSpeech-1.1/`. The provided LJSpeech configurations use the relative path `./raw_data/LJSpeech-1.1/wavs`, so no machine-specific absolute path is required.
4. Dataset Verification
The StyleTTS2 dataset pipeline has been verified using the reorganized dataset module.
The DataLoader successfully:
- reads LJSpeech audio
- loads phoneme sequences
- generates mel-spectrograms
- constructs training batches
- constructs validation batches
A tested batch contained eight returned elements including text, lengths, mel features, and auxiliary training data.
The mel-spectrogram configuration uses:
Sampling rate: 24000 Hz
Mel channels:  80
FFT size:      2048
Window length: 1200
Hop length:    300

Note that the original LJSpeech WAV files are sampled at 22050 Hz. The StyleTTS2 data pipeline performs the required audio processing according to the configured training sampling rate.
5. Model
The StyleTTS2 model implementation is located in:
Model/

Important components include:
Model/models.py
Model/losses.py
Model/optimizers.py
text_utils.py
Model/Modules/
Model/Utils/

The model uses several auxiliary pretrained components during training:
ASR / Text Aligner
Model/Utils/ASR/

The tested checkpoint is:
Model/Utils/ASR/epoch_00080.pth

F0 Extractor
Model/Utils/JDC/

The tested checkpoint is:
Model/Utils/JDC/bst.t7

PLBERT
Model/Utils/PLBERT/

The tested checkpoint is:
Model/Utils/PLBERT/step_1000000.t7

6. Model Verification
The complete StyleTTS2 model has been successfully initialized in the reproduction environment.
The initialized components include:
bert
bert_encoder
predictor
decoder
text_encoder
predictor_encoder
style_encoder
diffusion
text_aligner
pitch_extractor
mpd
msd
wd

The complete initialized training model contains approximately:
187.05 M parameters

This number includes the StyleTTS2 network together with the auxiliary modules used during training.
7. Training Pipeline
Training scripts are located in:
Training/

The main scripts are:
Training/train_first.py
Training/train_second.py
Training/train_finetune.py
Training/train_finetune_accelerate.py

StyleTTS2 uses a multi-stage training strategy.
First Stage
The first-stage training script is:
python -m Training.train_first \
    -p Training/Configs/config.yml

This stage trains the first-stage acoustic model and alignment-related components.
Second Stage
The second-stage training script is:
python -m Training.train_second \
    -p Training/Configs/config.yml

The second stage introduces additional StyleTTS2 objectives including style diffusion and joint training components.
Fine-tuning
Fine-tuning is available through:
python -m Training.train_finetune \
    -p Training/Configs/config_ft.yml

An Accelerate-based fine-tuning implementation is also retained:
Training/train_finetune_accelerate.py

8. Smoke Training
A lightweight training configuration is provided to verify that the complete training pipeline works without running the full StyleTTS2 training schedule.
Configuration:
Training/Configs/config_test.yml

Smoke training script:
Training/train_first_smoke.py

Run:
python -m Training.train_first_smoke \
    -p Training/Configs/config_test.yml

The tested smoke configuration uses:
epochs_1st: 1
batch_size: 4
save_freq: 1
log_interval: 10

The smoke experiment successfully completed:
Model initialization
        ↓
Dataset loading
        ↓
Forward propagation
        ↓
Loss calculation
        ↓
Backward propagation
        ↓
Optimizer update
        ↓
Validation
        ↓
Checkpoint saving

The experiment completed 10 training iterations.
A validation loss was successfully calculated:
Validation loss: 0.923

This smoke experiment is intended to verify reproducibility of the training pipeline rather than reproduce the final speech quality reported by the original StyleTTS2 work.
9. Training Output
The smoke training experiment generates:
Models/LJSpeech_smoke/
├── config_test.yml
├── first_stage.pth
├── epoch_1st_00000.pth
├── train.log
└── tensorboard/

The generated checkpoint is approximately:
1.27 GB

Large checkpoints and training outputs should not be committed to Git.
10. Validation
The reproduction repository provides a lightweight checkpoint validation script:
Evaluate/validate.py

Run:
python -m Evaluate.validate \
    --checkpoint Models/LJSpeech_smoke/first_stage.pth

A successfully generated checkpoint produces output similar to:
=== StyleTTS2 Checkpoint Validation ===

Checkpoint size: 1269.43 MB

[OK] Checkpoint loaded successfully

Top-level checkpoint keys:
  - net
  - optimizer
  - iters
  - val_loss
  - epoch

Epoch: 0
Iterations: 10

=== VALIDATION OK ===

This verifies that the training pipeline generated a valid and reloadable StyleTTS2 checkpoint.
11. Inference
The original StyleTTS2 inference notebooks are retained under:
Evaluate/

including:
Evaluate/Inference_LJSpeech.ipynb
Evaluate/Inference_LibriTTS.ipynb

These notebooks provide the complete inference workflow for trained StyleTTS2 models.
Full inference requires the corresponding trained or pretrained StyleTTS2 checkpoint.
Some LibriTTS demonstrations also require reference audio files.
These external model and audio assets are not required for the lightweight training and validation reproduction described in this repository.
12. Checkpoint Compatibility
The reproduction environment uses a recent PyTorch version.
Starting from PyTorch 2.6, the default behavior of torch.load() changed to:
weights_only=True


Some pretrained StyleTTS2 auxiliary checkpoints contain objects that cannot be loaded with this default behavior.
For trusted StyleTTS2 checkpoints, the reproduction code explicitly uses:
torch.load(    path,    map_location="cpu",    weights_only=False,)


This change is required for compatibility with the tested PyTorch environment.
Only use weights_only=False with checkpoints from sources you trust.
13. Configuration
Training configurations are stored in:
Training/Configs/

Available configurations include:
config.yml
config_test.yml
config_ft.yml
config_libritts.yml

Important paths include:
F0_path: "Model/Utils/JDC/bst.t7"
ASR_config: "Model/Utils/ASR/config.yml"
ASR_path: "Model/Utils/ASR/epoch_00080.pth"
PLBERT_dir: "Model/Utils/PLBERT/"

Dataset metadata is configured as:
data_params:
  train_data: "Dataset/Data/train_list.txt"
  val_data: "Dataset/Data/val_list.txt"
  OOD_data: "Dataset/Data/OOD_texts.txt"

The root_path value must be changed to the location of the LJSpeech WAV directory on the target machine.
14. Reproduction Workflow
For a new environment, the recommended workflow is:
1. Clone repository
        ↓
2. Create Python environment
        ↓
3. Install dependencies
        ↓
4. Download LJSpeech
        ↓
5. Configure root_path
        ↓
6. Verify Dataset
        ↓
7. Initialize Model
        ↓
8. Run smoke training
        ↓
9. Generate checkpoint
        ↓
10. Validate checkpoint

The minimum reproduction goal is reached when the smoke training completes and the generated checkpoint passes the validation script.
15. Reproduction Status
The following components have been verified:
Component	Status
Python environment	Verified
CUDA / GPU	Verified
LJSpeech loading	Verified
DataLoader	Verified
ASR model loading	Verified
F0 model loading	Verified
PLBERT loading	Verified
StyleTTS2 initialization	Verified
Forward training pipeline	Verified
Backpropagation	Verified
Optimizer update	Verified
Validation	Verified
Checkpoint saving	Verified
Checkpoint reloading	Verified


The complete lightweight reproduction pipeline is therefore:
LJSpeech
   ↓
Dataset
   ↓
StyleTTS2 Model
   ↓
First-stage Smoke Training
   ↓
Validation
   ↓
Checkpoint
   ↓
Checkpoint Reload Verification

16. Notes
The smoke training configuration is designed for pipeline verification, not for reproducing the final perceptual quality reported by the original StyleTTS2 paper.
Full StyleTTS2 training requires substantially more training time and computational resources.
Raw datasets, generated checkpoints, experiment logs, and other large local artifacts should remain outside Git version control.
The reorganized directory structure is intended to make the codebase easier to understand, reproduce, and maintain.
17. Reference
This reproduction is based on the StyleTTS2 project:
StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial Training with Large Speech Language Models
The original implementation and paper should be cited when using StyleTTS2 for research purposes.

## Pretrained Auxiliary Weights

StyleTTS2 requires three auxiliary pretrained checkpoints during training:

```text
Model/Utils/ASR/epoch_00080.pth
Model/Utils/JDC/bst.t7
Model/Utils/PLBERT/step_1000000.t7

These large pretrained checkpoint files are not included in the reorganized repository. They should be obtained from the original StyleTTS2 resources and placed at the paths shown above before training.
The corresponding model definitions and configuration files are retained under Model/Utils/.