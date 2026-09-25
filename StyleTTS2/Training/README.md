Training
This directory contains the StyleTTS2 training scripts and configuration files.
Training Scripts
- train_first.py: first-stage StyleTTS2 training.
- train_second.py: second-stage joint training.
- train_finetune.py: model fine-tuning.
- train_finetune_accelerate.py: Accelerate-based fine-tuning.
- train_first_smoke.py: lightweight first-stage reproduction test.
Configuration
Training configurations are stored in:
Training/Configs/

The main LJSpeech configuration is:
Training/Configs/config.yml

The lightweight reproduction test uses:
Training/Configs/config_test.yml

Smoke Training
A first-stage smoke training experiment was successfully completed using:
python -m Training.train_first_smoke \
    -p Training/Configs/config_test.yml

The smoke test performs:
- StyleTTS2 model initialization
- LJSpeech DataLoader construction
- forward propagation
- loss calculation
- backward propagation
- optimizer update
- validation
- checkpoint saving
The reproduction test completed 10 training iterations and successfully produced a first-stage checkpoint.
Output
Training outputs are written to the directory specified by log_dir.
For the smoke test:
Models/LJSpeech_smoke/

Large checkpoint files are local experiment artifacts and should not be committed to Git.
