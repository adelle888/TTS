Model
This directory contains the StyleTTS2 model architecture and the pretrained auxiliary components required by the training pipeline.
Main Components
- models.py: StyleTTS2 model construction.
- losses.py: training loss functions.
- optimizers.py: optimizer and scheduler construction.
- text_utils.py: text processing utilities.
- Modules/: StyleTTS2 neural network modules.
- Utils/ASR/: pretrained text aligner.
- Utils/JDC/: pretrained F0 extractor.
- Utils/PLBERT/: pretrained PLBERT model.
Model Verification
The complete StyleTTS2 model has been successfully initialized using the reproduction environment.
The initialized model contains approximately:
187.05 M parameters

This includes the acoustic model and auxiliary training components such as the text aligner, pitch extractor, discriminators, and style diffusion modules.
eof
