Evaluate

This directory contains the FastSpeech2 validation and synthesis entry points.

Directory Structure
Evaluate/
├── evaluate.py
├── synthesize.py
└── README.md
1. Independent Evaluation

The main evaluation entry point is:

Evaluate/evaluate.py

It loads a saved FastSpeech2 checkpoint and evaluates the model using the validation dataset.

2. Run Evaluation

After completing the 100-step smoke test, evaluate the saved Step-100 checkpoint with:

python Evaluate/evaluate.py \
  --restore_step 100 \
  -p Training/config/LJSpeech/preprocess.yaml \
  -m Training/config/LJSpeech/model.yaml \
  -t Training/config/LJSpeech/train_test.yaml

The corresponding checkpoint is loaded from:

output/ckpt/LJSpeech_test/100.pth.tar
3. Final Verified Result

The independent evaluation was executed after the final directory reorganization.

The verified output was:

Validation Step 100

Total Loss:        8.8164
Mel Loss:          3.6938
Mel PostNet Loss:  2.3898
Pitch Loss:        1.2753
Energy Loss:       1.0990
Duration Loss:     0.3585

This result is identical to the Step-100 validation result produced during the corresponding training run.

This confirms that the saved checkpoint can be loaded and evaluated independently after the project reorganization.

4. Evaluation Pipeline
output/ckpt/LJSpeech_test/100.pth.tar
                 ↓
        Load FastSpeech2
                 ↓
       Validation Dataset
                 ↓
      Forward Propagation
                 ↓
       Loss Calculation
                 ↓
┌───────────────────────────┐
│ Mel Loss                  │
│ Mel PostNet Loss          │
│ Pitch Loss                │
│ Energy Loss               │
│ Duration Loss             │
└───────────────────────────┘
                 ↓
        Validation Result
5. Synthesis

Speech synthesis is implemented in:

Evaluate/synthesize.py

The synthesis pipeline uses the FastSpeech2 acoustic model to predict mel spectrograms and a vocoder to convert the mel spectrograms into waveforms.

6. Scope

The purpose of this evaluation is to verify that:

A saved checkpoint can be loaded.
The validation dataset can be processed.
Model inference executes correctly.
All FastSpeech2 validation losses can be calculated.
The reorganized evaluation pipeline works independently.

The 100-step smoke-test result should not be interpreted as the final performance of the FastSpeech2 model.

Verified Status
Checkpoint loading       ✓
Validation data loading  ✓
Model inference          ✓
Loss calculation         ✓
Independent evaluation  ✓