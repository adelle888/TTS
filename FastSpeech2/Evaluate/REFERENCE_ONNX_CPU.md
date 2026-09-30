# Reference ONNX CPU run

This script times acoustic FastSpeech2 inference and, if supplied, a PWGAN vocoder. It saves model hashes and per-call times. It does not time the text front end or model initialization.

```bash
python -m pip install numpy onnxruntime
python FastSpeech2/Evaluate/reference_onnx_cpu.py \
  --acoustic /external/path/fastspeech2_csmsc.onnx \
  --phone-map /external/path/phone_id_map.txt \
  --vocoder /external/path/pwgan_csmsc.onnx \
  --out /external/path/tts_cpu_run
```

The Windows reference run used **PaddleSpeech CSMSC ONNX assets**, while this repository's FastSpeech2 source documents an **LJSpeech reproduction** and does not include its trained checkpoint. This script does not convert the repository model to ONNX or Vela, and the output is not an M33 or power baseline. Leave `--vocoder` out to measure acoustic text-to-mel only.
