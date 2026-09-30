"""Host CPU baseline for explicitly supplied FastSpeech2 and optional vocoder ONNX files.

The reference ONNX files used in this experiment come from PaddleSpeech,
not from a checkpoint trained by this repository.
"""

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acoustic", required=True, type=Path)
    parser.add_argument("--phone-map", required=True, type=Path)
    parser.add_argument("--vocoder", type=Path, help="omit to time text-to-mel only")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    acoustic = ort.InferenceSession(str(args.acoustic), options, providers=["CPUExecutionProvider"])
    vocoder = (ort.InferenceSession(str(args.vocoder), options, providers=["CPUExecutionProvider"])
               if args.vocoder else None)
    phone_ids = {}
    for line in args.phone_map.read_text(encoding="utf-8").splitlines():
        token, number = line.split()
        phone_ids[token] = int(number)
    # Fixed sample used by the local PC workload; the text front end is not timed.
    sample = np.asarray([phone_ids[p] for p in ("n", "i3", "h", "ao3")], dtype=np.int64)
    acoustic_input = acoustic.get_inputs()[0].name
    vocoder_input = vocoder.get_inputs()[0].name if vocoder else None
    rows = []
    for index in range(args.count):
        start = time.perf_counter_ns()
        mel = acoustic.run(None, {acoustic_input: sample})[0]
        acoustic_ms = (time.perf_counter_ns() - start) / 1e6
        if mel.ndim != 2 or mel.shape[1] != 80 or not np.isfinite(mel).all():
            raise RuntimeError(f"invalid mel output: {mel.shape}")
        vocoder_ms, samples = "", ""
        if vocoder:
            start = time.perf_counter_ns()
            audio = vocoder.run(None, {vocoder_input: mel})[0]
            vocoder_ms = round((time.perf_counter_ns() - start) / 1e6, 4)
            if not np.isfinite(audio).all():
                raise RuntimeError("nonfinite audio output")
            samples = int(audio.size)
        rows.append({"iteration": index, "acoustic_ms": round(acoustic_ms, 4),
                     "vocoder_ms": vocoder_ms, "mel_frames": int(mel.shape[0]),
                     "audio_samples": samples})
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "calls.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {"execution": "host CPU ONNX Runtime", "acoustic_sha256": digest(args.acoustic),
                "phone_map_sha256": digest(args.phone_map),
                "vocoder_sha256": digest(args.vocoder) if args.vocoder else None,
                "input": "fixed phoneme IDs for ni3 hao3; no text-front-end timing",
                "scope": "PaddleSpeech reference assets; not this repository's checkpoint or an M33 run"}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"saved {len(rows)} CPU calls to {args.out}")


if __name__ == "__main__":
    main()
