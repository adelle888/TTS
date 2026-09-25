from pathlib import Path

REQUIRED_FILES = {
    "JDC F0 extractor": Path("Model/Utils/JDC/bst.t7"),
    "ASR text aligner": Path("Model/Utils/ASR/epoch_00080.pth"),
    "PLBERT": Path("Model/Utils/PLBERT/step_1000000.t7"),
}

print("=== StyleTTS2 Pretrained Model Check ===")

missing = []

for name, path in REQUIRED_FILES.items():
    if path.exists():
        size_mb = path.stat().st_size / (1024 ** 2)
        print(f"[OK]      {name:<20} {size_mb:8.2f} MB  {path}")
    else:
        print(f"[MISSING] {name:<20} {'':>8}     {path}")
        missing.append(path)

if missing:
    print("\nMissing pretrained files:")
    for path in missing:
        print(f"  - {path}")

    print(
        "\nPlease obtain the official StyleTTS2 pretrained auxiliary "
        "weights and place them at the paths shown above."
    )
    raise SystemExit(1)

print("\n=== ALL PRETRAINED MODELS OK ===")
