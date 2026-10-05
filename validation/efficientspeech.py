import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.efficientspeech import (
    EfficientSpeech,
    EfficientSpeechLoss,
)
from datasets.efficientspeech import (
    EfficientSpeechDataset,
    efficientspeech_collate_fn,
)


ROOT = Path(__file__).resolve().parents[1]


def move_to_device(data, device):
    for key, value in data.items():
        if torch.is_tensor(value):
            data[key] = value.to(device)
    return data


def main(args):

    print("=" * 60)
    print("EfficientSpeech Validation")
    print("=" * 60)

    # --------------------------------------------------
    # Config
    # --------------------------------------------------

    with open(
        ROOT / "configs" / "efficientspeech.yaml",
        "r",
    ) as f:
        cfg = yaml.safe_load(f)

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Model      : efficientspeech")
    print("Dataset    :", cfg["dataset"])
    print("Device     :", device)

    if torch.cuda.is_available():
        print(
            "GPU        :",
            torch.cuda.get_device_name(0),
        )

    # --------------------------------------------------
    # Validation dataset
    # --------------------------------------------------

    dataset = EfficientSpeechDataset(
        "val.txt",
        cfg["preprocess"],
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=efficientspeech_collate_fn,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print("Samples    :", len(dataset))
    print("Batch      :", args.batch_size)

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = EfficientSpeech(
        preprocess_config=cfg["preprocess"],
        **cfg["model"],
    ).to(device)

    criterion = EfficientSpeechLoss()

    params = sum(
        p.numel()
        for p in model.parameters()
    )

    print("Parameters :", f"{params:,}")

    # --------------------------------------------------
    # Checkpoint
    # --------------------------------------------------

    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    checkpoint_step = checkpoint.get(
        "step",
        "unknown",
    )

    print("Checkpoint :", checkpoint_path)
    print("Step       :", checkpoint_step)

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    model.eval()

    totals = {
        "total": 0.0,
        "mel": 0.0,
        "pitch": 0.0,
        "energy": 0.0,
        "duration": 0.0,
    }

    num_batches = 0

    max_batches = args.max_batches

    if max_batches is None:
        progress_total = len(loader)
    else:
        progress_total = min(
            len(loader),
            max_batches,
        )

    progress = tqdm(
        total=progress_total,
        desc="Validation",
    )

    with torch.no_grad():

        for x, y in loader:

            x = move_to_device(
                x,
                device,
            )

            y = move_to_device(
                y,
                device,
            )

            predictions = model(
                x,
                train=True,
            )

            targets = {
                "pitch": x["pitch"],
                "energy": x["energy"],
                "duration": x["duration"],
                "mel": y["mel"],
            }

            losses = criterion(
                predictions,
                targets,
            )

            for name in totals:
                totals[name] += (
                    losses[name].item()
                )

            num_batches += 1

            progress.update(1)

            progress.set_postfix(
                loss=f"{losses['total'].item():.4f}"
            )

            if (
                max_batches is not None
                and num_batches >= max_batches
            ):
                break

    progress.close()

    # --------------------------------------------------
    # Average
    # --------------------------------------------------

    if num_batches == 0:
        raise RuntimeError(
            "No validation batches were processed."
        )

    averages = {
        name: value / num_batches
        for name, value in totals.items()
    }

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("Validation Results")
    print("=" * 60)

    print(
        f"Total       : {averages['total']:.6f}"
    )

    print(
        f"Mel         : {averages['mel']:.6f}"
    )

    print(
        f"Pitch       : {averages['pitch']:.6f}"
    )

    print(
        f"Energy      : {averages['energy']:.6f}"
    )

    print(
        f"Duration    : {averages['duration']:.6f}"
    )

    print("-" * 60)

    print(
        f"Batches     : {num_batches}"
    )

    print(
        f"Checkpoint  : step {checkpoint_step}"
    )

    print("=" * 60)
    print("[OK] EfficientSpeech validation finished")
    print("=" * 60)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="EfficientSpeech validation"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    main(args)
