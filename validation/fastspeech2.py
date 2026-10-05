import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.fastspeech2 import (
    FastSpeech2,
    FastSpeech2Loss,
)
from datasets.fastspeech2 import (
    FastSpeech2Dataset,
    fastspeech2_collate_fn,
)
from common.utils import to_device


ROOT = Path(__file__).resolve().parents[1]


def load_config(model_name):
    config_path = ROOT / "configs" / f"{model_name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config not found: {config_path}"
        )

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_model(model_name, cfg):
    if model_name == "fastspeech2":
        model = FastSpeech2(
            cfg["preprocess"],
            cfg["model"],
        )

        criterion = FastSpeech2Loss(
            cfg["preprocess"],
            cfg["model"],
        )

        return model, criterion

    raise ValueError(
        f"Unsupported model: {model_name}"
    )


def build_dataset(model_name, cfg, split="val"):
    if model_name == "fastspeech2":
        dataset = FastSpeech2Dataset(
            f"{split}.txt",
            cfg["preprocess"],
        )

        return dataset, fastspeech2_collate_fn

    raise ValueError(
        f"Unsupported model: {model_name}"
    )


def main(args):
    print("=" * 60)
    print("TTS Validation")
    print("=" * 60)

    model_name = "fastspeech2"
    cfg = load_config(model_name)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Model      : {model_name}")
    print(f"Dataset    : {cfg['dataset']}")
    print(f"Device     : {device}")

    if torch.cuda.is_available():
        print(f"GPU        : {torch.cuda.get_device_name(0)}")

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset, collate_fn = build_dataset(
        model_name,
        cfg,
        split="val",
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Samples    : {len(dataset)}")
    print(f"Batch      : {args.batch_size}")

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model, criterion = build_model(
        model_name,
        cfg,
    )

    model = model.to(device)
    criterion = criterion.to(device)

    params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Parameters : {params:,}")

    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

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
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    checkpoint_step = checkpoint.get(
        "step",
        "unknown",
    )

    print(f"Checkpoint : {checkpoint_path}")
    print(f"Step       : {checkpoint_step}")

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    model.eval()

    loss_sums = [0.0] * 6
    num_batches = 0

    progress = tqdm(
        loader,
        desc="Validation",
    )

    with torch.no_grad():
        for batch in progress:
            batch = to_device(
                batch,
                device,
            )

            predictions = model(
                *(batch[2:])
            )

            losses = criterion(
                batch,
                predictions,
            )

            for i, loss in enumerate(losses):
                loss_sums[i] += loss.item()

            num_batches += 1

            progress.set_postfix(
                loss=f"{losses[0].item():.4f}"
            )

            if (
                args.max_batches is not None
                and num_batches >= args.max_batches
            ):
                break

    progress.close()

    if num_batches == 0:
        raise RuntimeError(
            "Validation produced zero batches."
        )

    avg_losses = [
        loss_sum / num_batches
        for loss_sum in loss_sums
    ]

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("Validation Results")
    print("=" * 60)

    print(f"Total       : {avg_losses[0]:.6f}")
    print(f"Mel         : {avg_losses[1]:.6f}")
    print(f"PostNet Mel : {avg_losses[2]:.6f}")
    print(f"Pitch       : {avg_losses[3]:.6f}")
    print(f"Energy      : {avg_losses[4]:.6f}")
    print(f"Duration    : {avg_losses[5]:.6f}")

    print("-" * 60)
    print(f"Batches     : {num_batches}")
    print(f"Checkpoint  : step {checkpoint_step}")
    print("=" * 60)
    print("[OK] Validation finished")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FastSpeech2 validation"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to FastSpeech2 checkpoint",
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
        help="Limit validation batches for smoke testing",
    )

    args = parser.parse_args()

    main(args)
