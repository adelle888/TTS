import argparse
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.fastspeech2 import (
    FastSpeech2,
    FastSpeech2Loss,
    ScheduledOptim,
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


def build_dataset(model_name, cfg, split="train"):
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
    print("TTS Unified Training")
    print("=" * 60)

    model_name = "fastspeech2"
    cfg = load_config(model_name)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Model   : {model_name}")
    print(f"Dataset : {cfg['dataset']}")
    print(f"Device  : {device}")

    if torch.cuda.is_available():
        print(f"GPU     : {torch.cuda.get_device_name(0)}")

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset, collate_fn = build_dataset(
        model_name,
        cfg,
        split="train",
    )

    batch_size = args.batch_size

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Samples : {len(dataset)}")
    print(f"Batch   : {batch_size}")

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

    print(f"Parameters: {params:,}")

    optimizer = ScheduledOptim(
        model,
        cfg["train"],
        cfg["model"],
        current_step=0,
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output_dir = (
        ROOT
        / "outputs"
        / model_name
    )

    checkpoint_dir = (
        output_dir
        / "checkpoints"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()

    step = 0

    progress = tqdm(
        total=args.steps,
        desc="Training",
    )

    while step < args.steps:
        for batch in loader:
            batch = to_device(
                batch,
                device,
            )

            optimizer.zero_grad()

            predictions = model(
                *(batch[2:])
            )

            losses = criterion(
                batch,
                predictions,
            )

            total_loss = losses[0]

            total_loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                cfg["train"]["optimizer"]["grad_clip_thresh"],
            )

            optimizer.step_and_update_lr()

            step += 1

            progress.update(1)

            if (
                step == 1
                or step % args.log_interval == 0
            ):
                lr = optimizer._optimizer.param_groups[0]["lr"]

                print(
                    f"\nStep {step:6d} | "
                    f"Loss {losses[0].item():.4f} | "
                    f"Mel {losses[1].item():.4f} | "
                    f"PostNet {losses[2].item():.4f} | "
                    f"Pitch {losses[3].item():.4f} | "
                    f"Energy {losses[4].item():.4f} | "
                    f"Duration {losses[5].item():.4f} | "
                    f"LR {lr:.8f}"
                )

            if (
                step % args.save_interval == 0
                or step == args.steps
            ):
                checkpoint_path = (
                    checkpoint_dir
                    / f"step_{step}.pt"
                )

                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer":
                            optimizer._optimizer.state_dict(),
                        "step": step,
                        "config": cfg,
                    },
                    checkpoint_path,
                )

                print(
                    f"\n[Checkpoint] {checkpoint_path}"
                )

            if step >= args.steps:
                break

    progress.close()

    print()
    print("=" * 60)
    print("[OK] Training finished")
    print(f"Final step: {step}")
    print(f"Checkpoints: {checkpoint_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified TTS training entry"
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=10,
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
        "--log-interval",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--save-interval",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    main(args)
