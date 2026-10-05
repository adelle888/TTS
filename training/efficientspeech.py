import argparse
from pathlib import Path

import torch
import torch.nn as nn
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
    print("EfficientSpeech Training")
    print("=" * 60)

    with open(ROOT / "configs/efficientspeech.yaml") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Model   : efficientspeech")
    print("Dataset :", cfg["dataset"])
    print("Device  :", device)

    if torch.cuda.is_available():
        print("GPU     :", torch.cuda.get_device_name(0))

    dataset = EfficientSpeechDataset(
        "train.txt",
        cfg["preprocess"],
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=efficientspeech_collate_fn,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print("Samples :", len(dataset))
    print("Batch   :", args.batch_size)

    model = EfficientSpeech(
        preprocess_config=cfg["preprocess"],
        **cfg["model"],
    ).to(device)

    criterion = EfficientSpeechLoss()

    params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("Parameters:", f"{params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.get("training", {}).get("lr", 1e-3),
        weight_decay=cfg.get(
            "training", {}
        ).get("weight_decay", 1e-5),
    )

    output_dir = (
        ROOT
        / "outputs"
        / "efficientspeech"
        / "checkpoints"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.train()

    step = 0

    progress = tqdm(
        total=args.steps,
        desc="Training",
    )

    while step < args.steps:

        for x, y in loader:

            x = move_to_device(x, device)
            y = move_to_device(y, device)

            optimizer.zero_grad()

            pred = model(
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
                pred,
                targets,
            )

            loss = losses["total"]

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            step += 1
            progress.update(1)

            if (
                step == 1
                or step % args.log_interval == 0
            ):
                print(
                    f"\nStep {step:6d} | "
                    f"Loss {losses['total'].item():.4f} | "
                    f"Mel {losses['mel'].item():.4f} | "
                    f"Pitch {losses['pitch'].item():.4f} | "
                    f"Energy {losses['energy'].item():.4f} | "
                    f"Duration {losses['duration'].item():.4f}"
                )

            if (
                step % args.save_interval == 0
                or step == args.steps
            ):

                path = (
                    output_dir
                    / f"step_{step}.pt"
                )

                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": step,
                        "config": cfg,
                    },
                    path,
                )

                print(
                    f"\n[Checkpoint] {path}"
                )

            if step >= args.steps:
                break

    progress.close()

    print()
    print("=" * 60)
    print("[OK] EfficientSpeech training finished")
    print("Final step :", step)
    print("Checkpoint :", output_dir)
    print("=" * 60)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

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

    main(parser.parse_args())
