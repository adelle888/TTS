import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))

from models.styletts2 import build_styletts2
from datasets.styletts2 import build_dataloader
from common.utils import length_to_mask, log_norm
from models.styletts2_modules.losses import MultiResolutionSTFTLoss


def load_config():
    with open(ROOT / "configs/styletts2.yaml", "r") as f:
        return yaml.safe_load(f)


def resolve_style_path(path):
    p = Path(path)

    if p.is_absolute():
        return str(p)

    return str(ROOT / p)

def build_val_loader(cfg, batch_size, num_workers):
    data_cfg = cfg.get("data_params", {})

    val_path = data_cfg.get(
        "val_data",
        str(ROOT / "datasets/styletts2_data/val_list.txt"),
    )

    root_path = data_cfg.get(
        "root_path",
        str(ROOT / "data/LJSpeech-1.1/wavs"),
    )

    ood_path = data_cfg.get(
        "OOD_data",
        str(ROOT / "datasets/styletts2_data/OOD_texts.txt"),
    )

    min_length = data_cfg.get(
        "min_length",
        50,
    )

    val_path = resolve_style_path(val_path)
    ood_path = resolve_style_path(ood_path)

    if not Path(root_path).is_absolute():
        root_path = str(ROOT / root_path)

    with open(
        val_path,
        "r",
        encoding="utf-8",
    ) as f:
        val_list = f.readlines()

    loader = build_dataloader(
        val_list,
        root_path,
        validation=True,
        OOD_data=ood_path,
        min_length=min_length,
        batch_size=batch_size,
        num_workers=num_workers,
        device="cpu",
    )

    return loader, len(val_list)


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    state = checkpoint["model"]

    loaded = []

    for key in model:
        if key in state:
            model[key].load_state_dict(
                state[key],
                strict=True,
            )
            loaded.append(key)

    return checkpoint, loaded


def main(args):
    cfg = load_config()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("StyleTTS2 Validation")
    print("=" * 60)

    print("Model      : styletts2")
    print("Dataset    :", cfg["dataset"])
    print("Device     :", device)

    if torch.cuda.is_available():
        print(
            "GPU        :",
            torch.cuda.get_device_name(0),
        )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("\nBuilding model...")

    model = build_styletts2(cfg)

    for key in model:
        model[key] = model[key].to(device)
        model[key].eval()

    try:
        n_down = model.text_aligner.n_down
    except AttributeError:
        n_down = model["text_aligner"].n_down

    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.is_absolute():
        checkpoint_path = (
            ROOT / checkpoint_path
        )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{checkpoint_path}"
        )

    checkpoint, loaded = load_checkpoint(
        model,
        checkpoint_path,
        device,
    )

    print("[OK] Model built")
    print(
        "Checkpoint :",
        checkpoint_path,
    )
    print(
        "Step       :",
        checkpoint.get("step", "unknown"),
    )
    print(
        "Modules    :",
        len(loaded),
        "loaded",
    )

    # --------------------------------------------------------
    # Validation dataset
    # --------------------------------------------------------

    loader, num_samples = build_val_loader(
        cfg,
        args.batch_size,
        args.num_workers,
    )

    print(
        "Samples    :",
        num_samples,
    )
    print(
        "Batch      :",
        args.batch_size,
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = (
        MultiResolutionSTFTLoss()
        .to(device)
    )

    max_len = cfg.get(
        "max_len",
        cfg.get(
            "data_params",
            {},
        ).get(
            "max_len",
            400,
        ),
    )

    total_loss = 0.0
    valid_batches = 0

    progress = tqdm(
        loader,
        desc="Validation",
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    with torch.no_grad():

        for batch_idx, batch in enumerate(
            progress
        ):

            if (
                args.max_batches is not None
                and
                valid_batches
                >= args.max_batches
            ):
                break

            waves = batch[0]

            batch_gpu = [
                b.to(device)
                for b in batch[1:]
            ]

            (
                texts,
                input_lengths,
                _,
                _,
                mels,
                mel_input_length,
                _,
            ) = batch_gpu

            # --------------------------------------------
            # Masks + alignment
            # --------------------------------------------

            mask = length_to_mask(
                mel_input_length
                // (2 ** n_down)
            ).to(device)

            text_mask = length_to_mask(
                input_lengths
            ).to(device)

            _, _, s2s_attn = (
                model.text_aligner(
                    mels,
                    mask,
                    texts,
                )
            )

            s2s_attn = (
                s2s_attn
                .transpose(-1, -2)
            )

            s2s_attn = s2s_attn[..., 1:]

            s2s_attn = (
                s2s_attn
                .transpose(-1, -2)
            )

            attn_mask = (
                (~mask)
                .unsqueeze(-1)
                .expand(
                    mask.shape[0],
                    mask.shape[1],
                    text_mask.shape[-1],
                )
                .float()
                .transpose(-1, -2)
            )

            attn_mask = (
                attn_mask
                *
                (~text_mask)
                .unsqueeze(-1)
                .expand(
                    text_mask.shape[0],
                    text_mask.shape[1],
                    mask.shape[-1],
                )
                .float()
            )

            attn_mask = attn_mask < 1

            s2s_attn = (
                s2s_attn
                .masked_fill(
                    attn_mask,
                    0.0,
                )
            )

            # --------------------------------------------
            # Text encoding
            # --------------------------------------------

            t_en = model.text_encoder(
                texts,
                input_lengths,
                text_mask,
            )

            asr = t_en @ s2s_attn

            # --------------------------------------------
            # Fixed validation clips
            #
            # Unlike training, use start=0 so validation
            # is deterministic.
            # --------------------------------------------

            mel_len = min(
                int(
                    mel_input_length
                    .min()
                    .item()
                    / 2
                    - 1
                ),
                max_len // 2,
            )

            if mel_len <= 0:
                continue

            en = []
            gt = []
            wav = []

            valid_batch = True

            for bib in range(
                len(mel_input_length)
            ):

                current_mel_length = int(
                    mel_input_length[
                        bib
                    ].item()
                    / 2
                )

                if (
                    current_mel_length
                    <= mel_len
                ):
                    valid_batch = False
                    break

                start = 0

                en.append(
                    asr[
                        bib,
                        :,
                        start:
                        start + mel_len
                    ]
                )

                gt.append(
                    mels[
                        bib,
                        :,
                        start * 2:
                        (start + mel_len)
                        * 2
                    ]
                )

                y = waves[bib][
                    (start * 2) * 300:
                    ((start + mel_len) * 2)
                    * 300
                ]

                wav.append(
                    torch.from_numpy(y)
                    .to(device)
                )

            if not valid_batch:
                continue

            en = torch.stack(en)
            gt = torch.stack(gt).detach()

            try:
                wav = (
                    torch.stack(wav)
                    .float()
                    .detach()
                )
            except RuntimeError:
                continue

            if gt.shape[-1] < 80:
                continue

            # --------------------------------------------
            # Pitch + norm
            # --------------------------------------------

            real_norm = log_norm(
                gt.unsqueeze(1)
            ).squeeze(1)

            F0_real, _, _ = (
                model.pitch_extractor(
                    gt.unsqueeze(1)
                )
            )

            # --------------------------------------------
            # Style
            # --------------------------------------------

            multispeaker = cfg.get(
                "model_params",
                {},
            ).get(
                "multispeaker",
                False,
            )

            # LJSpeech is single-speaker.
            # Same segment is sufficient for
            # reconstruction validation.
            style_input = gt.unsqueeze(1)

            if multispeaker:
                style_input = gt.unsqueeze(1)

            s = model.style_encoder(
                style_input
            )

            # --------------------------------------------
            # Decoder
            # --------------------------------------------

            y_rec = model.decoder(
                en,
                F0_real,
                real_norm,
                s,
            )

            # --------------------------------------------
            # Loss
            # --------------------------------------------

            loss = criterion(
                y_rec.squeeze(1),
                wav,
            )

            if not torch.isfinite(loss):
                print(
                    "\n[WARN] "
                    "Non-finite validation loss; "
                    "skipping batch"
                )
                continue

            total_loss += loss.item()
            valid_batches += 1

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    if valid_batches == 0:
        raise RuntimeError(
            "No valid validation batches."
        )

    average_loss = (
        total_loss
        / valid_batches
    )

    print()
    print("=" * 60)
    print("Validation Results")
    print("=" * 60)

    print(
        "STFT / Mel Loss :",
        f"{average_loss:.6f}",
    )

    print("-" * 60)

    print(
        "Batches         :",
        valid_batches,
    )

    print(
        "Checkpoint      : step",
        checkpoint.get(
            "step",
            "unknown",
        ),
    )

    print("=" * 60)
    print(
        "[OK] StyleTTS2 validation finished"
    )
    print("=" * 60)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "StyleTTS2 validation"
        )
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
