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

def build_loader(cfg, batch_size, num_workers):
    data_cfg = cfg.get("data_params", {})

    train_path = data_cfg.get(
        "train_data",
        str(ROOT / "datasets/styletts2_data/train_list.txt"),
    )

    root_path = data_cfg.get(
        "root_path",
        str(ROOT / "data/LJSpeech-1.1/wavs"),
    )

    ood_path = data_cfg.get(
        "OOD_data",
        str(ROOT / "datasets/styletts2_data/OOD_texts.txt"),
    )

    min_length = data_cfg.get("min_length", 50)

    train_path = resolve_style_path(train_path)
    ood_path = resolve_style_path(ood_path)

    if not Path(root_path).is_absolute():
        root_path = str(ROOT / root_path)

    with open(train_path, "r", encoding="utf-8") as f:
        train_list = f.readlines()

    loader = build_dataloader(
        train_list,
        root_path,
        validation=False,
        OOD_data=ood_path,
        min_length=min_length,
        batch_size=batch_size,
        num_workers=num_workers,
        device="cpu",
    )

    return loader, len(train_list)


def main(args):
    cfg = load_config()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("StyleTTS2 Minimal Training")
    print("=" * 60)
    print("Model   : styletts2")
    print("Dataset :", cfg["dataset"])
    print("Device  :", device)

    if torch.cuda.is_available():
        print("GPU     :", torch.cuda.get_device_name(0))

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("\nBuilding model...")

    model = build_styletts2(cfg)

    for key in model:
        model[key] = model[key].to(device)

    # Minimal Stage-1 training:
    # train acoustic reconstruction path only.
    train_keys = [
        "text_encoder",
        "style_encoder",
        "decoder",
    ]

    for key in model:
        model[key].eval()

    for key in train_keys:
        model[key].train()

    try:
        n_down = model.text_aligner.n_down
    except AttributeError:
        n_down = model["text_aligner"].n_down

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    params = []

    for key in train_keys:
        params.extend(
            p
            for p in model[key].parameters()
            if p.requires_grad
        )

    lr = float(
        cfg.get(
            "optimizer_params",
            {},
        ).get("lr", 1e-4)
    )

    optimizer = torch.optim.AdamW(
        params,
        lr=lr,
    )

    criterion = MultiResolutionSTFTLoss().to(device)

    print("[OK] Model ready")
    print("Train modules :", ", ".join(train_keys))
    print("Learning rate :", lr)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    loader, num_samples = build_loader(
        cfg,
        args.batch_size,
        args.num_workers,
    )

    print("Samples       :", num_samples)
    print("Batch size    :", args.batch_size)

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    checkpoint_dir = (
        ROOT
        / "outputs"
        / "styletts2"
        / "checkpoints"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    max_len = cfg.get(
        "max_len",
        cfg.get(
            "data_params",
            {},
        ).get("max_len", 400),
    )

    step = 0

    progress = tqdm(
        total=args.steps,
        desc="Training",
    )

    while step < args.steps:

        for batch in loader:

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

            # ------------------------------------------------
            # Alignment path
            # ------------------------------------------------

            with torch.no_grad():

                mask = length_to_mask(
                    mel_input_length
                    // (2 ** n_down)
                ).to(device)

                text_mask = length_to_mask(
                    input_lengths
                ).to(device)

                _, _, s2s_attn = model.text_aligner(
                    mels,
                    mask,
                    texts,
                )

                s2s_attn = s2s_attn.transpose(
                    -1,
                    -2,
                )

                s2s_attn = s2s_attn[..., 1:]

                s2s_attn = s2s_attn.transpose(
                    -1,
                    -2,
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
                    * (~text_mask)
                    .unsqueeze(-1)
                    .expand(
                        text_mask.shape[0],
                        text_mask.shape[1],
                        mask.shape[-1],
                    )
                    .float()
                )

                attn_mask = attn_mask < 1

                s2s_attn = s2s_attn.masked_fill(
                    attn_mask,
                    0.0,
                )

            # ------------------------------------------------
            # Text encoder
            # ------------------------------------------------

            t_en = model.text_encoder(
                texts,
                input_lengths,
                text_mask,
            )

            asr = t_en @ s2s_attn.detach()

            # ------------------------------------------------
            # Training clips
            # ------------------------------------------------

            mel_len = min(
                int(
                    mel_input_length.min().item()
                    / 2
                    - 1
                ),
                max_len // 2,
            )

            mel_len_st = int(
                mel_input_length.min().item()
                / 2
                - 1
            )

            if mel_len <= 0:
                continue

            en = []
            gt = []
            wav = []
            st = []

            valid_batch = True

            for bib in range(
                len(mel_input_length)
            ):

                current_mel_length = int(
                    mel_input_length[bib].item()
                    / 2
                )

                max_start = (
                    current_mel_length
                    - mel_len
                )

                if max_start <= 0:
                    valid_batch = False
                    break

                random_start = np.random.randint(
                    0,
                    max_start,
                )

                en.append(
                    asr[
                        bib,
                        :,
                        random_start:
                        random_start + mel_len
                    ]
                )

                gt.append(
                    mels[
                        bib,
                        :,
                        random_start * 2:
                        (random_start + mel_len) * 2
                    ]
                )

                y = waves[bib][
                    (random_start * 2) * 300:
                    ((random_start + mel_len) * 2)
                    * 300
                ]

                wav.append(
                    torch.from_numpy(y).to(device)
                )

                max_style_start = (
                    current_mel_length
                    - mel_len_st
                )

                if max_style_start <= 0:
                    style_start = 0
                else:
                    style_start = np.random.randint(
                        0,
                        max_style_start,
                    )

                st.append(
                    mels[
                        bib,
                        :,
                        style_start * 2:
                        (style_start + mel_len_st)
                        * 2
                    ]
                )

            if not valid_batch:
                continue

            en = torch.stack(en)
            gt = torch.stack(gt).detach()
            st = torch.stack(st).detach()

            try:
                wav = torch.stack(
                    wav
                ).float().detach()
            except RuntimeError:
                continue

            if gt.shape[-1] < 80:
                continue

            # ------------------------------------------------
            # Pitch + norm
            # ------------------------------------------------

            with torch.no_grad():

                real_norm = log_norm(
                    gt.unsqueeze(1)
                ).squeeze(1).detach()

                F0_real, _, _ = (
                    model.pitch_extractor(
                        gt.unsqueeze(1)
                    )
                )

            # ------------------------------------------------
            # Style
            # ------------------------------------------------

            multispeaker = cfg.get(
                "model_params",
                {},
            ).get(
                "multispeaker",
                False,
            )

            style_input = (
                st.unsqueeze(1)
                if multispeaker
                else gt.unsqueeze(1)
            )

            s = model.style_encoder(
                style_input
            )

            # ------------------------------------------------
            # Decoder
            # ------------------------------------------------

            y_rec = model.decoder(
                en,
                F0_real,
                real_norm,
                s,
            )

            # ------------------------------------------------
            # Loss + backward
            # ------------------------------------------------

            loss = criterion(
                y_rec.squeeze(1),
                wav,
            )

            if not torch.isfinite(loss):
                print(
                    "\n[WARN] Non-finite loss, "
                    "skipping batch"
                )
                continue

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                params,
                10.0,
            )

            optimizer.step()

            step += 1

            progress.update(1)

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

            if (
                step == 1
                or step % args.log_interval == 0
            ):
                print(
                    f"\nStep {step:6d} | "
                    f"STFT Loss "
                    f"{loss.item():.6f}"
                )

            # ------------------------------------------------
            # Checkpoint
            # ------------------------------------------------

            if (
                step % args.save_interval == 0
                or step == args.steps
            ):

                checkpoint_path = (
                    checkpoint_dir
                    / f"step_{step}.pt"
                )

                state = {
                    key: model[key].state_dict()
                    for key in model
                }

                torch.save(
                    {
                        "model": state,
                        "optimizer":
                            optimizer.state_dict(),
                        "step": step,
                        "config": cfg,
                    },
                    checkpoint_path,
                )

                print(
                    "\n[Checkpoint]",
                    checkpoint_path,
                )

            if step >= args.steps:
                break

    progress.close()

    print()
    print("=" * 60)
    print(
        "[OK] StyleTTS2 training finished"
    )
    print("Final step :", step)
    print("Checkpoint :", checkpoint_dir)
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

    args = parser.parse_args()

    main(args)
