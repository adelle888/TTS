import torch
import torch.nn.functional as F


def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]

    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(
        0,
        max_len,
        device=lengths.device
    ).unsqueeze(0).expand(batch_size, -1)

    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max(x.size(0) for x in input_ele)

    out_list = []

    for batch in input_ele:
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch,
                (0, max_len - batch.size(0)),
                "constant",
                0.0,
            )

        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch,
                (0, 0, 0, max_len - batch.size(0)),
                "constant",
                0.0,
            )

        else:
            raise ValueError(
                f"Unsupported tensor shape: {batch.shape}"
            )

        out_list.append(one_batch_padded)

    return torch.stack(out_list)


def pad_1D(inputs, PAD=0):
    import numpy as np

    max_len = max(len(x) for x in inputs)

    return np.stack([
        np.pad(
            x,
            (0, max_len - x.shape[0]),
            mode="constant",
            constant_values=PAD,
        )
        for x in inputs
    ])


def pad_2D(inputs, maxlen=None):
    import numpy as np

    max_len = (
        maxlen
        if maxlen is not None
        else max(x.shape[0] for x in inputs)
    )

    outputs = []

    for x in inputs:
        if x.shape[0] > max_len:
            raise ValueError(
                f"Input length {x.shape[0]} > max_len {max_len}"
            )

        outputs.append(
            np.pad(
                x,
                ((0, max_len - x.shape[0]), (0, 0)),
                mode="constant",
                constant_values=0,
            )
        )

    return np.stack(outputs)


def to_device(data, device):
    (
        ids,
        raw_texts,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels,
        mel_lens,
        max_mel_len,
        pitches,
        energies,
        durations,
    ) = data

    speakers = torch.from_numpy(speakers).long().to(device)
    texts = torch.from_numpy(texts).long().to(device)
    src_lens = torch.from_numpy(src_lens).long().to(device)

    mels = torch.from_numpy(mels).float().to(device)
    mel_lens = torch.from_numpy(mel_lens).long().to(device)

    pitches = torch.from_numpy(pitches).float().to(device)
    energies = torch.from_numpy(energies).float().to(device)
    durations = torch.from_numpy(durations).long().to(device)

    return (
        ids,
        raw_texts,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels,
        mel_lens,
        max_mel_len,
        pitches,
        energies,
        durations,
    )
