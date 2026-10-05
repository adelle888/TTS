import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from common.text import phonemes_to_sequence


def pad_1d(inputs, pad_value=0):
    max_len = max(len(x) for x in inputs)

    return np.stack([
        np.pad(
            x,
            (0, max_len - len(x)),
            mode="constant",
            constant_values=pad_value,
        )
        for x in inputs
    ])


def pad_2d(inputs):
    max_len = max(x.shape[0] for x in inputs)
    feature_dim = inputs[0].shape[1]

    output = np.zeros(
        (len(inputs), max_len, feature_dim),
        dtype=np.float32,
    )

    for i, x in enumerate(inputs):
        output[i, :x.shape[0], :] = x

    return output


def get_mask_from_lengths(lengths, max_len=None):
    if max_len is None:
        max_len = int(lengths.max().item())

    ids = torch.arange(
        max_len,
        device=lengths.device,
    ).unsqueeze(0)

    return ids >= lengths.unsqueeze(1)


class EfficientSpeechDataset(Dataset):
    def __init__(self, filename, preprocess_config):
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]

        self.max_text_length = (
            preprocess_config["preprocessing"]["text"]["max_length"]
        )

        (
            self.basename,
            self.speaker,
            self.text,
            self.raw_text,
        ) = self.process_meta(filename)

        speakers_path = os.path.join(
            self.preprocessed_path,
            "speakers.json",
        )

        if os.path.exists(speakers_path):
            with open(speakers_path, "r") as f:
                self.speaker_map = json.load(f)
        else:
            self.speaker_map = {}

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        raw_text = self.raw_text[idx]

        # EfficientSpeech LJSpeech preprocessing stores
        # ARPAbet phonemes in {...}
        phoneme = np.array(
            phonemes_to_sequence(self.text[idx]),
            dtype=np.int64,
        )

        mel = np.load(
            os.path.join(
                self.preprocessed_path,
                "mel",
                f"{speaker}-mel-{basename}.npy",
            )
        )

        pitch = np.load(
            os.path.join(
                self.preprocessed_path,
                "pitch",
                f"{speaker}-pitch-{basename}.npy",
            )
        )

        energy = np.load(
            os.path.join(
                self.preprocessed_path,
                "energy",
                f"{speaker}-energy-{basename}.npy",
            )
        )

        duration = np.load(
            os.path.join(
                self.preprocessed_path,
                "duration",
                f"{speaker}-duration-{basename}.npy",
            )
        )

        x = {
            "phoneme": phoneme,
            "text": raw_text,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
        }

        y = {
            "mel": mel,
        }

        return x, y

    def process_meta(self, filename):
        metadata_path = os.path.join(
            self.preprocessed_path,
            filename,
        )

        basename = []
        speaker = []
        text = []
        raw_text = []

        with open(
            metadata_path,
            "r",
            encoding="utf-8",
        ) as f:
            for line in f:
                n, s, t, r = line.rstrip("\n").split("|")

                if len(r) > self.max_text_length:
                    continue

                basename.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)

        return basename, speaker, text, raw_text


def efficientspeech_collate_fn(batch):
    x, y = zip(*batch)

    # Same behavior as original EfficientSpeech:
    # sort batch by phoneme length descending.
    lengths = np.array([
        item["phoneme"].shape[0]
        for item in x
    ])

    idxs = np.argsort(-lengths).tolist()

    phonemes = [
        x[i]["phoneme"]
        for i in idxs
    ]

    texts = [
        x[i]["text"]
        for i in idxs
    ]

    pitches = [
        x[i]["pitch"]
        for i in idxs
    ]

    energies = [
        x[i]["energy"]
        for i in idxs
    ]

    durations = [
        x[i]["duration"]
        for i in idxs
    ]

    mels = [
        y[i]["mel"]
        for i in idxs
    ]

    phoneme_lens = np.array([
        p.shape[0]
        for p in phonemes
    ])

    mel_lens = np.array([
        m.shape[0]
        for m in mels
    ])

    phonemes = torch.from_numpy(
        pad_1d(phonemes)
    ).long()

    pitches = torch.from_numpy(
        pad_1d(pitches)
    ).float()

    energies = torch.from_numpy(
        pad_1d(energies)
    ).float()

    durations = torch.from_numpy(
        pad_1d(durations)
    ).long()

    mels = torch.from_numpy(
        pad_2d(mels)
    ).float()

    phoneme_lens = torch.from_numpy(
        phoneme_lens
    ).long()

    mel_lens = torch.from_numpy(
        mel_lens
    ).long()

    phoneme_mask = get_mask_from_lengths(
        phoneme_lens
    )

    mel_mask = get_mask_from_lengths(
        mel_lens
    )

    x_batch = {
        "phoneme": phonemes,
        "phoneme_len": phoneme_lens,
        "phoneme_mask": phoneme_mask,
        "text": texts,
        "mel_len": mel_lens,
        "mel_mask": mel_mask,
        "pitch": pitches,
        "energy": energies,
        "duration": durations,
    }

    y_batch = {
        "mel": mels,
    }

    return x_batch, y_batch
