import json
import os

import numpy as np
from torch.utils.data import Dataset

from common.text import phonemes_to_sequence
from common.utils import pad_1D, pad_2D


class FastSpeech2Dataset(Dataset):

    def __init__(self, metadata_file, preprocess_config):
        self.preprocessed_path = (
            preprocess_config["path"]["preprocessed_path"]
        )

        self.basename = []
        self.speaker = []
        self.text = []
        self.raw_text = []

        metadata_path = os.path.join(
            self.preprocessed_path,
            metadata_file,
        )

        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                n, s, t, r = line.rstrip("\n").split("|", 3)

                self.basename.append(n)
                self.speaker.append(s)
                self.text.append(t)
                self.raw_text.append(r)

        with open(
            os.path.join(
                self.preprocessed_path,
                "speakers.json",
            )
        ) as f:
            self.speaker_map = json.load(f)

    def __len__(self):
        return len(self.basename)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]

        speaker_id = self.speaker_map[speaker]

        text = np.array(
            phonemes_to_sequence(self.text[idx]),
            dtype=np.int64,
        )

        prefix = self.preprocessed_path

        mel = np.load(
            os.path.join(
                prefix,
                "mel",
                f"{speaker}-mel-{basename}.npy",
            )
        )

        pitch = np.load(
            os.path.join(
                prefix,
                "pitch",
                f"{speaker}-pitch-{basename}.npy",
            )
        )

        energy = np.load(
            os.path.join(
                prefix,
                "energy",
                f"{speaker}-energy-{basename}.npy",
            )
        )

        duration = np.load(
            os.path.join(
                prefix,
                "duration",
                f"{speaker}-duration-{basename}.npy",
            )
        )

        return {
            "id": basename,
            "speaker": speaker_id,
            "text": text,
            "raw_text": self.raw_text[idx],
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
        }


def fastspeech2_collate_fn(batch):

    ids = [x["id"] for x in batch]

    raw_texts = [
        x["raw_text"]
        for x in batch
    ]

    speakers = np.array(
        [x["speaker"] for x in batch]
    )

    texts = [
        x["text"]
        for x in batch
    ]

    mels = [
        x["mel"]
        for x in batch
    ]

    pitches = [
        x["pitch"]
        for x in batch
    ]

    energies = [
        x["energy"]
        for x in batch
    ]

    durations = [
        x["duration"]
        for x in batch
    ]

    text_lens = np.array([
        x.shape[0]
        for x in texts
    ])

    mel_lens = np.array([
        x.shape[0]
        for x in mels
    ])

    texts = pad_1D(texts)
    mels = pad_2D(mels)
    pitches = pad_1D(pitches)
    energies = pad_1D(energies)
    durations = pad_1D(durations)

    return (
        ids,
        raw_texts,
        speakers,
        texts,
        text_lens,
        max(text_lens),
        mels,
        mel_lens,
        max(mel_lens),
        pitches,
        energies,
        durations,
    )
