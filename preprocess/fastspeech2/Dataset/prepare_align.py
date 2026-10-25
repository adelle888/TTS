import argparse

import yaml

from preprocess.fastspeech2.Dataset.preprocessor import ljspeech, aishell3, libritts


def main(config):
    if "LJSpeech" in config["dataset"]:
        ljspeech.prepare_align(config)
    if "AISHELL3" in config["dataset"]:
        aishell3.prepare_align(config)
    if "LibriTTS" in config["dataset"]:
        libritts.prepare_align(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/fastspeech2.yaml",
        help="Path to unified TTS configuration",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        full_config = yaml.safe_load(f)

    main(full_config["preprocess"])
