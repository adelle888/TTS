import argparse
from pathlib import Path

import yaml

from preprocess.efficientspeech.Dataset.preprocessor.preprocessor import Preprocessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/efficientspeech.yaml",
        help="Path to unified TTS configuration",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()

    with config_path.open("r", encoding="utf-8") as f:
        full_config = yaml.safe_load(f)

    config = full_config["preprocess"]

    preprocessor = Preprocessor(config)
    preprocessor.build_from_path()


if __name__ == "__main__":
    main()
