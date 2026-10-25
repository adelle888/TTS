import argparse
import yaml
from preprocess.efficientspeech.Dataset.preprocessor import ljspeech


def main(config):
    if "LJSpeech" in config["dataset"]:
        ljspeech.prepare_align(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/efficientspeech.yaml",
        help="Path to unified TTS configuration",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        full_config = yaml.safe_load(f)

    main(full_config["preprocess"])
