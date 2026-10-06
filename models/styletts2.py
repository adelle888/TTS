"""StyleTTS2 model wrapper for the unified TTS repository."""

from pathlib import Path

from models.styletts2_modules.models import (
    build_model,
    load_ASR_models,
    load_F0_models,
)

from models.styletts2_modules.Utils.PLBERT.util import (
    load_plbert,
)

from common.utils import recursive_munch


ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path):
    path = Path(path)

    if path.is_absolute():
        return str(path)

    return str(ROOT / path)


def build_styletts2(config):
    """Build StyleTTS2 from the unified repository config."""

    asr_path = resolve_path(
        config["ASR_path"]
    )

    asr_config = resolve_path(
        config["ASR_config"]
    )

    f0_path = resolve_path(
        config["F0_path"]
    )

    plbert_dir = resolve_path(
        config["PLBERT_dir"]
    )

    text_aligner = load_ASR_models(
        asr_path,
        asr_config,
    )

    pitch_extractor = load_F0_models(
        f0_path,
    )

    plbert = load_plbert(
        plbert_dir,
    )

    model_params = recursive_munch(
        config["model_params"],
    )

    model = build_model(
        model_params,
        text_aligner,
        pitch_extractor,
        plbert,
    )

    return model
