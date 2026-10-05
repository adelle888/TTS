"""StyleTTS2 model wrapper.

The upstream StyleTTS2 implementation is kept under StyleTTS2/
because the architecture depends on ASR, JDC, PLBERT, diffusion,
decoder and discriminator modules.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STYLE_ROOT = ROOT / "StyleTTS2"

if str(STYLE_ROOT) not in sys.path:
    sys.path.insert(0, str(STYLE_ROOT))

from Model.models import (
    build_model,
    load_ASR_models,
    load_F0_models,
)

from Model.Utils.PLBERT.util import load_plbert


def build_styletts2(config):
    """Build StyleTTS2 from the unified repository config."""

    text_aligner = load_ASR_models(
        config["ASR_path"],
        config["ASR_config"],
    )

    pitch_extractor = load_F0_models(
        config["F0_path"]
    )

    plbert = load_plbert(
        config["PLBERT_dir"]
    )

    from utils import recursive_munch

    model_params = recursive_munch(
        config["model_params"]
    )

    model = build_model(
        model_params,
        text_aligner,
        pitch_extractor,
        plbert,
    )

    return model
