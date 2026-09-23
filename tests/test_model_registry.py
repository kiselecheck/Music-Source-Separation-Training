from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from ml_collections import ConfigDict
from torch import nn

import msst
from msst.utils.model_utils import demix
from msst.utils.settings import get_model_from_config
from msst._training import compute_training_loss


MODEL_NAME = "test_extension_model"


class TinyExtensionModel(msst.MSSTBaseModel):
    def __init__(self, config):
        super().__init__()
        self.channels = int(config.audio.num_channels)
        self.stems = len(config.training.instruments)
        self.projection = nn.Conv1d(
            self.channels, self.channels * self.stems, kernel_size=1, bias=False
        )

    def forward(self, mix: torch.Tensor) -> torch.Tensor:
        projected = self.projection(mix)
        return projected.reshape(
            mix.shape[0], self.stems, self.channels, mix.shape[-1]
        )


@pytest.fixture(autouse=True)
def clean_registry():
    msst.unregister_model(MODEL_NAME)
    yield
    msst.unregister_model(MODEL_NAME)


def _config_text() -> str:
    return """
audio:
  num_channels: 2
  chunk_size: 16
  sample_rate: 44100
training:
  instruments: [vocals, other]
  target_instrument: null
inference:
  chunk_size: 16
  batch_size: 2
  num_overlap: 1
"""


def test_external_model_can_register_and_load_without_model_section(tmp_path: Path):
    config_path = tmp_path / "custom.yaml"
    config_path.write_text(_config_text(), encoding="utf-8")

    registration = msst.register_model(MODEL_NAME, TinyExtensionModel)
    assert registration.name == MODEL_NAME
    assert MODEL_NAME in msst.registered_models()

    model, config = get_model_from_config(MODEL_NAME, str(config_path))

    assert isinstance(model, TinyExtensionModel)
    assert config.training.instruments == ["vocals", "other"]
    with torch.inference_mode():
        output = model(torch.randn(1, 2, 16))
    assert output.shape == (1, 2, 2, 16)


def test_registered_model_uses_normal_generic_demix_path(tmp_path: Path):
    config_path = tmp_path / "custom.yaml"
    config_path.write_text(_config_text(), encoding="utf-8")
    msst.register_model(MODEL_NAME, TinyExtensionModel)
    model, config = get_model_from_config(MODEL_NAME, str(config_path))

    with torch.inference_mode():
        separated = demix(
            config,
            model,
            torch.randn(2, 23),
            torch.device("cpu"),
            MODEL_NAME,
        )

    assert set(separated) == {"vocals", "other"}
    assert separated["vocals"].shape == (2, 23)
    assert separated["other"].shape == (2, 23)


def test_registered_model_uses_normal_standard_training_loss(tmp_path: Path):
    config_path = tmp_path / "custom.yaml"
    config_path.write_text(_config_text(), encoding="utf-8")
    msst.register_model(MODEL_NAME, TinyExtensionModel)
    model, config = get_model_from_config(MODEL_NAME, str(config_path))
    args = SimpleNamespace(model_type=MODEL_NAME, use_standard_loss=False)

    target = torch.randn(1, 2, 2, 16)
    mixture = torch.randn(1, 2, 16)
    loss, active_stem_mask = compute_training_loss(
        model,
        config,
        args,
        (target, mixture),
        torch.device("cpu"),
        [],
        lambda prediction, expected, _mixture, _mask: (
            prediction - expected
        ).square().mean(),
        use_amp=False,
    )

    assert active_stem_mask is None
    assert loss.ndim == 0
    loss.backward()


def test_registry_rejects_builtin_replacement_and_duplicates():
    with pytest.raises(ValueError, match="built-in"):
        msst.register_model("mdx23c", TinyExtensionModel)

    msst.register_model(MODEL_NAME, TinyExtensionModel)
    with pytest.raises(ValueError, match="already registered"):
        msst.register_model(MODEL_NAME, TinyExtensionModel)


def test_registry_validates_names_and_supports_factory():
    with pytest.raises(ValueError, match="model name"):
        msst.register_model("not a model", TinyExtensionModel)

    def factory(config, *, pretrained=False):
        assert pretrained is True
        return TinyExtensionModel(config)

    msst.register_model(MODEL_NAME, factory)
    config = ConfigDict(
        {
            "audio": {"num_channels": 2},
            "training": {"instruments": ["vocals"], "target_instrument": None},
        }
    )
    from msst.utils.settings import build_model_from_config

    built = build_model_from_config(MODEL_NAME, config, pretrained=True)
    assert isinstance(built, TinyExtensionModel)
