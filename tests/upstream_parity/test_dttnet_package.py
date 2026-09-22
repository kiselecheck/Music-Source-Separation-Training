"""Package-side registration and forward smoke checks for DTTNet."""

from types import SimpleNamespace

import torch
from ml_collections import ConfigDict

from msst.models.DTTNet import DPTDFNet
from msst.utils.settings import build_model_from_config, load_config


def namespace(**values):
    return SimpleNamespace(**values)


def tiny_config():
    return namespace(
        audio=namespace(chunk_size=60, dim_f=8, hop_length=4, n_fft=16, num_channels=2),
        model=namespace(
            norm="Identity",
            act="relu",
            num_blocks=2,
            g=16,
            l=1,
            k=3,
            bn=2,
            bias=False,
            block_type="TFC_TDF_Res2",
            bandsequence=namespace(
                rnn_type="LSTM", bidirectional=True, num_layers=1, n_heads=1
            ),
        ),
        training=ConfigDict({"instruments": ["vocals"], "target_instrument": None}),
    )


def test_dttnet_loader_builds_model_and_forward_shape():
    model = build_model_from_config("dttnet", tiny_config()).eval()

    assert isinstance(model, DPTDFNet)
    with torch.inference_mode():
        result = model(torch.randn(1, 2, 60))
    assert result.shape == (1, 1, 2, 60)


def test_upstream_vocal_config_loads_for_dttnet():
    config = load_config("dttnet", "configs/config_vocals_DTTNet.yaml")

    assert config.audio.sample_rate == 44100
    assert config.training.instruments == ["vocals", "other"]
    assert config.model.block_type == "TFC_TDF_Res2"
