import os

import numpy as np
import pytest
import torch
from torch import nn

from msst.models.bs_roformer.mel_band_roformer import MelBandRoformer
from msst.utils.model_utils import compile_model, is_compiled_model
from msst.utils.metrics import get_metrics


def test_compile_model_uses_torch_compile_for_generic_models(monkeypatch):
    model = nn.Linear(4, 4)
    calls = []

    def fake_compile(module, *, dynamic):
        calls.append((module, dynamic))
        return module

    monkeypatch.setattr(torch, "compile", fake_compile)

    assert compile_model(model) is model
    assert calls == [(model, False)]


def test_compile_model_ignores_missing_cc_and_enables_graph_cache(monkeypatch):
    monkeypatch.setenv("CC", "missing-compiler --flag")
    monkeypatch.delenv("TORCHINDUCTOR_FX_GRAPH_CACHE", raising=False)
    monkeypatch.setattr("msst.utils.model_utils.shutil.which", lambda _name: None)
    monkeypatch.setattr(torch, "compile", lambda module, *, dynamic: module)

    compile_model(nn.Linear(2, 2))

    assert "CC" not in os.environ
    assert os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] == "1"


def test_compile_model_marks_roformer_submodules(monkeypatch):
    calls = []

    def fake_compile(module, *, dynamic):
        calls.append((module, dynamic))

    monkeypatch.setattr(nn.Module, "compile", fake_compile, raising=False)

    class RoformerParts(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.ModuleList([nn.Linear(2, 2)])])
            self.band_split = nn.Linear(2, 2)
            self.mask_estimators = nn.ModuleList([nn.Linear(2, 2)])

    model = RoformerParts()

    assert compile_model(model) is model
    assert len(calls) == 3
    assert all(dynamic is False for _, dynamic in calls)
    assert is_compiled_model(model)


def test_mel_band_roformer_single_stem_training_loss_keeps_stem_axis():
    model = MelBandRoformer(
        dim=8,
        depth=0,
        stereo=False,
        num_stems=1,
        num_bands=4,
        dim_head=4,
        heads=2,
        flash_attn=False,
        sample_rate=8000,
        stft_n_fft=16,
        stft_hop_length=8,
        stft_win_length=16,
        multi_stft_resolution_loss_weight=0.5,
        multi_stft_resolutions_window_sizes=(16, 8),
        multi_stft_hop_size=4,
    )

    mixture = torch.randn(1, 1, 64)
    target = torch.randn(1, 1, 1, 64)
    loss = model(mixture, target)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_sdr_matches_cpu_sdr():
    rng = np.random.default_rng(17)
    reference = rng.standard_normal((2, 8192)).astype(np.float32)
    estimate = (reference + 0.1 * rng.standard_normal((2, 8192))).astype(np.float32)
    mixture = reference + estimate

    cpu_metrics = get_metrics(["sdr", "k_sdr"], reference, estimate, mixture, "cpu")
    cuda_metrics = get_metrics(
        ["sdr", "k_sdr"], reference, estimate, mixture, "cuda:0"
    )

    assert cuda_metrics["sdr"] == pytest.approx(cpu_metrics["sdr"], abs=1e-6)
    assert cuda_metrics["k_sdr"] == pytest.approx(cpu_metrics["k_sdr"], abs=1e-6)
