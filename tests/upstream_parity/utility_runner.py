"""Compare shared audio, dataset, loss, and one-step optimizer behavior."""

from __future__ import annotations

import argparse
import sys
import time
import types
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from ml_collections import ConfigDict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=("upstream", "package"), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.source_root.resolve()))
    torch.set_num_threads(1)
    torch.manual_seed(107)
    random = np.random.default_rng(712)

    if args.implementation == "upstream":
        from utils.audio_utils import denormalize_audio, normalize_audio, read_audio_transposed
        from utils.losses import masked_loss, spec_masked_loss, spec_rmse_loss
        from utils.model_utils import get_optimizer
        # The upstream module imports augmentation packages at module scope;
        # load_chunk itself does not use them, so placeholders let this focused
        # parity check run in an inference-only environment.
        for optional_module in ("audiomentations", "pedalboard"):
            try:
                __import__(optional_module)
            except ModuleNotFoundError as error:
                if error.name != optional_module:
                    raise
                sys.modules[optional_module] = types.ModuleType(optional_module)
        try:
            from utils.dataset import load_chunk
        except ModuleNotFoundError as error:
            if error.name != "audiomentations":
                raise
            load_chunk = None
    else:
        from msst.utils.audio_utils import denormalize_audio, normalize_audio, read_audio_transposed
        from msst.utils.losses import masked_loss, spec_masked_loss, spec_rmse_loss
        from msst.utils.model_utils import get_optimizer
        try:
            from msst.utils.dataset import load_chunk
        except ModuleNotFoundError as error:
            if error.name != "audiomentations":
                raise
            load_chunk = None

    timings: dict[str, float] = {}
    started = time.perf_counter()
    waveform = random.standard_normal((2, 4096)).astype(np.float32) * 0.17
    waveform += np.asarray([[0.13], [-0.07]], dtype=np.float32)
    normalized, norm_params = normalize_audio(waveform)
    restored = denormalize_audio(normalized, norm_params)
    timings["normalization_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    audio_data, audio_rate = read_audio_transposed(str(args.audio))
    audio_info = sf.info(args.audio)
    chunk = (
        load_chunk(
            str(args.audio),
            length=audio_info.frames,
            chunk_size=audio_info.frames + 113,
            offset=47,
            target_channels=2,
        )
        if load_chunk is not None
        else None
    )
    timings["audio_io_seconds"] = time.perf_counter() - started

    generator = torch.Generator().manual_seed(812)
    prediction = torch.randn((2, 4, 2, 1024), generator=generator) + 0.08
    target = torch.randn((2, 4, 2, 1024), generator=generator)
    stft_config = {
        "n_fft": 128,
        "hop_length": 32,
        "win_length": 128,
        "window": torch.hann_window(128),
        "center": True,
        "pad_mode": "reflect",
    }
    loss_results: dict[str, torch.Tensor] = {}
    started = time.perf_counter()
    for loss_name, loss_function, options in (
        ("masked_loss", masked_loss, {"q": 0.8, "coarse": True}),
        ("spec_rmse_loss", spec_rmse_loss, {"stft_config": stft_config}),
        (
            "spec_masked_loss",
            spec_masked_loss,
            {"stft_config": stft_config, "q": 0.8, "coarse": True},
        ),
    ):
        estimate = prediction.clone().requires_grad_(True)
        loss = loss_function(estimate, target, **options)
        loss.backward()
        loss_results[f"{loss_name}_value"] = loss.detach()
        loss_results[f"{loss_name}_gradient"] = estimate.grad.detach()
    timings["losses_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Tanh(), torch.nn.Linear(16, 4))
    optimizer_config = ConfigDict(
        {"training": {"optimizer": "adam", "lr": 0.0003}}
    )
    optimizer = get_optimizer(optimizer_config, model)
    train_input = torch.randn((5, 8), generator=generator)
    train_target = torch.randn((5, 4), generator=generator)
    optimizer.zero_grad()
    train_loss = torch.nn.functional.mse_loss(model(train_input), train_target)
    train_loss.backward()
    optimizer.step()
    timings["optimizer_step_seconds"] = time.perf_counter() - started

    arrays = {
        "normalized": torch.from_numpy(normalized.copy()),
        "restored": torch.from_numpy(restored.copy()),
        "normalization_mean": torch.as_tensor(norm_params["mean"]),
        "normalization_std": torch.as_tensor(norm_params["std"]),
        "audio_data": torch.from_numpy(np.asarray(audio_data).copy()),
        "audio_rate": torch.as_tensor(audio_rate),
        "train_loss": train_loss.detach(),
    }
    if chunk is not None:
        arrays["loaded_chunk"] = torch.from_numpy(chunk.copy())
    arrays.update(loss_results)
    arrays.update(
        {f"updated_{name}": parameter.detach() for name, parameter in model.state_dict().items()}
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "arrays": arrays,
            "timings": timings,
            "dataset_load_chunk_available": load_chunk is not None,
        },
        args.output,
    )


if __name__ == "__main__":
    main()
