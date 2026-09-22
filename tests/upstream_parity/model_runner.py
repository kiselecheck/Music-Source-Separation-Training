"""Run a deterministic model forward pass from a selected source tree."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from ml_collections import ConfigDict


def _namespace(**values):
    return ConfigDict(values)


def _dttnet_config():
    return _namespace(
        audio=_namespace(chunk_size=60, dim_f=8, hop_length=4, n_fft=16, num_channels=2),
        model=_namespace(
            norm="Identity",
            act="relu",
            num_blocks=2,
            g=16,
            l=1,
            k=3,
            bn=2,
            bias=False,
            block_type="TFC_TDF_Res2",
            bandsequence=_namespace(
                rnn_type="LSTM", bidirectional=True, num_layers=1, n_heads=1
            ),
        ),
        training=_namespace(instruments=["vocals"], target_instrument=None),
        inference=_namespace(chunk_size=60, num_overlap=2, batch_size=2),
    )


def _mdx23c_config():
    return _namespace(
        audio=_namespace(dim_f=8, hop_length=4, n_fft=16, num_channels=2),
        model=_namespace(
            norm="Identity",
            act="relu",
            num_subbands=1,
            num_scales=1,
            scale=(2, 2),
            num_blocks_per_scale=1,
            num_channels=2,
            growth=2,
            bottleneck_factor=2,
        ),
        training=_namespace(instruments=["vocals"], target_instrument=None),
        inference=_namespace(chunk_size=60, num_overlap=2, batch_size=2),
    )


def _build_model(case: str, implementation: str):
    if case == "dttnet":
        if implementation == "upstream":
            from models.DTTNet import DPTDFNet
        else:
            from msst.models.DTTNet import DPTDFNet
        config = _dttnet_config()
        return DPTDFNet(config), 60, config

    if case == "mdx23c":
        if implementation == "upstream":
            from models.mdx23c_tfc_tdf_v3 import TFC_TDF_net
        else:
            from msst.models.mdx23c_tfc_tdf_v3 import TFC_TDF_net
        config = _mdx23c_config()
        return TFC_TDF_net(config), 60, config

    if case == "scnet":
        if implementation == "upstream":
            from models.scnet.scnet import SCNet
        else:
            from msst.models.scnet.scnet import SCNet
        return (
            SCNet(
                sources=["vocals", "other"],
                dims=[4, 4],
                nfft=16,
                hop_size=4,
                win_size=16,
                band_SR=[0.25, 0.25, 0.5],
                band_stride=[1, 1, 1],
                band_kernel=[1, 1, 1],
                conv_depths=[1, 1, 1],
                num_dplayer=2,
            ),
            64,
            _namespace(
                audio=_namespace(chunk_size=64, num_channels=2),
                training=_namespace(instruments=["vocals", "other"], target_instrument=None),
                inference=_namespace(chunk_size=64, num_overlap=2, batch_size=2),
            ),
        )

    raise ValueError(f"Unknown parity case: {case}")


def _run_pipeline(case, implementation, model, config, mixture, bigshifts, use_tta):
    if implementation == "upstream":
        from utils.model_utils import apply_tta, bigshifts_wrapper
    else:
        from msst.utils.model_utils import apply_tta, bigshifts_wrapper

    with torch.inference_mode():
        result = bigshifts_wrapper(
            config,
            model,
            mixture.numpy(),
            torch.device("cpu"),
            model_type=case,
            pbar=False,
            bigshifts=bigshifts,
        )
        if use_tta:
            result = apply_tta(
                config,
                model,
                mixture.numpy(),
                result,
                torch.device("cpu"),
                case,
                bigshifts=bigshifts,
                pbar=False,
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=("upstream", "package"), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--case", choices=("dttnet", "mdx23c", "scnet"), required=True)
    parser.add_argument("--pipeline", choices=("forward", "inference"), default="forward")
    parser.add_argument("--bigshifts", type=int, default=2)
    parser.add_argument("--use-tta", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    sys.path.insert(0, str(source_root))
    torch.set_num_threads(1)
    torch.manual_seed(20260921)
    model_started = time.perf_counter()
    model, sample_count, config = _build_model(args.case, args.implementation)
    model.eval()
    model_seconds = time.perf_counter() - model_started

    if args.reference is None:
        generator = torch.Generator().manual_seed(314159)
        length = sample_count if args.pipeline == "forward" else sample_count * 2 + 7
        mixture = torch.randn(1, 2, length, generator=generator)
    else:
        reference = torch.load(args.reference, map_location="cpu", weights_only=False)
        model.load_state_dict(reference["state_dict"], strict=True)
        mixture = reference["input"]

    inference_started = time.perf_counter()
    with torch.inference_mode():
        if args.pipeline == "forward":
            output = model(mixture)
        else:
            output = _run_pipeline(
                args.case,
                args.implementation,
                model,
                config,
                mixture[0],
                args.bigshifts,
                args.use_tta,
            )
    inference_seconds = time.perf_counter() - inference_started

    torch.save(
        {
            "input": mixture.detach().cpu(),
            "output": (
                {key: torch.as_tensor(value).detach().cpu() for key, value in output.items()}
                if isinstance(output, dict)
                else output.detach().cpu()
            ),
            "state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "timings": {
                "model_init_seconds": model_seconds,
                "inference_seconds": inference_seconds,
            },
        },
        args.output,
    )


if __name__ == "__main__":
    main()
