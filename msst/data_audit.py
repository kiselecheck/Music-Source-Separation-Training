"""Sample actual single-target training pairs for obvious task mismatches."""
from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import DataLoader


def inspect_training_pairs(config, args, metadata, metadata_path, count):
    """Inspect sampled batches on CPU, preserving caller RNG state.

    Identity means residual RMS <= 0.1% of target RMS, for a non-silent
    target. It is a diagnostic, not a claim that a pair is unusable.
    """
    from .utils.dataset import MSSDataset, _training_collate_fn

    if not getattr(config.training, 'target_instrument', None):
        raise ValueError('pair audit requires training.target_instrument (single-target training)')
    states = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    try:
        dataset = MSSDataset(config, args.data_path, metadata_path=metadata_path,
                             dataset_type=args.dataset_type, batch_size=1,
                             verbose=False, metadata=metadata)
        loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0,
                            collate_fn=_training_collate_fn(config, args.dataset_type))
        if len(loader) == 0:
            raise ValueError('training dataset has no samples')
        iterator = iter(loader)
        identity, silent, nonfinite = 0, 0, 0
        residuals = []
        for _ in range(count):
            try:
                data = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                data = next(iterator)
            target, mixture = data[:2]
            if target.ndim == mixture.ndim + 1 and target.shape[1] == 1:
                target = target[:, 0]
            if target.shape != mixture.shape:
                raise ValueError(f'expected a single target with mixture shape, got {tuple(target.shape)} and {tuple(mixture.shape)}')
            target, mixture = target.double(), mixture.double()
            if not torch.isfinite(target).all() or not torch.isfinite(mixture).all():
                nonfinite += 1
                continue
            target_rms = torch.sqrt(torch.mean(target.square())).item()
            if target_rms <= 1e-8:
                silent += 1
                continue
            residual_rms = torch.sqrt(torch.mean((mixture - target).square())).item()
            ratio = residual_rms / target_rms
            residuals.append(ratio)
            if ratio <= 0.001:
                identity += 1
        return {'samples': count, 'identity_like': identity, 'silent_targets': silent,
                'nonfinite': nonfinite, 'non_silent_samples': len(residuals),
                'median_residual_to_target_rms': float(np.median(residuals)) if residuals else None}
    finally:
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.set_rng_state(states[2])
