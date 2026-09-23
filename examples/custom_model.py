"""A small, trainable separator that uses the MSST custom-model interface.

Import ``TinySeparator`` and register it in the process that calls MSST:

    import msst
    from examples.custom_model import TinySeparator

    msst.register_model("tiny_separator", TinySeparator)
"""

from __future__ import annotations

import torch
from torch import nn

import msst


class TinySeparator(msst.MSSTBaseModel):
    """Predict the first of two stems; assign the residual to the second."""

    def __init__(self, config):
        super().__init__()
        self.channels = int(config.audio.num_channels)
        self.stems = len(config.training.instruments)
        if self.stems != 2:
            raise ValueError(
                "TinySeparator expects exactly two configured instruments"
            )

        self.predict_first = nn.Sequential(
            nn.Conv1d(self.channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(8, self.channels, kernel_size=1),
        )

    def separate(
        self, mixture: torch.Tensor, *, active_stem_ids=None
    ) -> torch.Tensor:
        if active_stem_ids is not None:
            raise ValueError("TinySeparator does not support selective stems")
        if mixture.ndim != 3 or mixture.shape[1] != self.channels:
            raise ValueError(
                "expected mixture with shape "
                f"[batch, {self.channels}, time], got {tuple(mixture.shape)}"
            )

        first = self.predict_first(mixture)
        second = mixture - first
        return torch.stack((first, second), dim=1)


__all__ = ["TinySeparator"]
