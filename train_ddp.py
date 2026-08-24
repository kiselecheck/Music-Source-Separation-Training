"""Backward-compatible entry point for DDP training."""

from msst._train_ddp import *  # noqa: F401,F403


if __name__ == "__main__":
    train_model_ddp()
