"""Backward-compatible entry point for Accelerate-based training."""

from msst._train_accelerate import *  # noqa: F401,F403


if __name__ == "__main__":
    train_model(None)
