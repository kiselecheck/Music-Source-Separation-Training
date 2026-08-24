"""Backward-compatible entry point for the packaged training runtime."""

from msst._training import *  # noqa: F401,F403


if __name__ == "__main__":
    train_model(None)
