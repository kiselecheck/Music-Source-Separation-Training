"""Backward-compatible entry point for the packaged ensemble runtime."""

from msst._ensemble import *  # noqa: F401,F403


if __name__ == "__main__":
    ensemble_files(None)
