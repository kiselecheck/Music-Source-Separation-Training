"""Backward-compatible entry point for the packaged inference runtime."""

from msst._inference import *  # noqa: F401,F403


if __name__ == "__main__":
    proc_folder(None)
