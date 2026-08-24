"""Backward-compatible entry point for the packaged validation runtime."""

from msst._validation import *  # noqa: F401,F403


if __name__ == "__main__":
    check_validation(None)
