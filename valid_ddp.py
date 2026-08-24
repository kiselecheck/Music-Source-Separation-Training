"""Backward-compatible entry point for DDP validation."""

from msst._valid_ddp import *  # noqa: F401,F403


if __name__ == "__main__":
    check_validation()
