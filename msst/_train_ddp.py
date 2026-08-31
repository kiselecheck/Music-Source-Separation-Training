# coding: utf-8

import contextlib
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from msst._training import train_model
from msst.utils.settings import (
    cleanup_ddp,
    find_free_port,
    parse_args_train,
    validate_train_args,
)


def train_model_single(rank: int, world_size: int, args=None):
    """
    Trains the model based on the provided arguments, including data preparation, optimizer setup,
    and loss calculation. The model is trained for multiple epochs with logging via wandb.

    Args:
        world_size:
        rank:
        args: Command-line arguments containing configuration paths, hyperparameters, and other settings.

    Returns:
        None
    """
    try:
        if rank == 0:
            train_model(args, rank, world_size)
            return

        log_directory = Path(args.results_path) / "ddp_logs"
        log_directory.mkdir(parents=True, exist_ok=True)
        log_path = log_directory / f"rank_{rank}.log"
        with log_path.open("a", encoding="utf-8", buffering=1) as rank_log:
            with contextlib.redirect_stdout(rank_log), contextlib.redirect_stderr(rank_log):
                train_model(args, rank, world_size)
    finally:
        if dist.is_initialized():
            cleanup_ddp()


def train_model_ddp(args=None):
    parsed_args = validate_train_args(parse_args_train(args))
    if not torch.cuda.is_available():
        raise RuntimeError("DDP training requires CUDA")
    available_devices = torch.cuda.device_count()
    invalid_devices = [
        device_id
        for device_id in parsed_args.device_ids
        if device_id >= available_devices
    ]
    if invalid_devices:
        raise ValueError(
            f"CUDA device ids {invalid_devices} are unavailable; "
            f"this process sees {available_devices} device(s)"
        )
    world_size = len(parsed_args.device_ids)
    if world_size == 0:
        raise ValueError("DDP requires at least one device id")
    os.makedirs(parsed_args.results_path, exist_ok=True)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(find_free_port())
    try:
        mp.spawn(
            train_model_single,
            args=(world_size, parsed_args),
            nprocs=world_size,
            join=True,
        )
    except Exception:
        if dist.is_initialized():
            cleanup_ddp()
        raise


if __name__ == "__main__":
    train_model_ddp()
