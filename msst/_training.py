# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'
import argparse
import json
import os
import sys

import numpy as np
from tqdm.auto import tqdm
import torch
import wandb
import torch.nn as nn
from torch.utils.data import DataLoader
from ml_collections import ConfigDict
from typing import List, Callable, Union
import torch.distributed as dist
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from msst.utils.settings import get_scheduler, parse_args_train, initialize_environment_ddp, \
    initialize_environment, get_model_from_config, validate_train_args, wandb_init
from msst.utils.model_utils import save_weights, normalize_batch, \
    save_last_weights, initialize_model_and_device
from msst.models.registry import model_uses_internal_loss

from msst._validation import valid_multi_gpu, valid
from msst.checkpoints import build_checkpoint_metadata


def log_epoch_history(args, epoch, optimizer, scheduler, best_metric,
                      metrics_avg, all_metrics, all_losses):
    """Append a compact, machine-readable epoch record to the results folder."""
    loss_values = all_losses.get(f'epoch_{epoch}', [])
    per_stem = {}
    for metric_name, stem_metrics in (all_metrics or {}).items():
        per_stem[metric_name] = {}
        for stem_name, values in stem_metrics.items():
            if isinstance(values, dict):
                values = list(values.values())
            values_array = np.asarray(values, dtype=np.float64)
            per_stem[metric_name][stem_name] = {
                'mean': float(values_array.mean()),
                'std': float(values_array.std()),
                'count': int(values_array.size),
            }

    record = {
        'epoch': int(epoch),
        'train_loss': float(np.mean(loss_values)) if loss_values else None,
        'learning_rate': float(optimizer.param_groups[0]['lr']),
        'scheduler': scheduler.name,
        'best_metric': float(best_metric),
        'metrics_avg': {
            name: float(value) for name, value in metrics_avg.items()
        },
        'per_stem': per_stem,
    }
    history_path = os.path.join(args.results_path, 'training_history.jsonl')
    with open(history_path, 'a', encoding='utf-8') as history_file:
        history_file.write(json.dumps(record, ensure_ascii=False) + '\n')

def _shared_active_stem_ids(active_stem_mask):
    """Convert a per-sample mask to IDs for models with shared output heads."""
    if active_stem_mask.ndim != 2:
        raise ValueError(
            "active_stem_mask must have shape [batch, stems], "
            f"got {tuple(active_stem_mask.shape)}"
        )
    if active_stem_mask.shape[0] > 1:
        first_row = active_stem_mask[:1].expand_as(active_stem_mask)
        if not torch.equal(active_stem_mask, first_row):
            raise ValueError(
                "Internal BS-Roformer-style loss requires the same active stems "
                "for every sample in a physical batch"
            )
    active_stem_ids = torch.nonzero(
        active_stem_mask[0], as_tuple=False
    ).flatten().tolist()
    if not active_stem_ids:
        raise ValueError("active_stem_mask does not contain any active stems")
    return active_stem_ids


def build_step_log(loss_value, average_loss, step, active_stem_mask, device):
    values = {
        'loss': 100 * loss_value,
        'avg_loss': 100 * average_loss,
        'i': step,
    }
    if isinstance(active_stem_mask, torch.Tensor):
        stems_per_sample = active_stem_mask.sum(dim=1)
        values.update({
            'active_stem_pairs': int(stems_per_sample.sum().item()),
            'active_stems_union': int(active_stem_mask.any(dim=0).sum().item()),
            'active_stems_per_sample_mean': float(stems_per_sample.float().mean().item()),
            'active_stems_per_sample_min': int(stems_per_sample.min().item()),
            'active_stems_per_sample_max': int(stems_per_sample.max().item()),
        })
    if device.type == 'cuda':
        values.update({
            'gpu_memory_allocated_gb': torch.cuda.memory_allocated(device) / (1024 ** 3),
            'gpu_memory_reserved_gb': torch.cuda.memory_reserved(device) / (1024 ** 3),
            'gpu_max_memory_allocated_gb': torch.cuda.max_memory_allocated(device) / (1024 ** 3),
        })
    return values


def forward_step(x, y, active_stem_mask, get_internal_loss, model, multi_loss,
                 device_ids, model_type, model_active_stem_ids=None):
    if get_internal_loss:
        active_stem_ids = active_stem_mask
        if isinstance(active_stem_mask, torch.Tensor):
            active_stem_ids = _shared_active_stem_ids(active_stem_mask)
        loss = model(x, y, active_stem_ids=active_stem_ids)
        if isinstance(device_ids, (list, tuple)):
            loss = loss.mean()
        return loss
    else:
        loss_active_stem_mask = active_stem_mask
        if model_type == 'scnet' and isinstance(active_stem_mask, torch.Tensor):
            if active_stem_mask.ndim != 2 or tuple(active_stem_mask.shape) != tuple(y.shape[:2]):
                raise ValueError(
                    "SCNet active_stem_mask must match target [batch, stems], "
                    f"got {tuple(active_stem_mask.shape)} for target {tuple(y.shape)}"
                )
            if model_active_stem_ids is None:
                active_stem_ids_tensor = torch.nonzero(
                    active_stem_mask.any(dim=0), as_tuple=False
                ).flatten()
                if active_stem_ids_tensor.numel() == 0:
                    raise ValueError("active_stem_mask does not contain any active stems")
                model_active_stem_ids = active_stem_ids_tensor.tolist()
                y = y.index_select(1, active_stem_ids_tensor)
                loss_active_stem_mask = active_stem_mask.index_select(
                    1, active_stem_ids_tensor
                )
            y_ = model(x, active_stem_ids=model_active_stem_ids)
        else:
            y_ = model(x)

        # Uncomment this if you have problem with gradients with enabled amp
        if 0:
            # Temporarily disable AMP (switch to float32) only for loss calculation
            with torch.autocast(device_type=y_.device.type, enabled=False):
                # Force cast predictions and targets to float32
                x_float = x.float() if x is not None else None
                return multi_loss(
                    y_.float(), y.float(), x_float, loss_active_stem_mask
                )

        return multi_loss(y_, y, x, loss_active_stem_mask)


def compute_training_loss(
    model: torch.nn.Module,
    config: ConfigDict,
    args: argparse.Namespace,
    data,
    device: torch.device,
    device_ids,
    multi_loss,
    use_amp: bool,
):
    """Compute one real training loss using the canonical batch semantics.

    This function is shared by full preflight checks and actual training, so a
    successful check exercises the same normalization, stem masking, forward,
    and loss path used inside an epoch.
    """

    if len(data) == 3:
        batch, mixes, active_stem_mask = data
    elif len(data) == 2:
        batch, mixes = data
        active_stem_mask = None
    else:
        raise ValueError(f'Expected a 2- or 3-item training batch, got {len(data)}')

    model_active_stem_ids = None
    if args.model_type == 'scnet' and isinstance(active_stem_mask, torch.Tensor):
        if active_stem_mask.ndim != 2 or tuple(active_stem_mask.shape) != tuple(batch.shape[:2]):
            raise ValueError(
                "SCNet active_stem_mask must match target [batch, stems], "
                f"got {tuple(active_stem_mask.shape)} for target {tuple(batch.shape)}"
            )
        active_stem_ids_tensor = torch.nonzero(
            active_stem_mask.any(dim=0), as_tuple=False
        ).flatten()
        if active_stem_ids_tensor.numel() == 0:
            raise ValueError("active_stem_mask does not contain any active stems")
        model_active_stem_ids = active_stem_ids_tensor.tolist()
        batch = batch.index_select(1, active_stem_ids_tensor)
        active_stem_mask = active_stem_mask.index_select(
            1, active_stem_ids_tensor
        )

    x = mixes.to(device)
    y = batch.to(device)
    if isinstance(active_stem_mask, torch.Tensor):
        active_stem_mask = active_stem_mask.to(device=device, dtype=torch.bool)

    if getattr(config.training, 'normalize', False):
        x, y = normalize_batch(x, y)
    get_internal_loss = (
        model_uses_internal_loss(args.model_type)
        and not args.use_standard_loss
    )
    with torch.autocast(
        device_type=device.type,
        enabled=bool(use_amp and device.type == 'cuda'),
    ):
        loss = forward_step(
            x,
            y,
            active_stem_mask,
            get_internal_loss,
            model,
            multi_loss,
            device_ids,
            args.model_type,
            model_active_stem_ids,
        )
    return loss, active_stem_mask



def train_one_epoch(model: torch.nn.Module, config: ConfigDict, args: argparse.Namespace,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, device_ids: List[int], epoch: int, use_amp: bool,
                    scaler,
                    scheduler,
                    gradient_accumulation_steps: int, train_loader: torch.utils.data.DataLoader,
                    multi_loss: Callable[[torch.Tensor, torch.Tensor, torch.Tensor,], torch.Tensor], all_losses=None, world_size=None, ema_model=None, safe_mode=None) -> None:
    """
    Train the model for one epoch.

    Args:
        world_size:
        scheduler:
        model: The model to train.
        config: Configuration object containing training parameters.
        args: Command-line arguments with specific settings (e.g., model type).
        optimizer: Optimizer used for training.
        device: Device to run the model on (CPU or GPU).
        device_ids: List of GPU device IDs if using multiple GPUs.
        epoch: The current epoch number.
        use_amp: Whether to use automatic mixed precision (AMP) for training.
        scaler: Scaler for AMP to manage gradient scaling.
        gradient_accumulation_steps: Number of gradient accumulation steps before updating the optimizer.
        train_loader: DataLoader for the training dataset.
        multi_loss: The loss function to use during training.

    Returns:
        None
    """
    ddp = True if world_size else False
    should_print = not dist.is_initialized() or dist.get_rank() == 0
    model.train()
    if not ddp:
        model.to(device)
    if should_print:
        print(f'Train epoch: {epoch} Learning rate: {optimizer.param_groups[0]["lr"]}')
        sys.stdout.flush()
    loss_val = 0.
    total = 0
    all_losses[f'epoch_{epoch}'] = []

    if ddp:
        pbar = tqdm(train_loader,
                    dynamic_ncols=True) if dist.get_rank() == 0 else train_loader
    else:
        pbar = tqdm(train_loader)

    for i, data in enumerate(pbar):
        if safe_mode:
            try:
                loss, active_stem_mask = compute_training_loss(
                    model,
                    config,
                    args,
                    data,
                    device,
                    device_ids,
                    multi_loss,
                    use_amp,
                )
            except Exception as e:
                print(f'Error: {e}')
                continue
        else:
            loss, active_stem_mask = compute_training_loss(
                model,
                config,
                args,
                data,
                device,
                device_ids,
                multi_loss,
                use_amp,
            )
        loss /= gradient_accumulation_steps
        scaler.scale(loss).backward()

        if ((i + 1) % gradient_accumulation_steps == 0) or (i == len(train_loader) - 1):

            scaler.unscale_(optimizer)

            if config.training.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            if ema_model is not None:
                if ddp:
                    ema_model.update_parameters(model.module)
                else:
                    ema_model.update_parameters(model)

            if scheduler.name in ['linear_scheduler', 'cosine_scheduler']:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if ddp:
            with torch.no_grad():
                loss_copy = loss.detach().clone()
                dist.all_reduce(loss_copy, op=dist.ReduceOp.SUM)
                loss_copy /= dist.get_world_size()
            if dist.get_rank() == 0:
                li = loss_copy.item() * gradient_accumulation_steps
                all_losses[f'epoch_{epoch}'].append(li)
                loss_val += li
                total += 1
                pbar.set_postfix({'loss': 100 * li, 'avg_loss': 100 * loss_val / (i + 1)})
                sys.stdout.flush()
                wandb.log(build_step_log(
                    li, loss_val / (i + 1), i, active_stem_mask, device
                ))
        else:
            li = loss.item() * gradient_accumulation_steps
            all_losses[f'epoch_{epoch}'].append(li)
            loss_val += li
            total += 1
            pbar.set_postfix({'loss': 100 * li, 'avg_loss': 100 * loss_val / (i + 1)})
            wandb.log(build_step_log(
                li, loss_val / (i + 1), i, active_stem_mask, device
            ))
            loss.detach()

    if should_print:
        print(f'Training loss: {loss_val / total}')
        wandb.log({'train_loss': loss_val / total, 'epoch': epoch, 'learning_rate': optimizer.param_groups[0]['lr']})


def compute_epoch_metrics(model: torch.nn.Module, args: argparse.Namespace, config: ConfigDict,
                          device: torch.device, device_ids: List[int], best_metric: float,
                          epoch: int, scheduler: torch.optim.lr_scheduler, optimizer,
                          all_time_all_metrics, all_losses,  world_size=None, metrics_avg=None, all_metrics=None) -> float:

    """
    Compute and log the metrics for the current epoch, and save model weights if the metric improves.

    Args:
        all_losses:
        all_metrics:
        metrics_avg:
        world_size:
        model: The model to evaluate.
        args: Command-line arguments containing configuration paths and other settings.
        config: Configuration dictionary containing training settings.
        device: The device (CPU or GPU) used for evaluation.
        device_ids: List of GPU device IDs when using multiple GPUs.
        best_metric: The best metric value seen so far.
        epoch: The current epoch number.
        scheduler: The learning rate scheduler to adjust the learning rate.
        optimizer:
        all_time_all_metrics:
    Returns:
        The updated best_metric.
    """

    ddp = True if world_size else False
    should_print = not dist.is_initialized() or dist.get_rank() == 0
    if not ddp:
        if torch.cuda.is_available() and len(device_ids) > 1:
            metrics_avg, all_metrics = valid_multi_gpu(
                model, args, config, args.device_ids, verbose=False, epoch=epoch
            )
        else:
            metrics_avg, all_metrics = valid(model, args, config, device, verbose=False, epoch=epoch)
        all_time_all_metrics[f"epoch_{epoch}"] = all_metrics

    metric_avg = metrics_avg[args.metric_for_scheduler]
    if metric_avg > best_metric:
        best_metric = metric_avg

        if args.each_metrics_in_name:
            stem_parts = []
            for stem_name, values in all_metrics[args.metric_for_scheduler].items():
                stem_values = np.array(values)
                mean_val = stem_values.mean()
                std_val = stem_values.std()
                stem_parts.append(
                    f"{stem_name}_{args.metric_for_scheduler}_{mean_val:.4f}_std_{std_val:.4f}"
                )
            stem_info = "__".join(stem_parts)
            store_path = (
                f"{args.results_path}/model_{args.model_type}_ep_{epoch}_{stem_info}.ckpt"
            )
        else:
            store_path = (
                f"{args.results_path}/model_{args.model_type}_ep_{epoch}_{args.metric_for_scheduler}_{metric_avg:.4f}.ckpt"
            )
        if should_print:
            print(f'Store weights: {store_path}')
            save_weights(
                store_path=store_path,
                model=model,
                device_ids=device_ids,
                optimizer=optimizer,
                epoch=epoch,
                all_time_all_metrics=all_time_all_metrics,
                all_losses=all_losses,
                best_metric=best_metric,
                args=args,
                scheduler=scheduler
            )

    if args.save_weights_every_epoch:
        metric_string = ''
        for m in metrics_avg:
            metric_string += '_{}_{:.4f}'.format(m, metrics_avg[m])
        store_path = f'{args.results_path}/model_{args.model_type}_ep_{epoch}{metric_string}.ckpt'
        save_weights(
            store_path=store_path,
            model=model,
            device_ids=device_ids,
            optimizer=optimizer,
            epoch=epoch,
            all_time_all_metrics=all_time_all_metrics,
            all_losses=all_losses,
            best_metric=best_metric,
            args=args,
            scheduler=scheduler
        )

    if scheduler.name in ['ReduceLROnPlateau']:
        scheduler.step(metric_avg)

    if should_print:
        log_epoch_history(
            args=args,
            epoch=epoch,
            optimizer=optimizer,
            scheduler=scheduler,
            best_metric=best_metric,
            metrics_avg=metrics_avg,
            all_metrics=all_metrics,
            all_losses=all_losses,
        )

    if should_print:
        wandb.log({'metric_main': metric_avg, 'best_metric': best_metric})
        for metric_name in metrics_avg:
            wandb.log({f'metric_{metric_name}': metrics_avg[metric_name]})

    return best_metric


def train_model(args: Union[argparse.Namespace, None], rank=None, world_size=None) -> None:
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

    from msst.utils.dataset import prepare_data
    from msst.utils.model_utils import load_start_checkpoint
    from msst.utils.model_utils import get_lora
    from msst.utils.losses import choice_loss
    from msst.utils.model_utils import get_optimizer, log_model_info

    args = validate_train_args(parse_args_train(args))
    ddp = True if world_size else False
    if ddp:
        ddp_device_id = args.device_ids[rank]
        initialize_environment_ddp(
            rank,
            world_size,
            args.seed,
            args.results_path,
            device_id=ddp_device_id,
        )
    else:
        initialize_environment(args.seed, args.results_path)
    model, config = get_model_from_config(
        args.model_type,
        args.config_path,
        pretrained=not bool(args.start_check_point),
    )
    if 'model_type' in config.training:
        args.model_type = config.training.model_type
    target_instrument = getattr(config.training, 'target_instrument', None)
    args.checkpoint_meta = build_checkpoint_metadata(
        model_type=args.model_type,
        instruments=list(config.training.instruments),
        target_instrument=target_instrument,
    )
    use_amp = getattr(config.training, 'use_amp', True)
    device_ids = args.device_ids
    if ddp:
        batch_size = config.training.batch_size
    else:
        batch_size = config.training.batch_size * len(device_ids)

    if not dist.is_initialized() or dist.get_rank() == 0:
        wandb_init(args, config, batch_size)

    train_loader = prepare_data(config, args, batch_size)

    if args.start_check_point:
        checkpoint = torch.load(args.start_check_point, weights_only=False, map_location='cpu')
        load_start_checkpoint(args, model, checkpoint, type_='train')
    model = get_lora(args, config, model)

    if args.freeze_layers is not None:
        freeze_layers = []
        train_layers = []
        for name, param in model.named_parameters():
            if any(name.startswith(prefix) for prefix in args.freeze_layers):
                freeze_layers.append(name)
                print('Freezing layer:', name)
                param.requires_grad = False
            else:
                train_layers.append(name)
        print('Trainable layers: {}'.format(len(train_layers)))
        print('Frozen layers: {}'.format(len(freeze_layers)))

    if ddp:
        device = torch.device(f'cuda:{ddp_device_id}')
        model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[ddp_device_id],
            find_unused_parameters=True,
        )
        model_module = model.module
    else:
        device, model = initialize_model_and_device(model, args.device_ids)
        # If model is DataParallel, get underlying module
        model_module = model.module if hasattr(model, 'module') else model

    ema_model = None
    if hasattr(config.training, 'ema_momentum') and config.training.ema_momentum > 0:
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Initializing EMA with decay: {config.training.ema_momentum}")
        ema_model = AveragedModel(model_module, multi_avg_fn=get_ema_multi_avg_fn(config.training.ema_momentum))

    if args.pre_valid:
        model_to_valid = ema_model if ema_model is not None else model
        if ddp:
            valid_multi_gpu(model_to_valid, args, config, args.device_ids, verbose=False)
        else:
            if torch.cuda.is_available() and len(args.device_ids) > 1:
                metrics_avg, all_metrics = valid_multi_gpu(model_to_valid, args, config, args.device_ids, verbose=True)
            else:
                metrics_avg, all_metrics = valid(model_to_valid, args, config, device, verbose=True)
            metric_avg = metrics_avg[args.metric_for_scheduler]
            save_weights(
                    store_path=f"{args.results_path}/pre_valid_{args.model_type}_{metric_avg}_{args.metric_for_scheduler}.ckpt",
                    model=model,
                    device_ids=device_ids,
                    optimizer=[],
                    epoch=-1,
                    all_time_all_metrics=[],
                    all_losses=[],
                    best_metric=[],
                    args=args,
                    scheduler=[]
                )

    gradient_accumulation_steps = int(getattr(config.training, 'gradient_accumulation_steps', 1))

    # load optimizer
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)

    if args.start_check_point and "optimizer_state_dict" in checkpoint and args.load_optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if args.start_check_point and "scheduler_state_dict" in checkpoint and args.load_scheduler:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # load num epoch
    if args.start_check_point and "epoch" in checkpoint and args.load_epoch:
        start_epoch = checkpoint["epoch"] + 1
    else:
        start_epoch = 0

    if args.start_check_point and "best_metric" in checkpoint and args.load_best_metric:
        best_metric = checkpoint["best_metric"]
    else:
        best_metric = float('-inf')

    if args.start_check_point and "all_metrics" in checkpoint and args.load_all_metrics:
        all_time_all_metrics = checkpoint["all_metrics"]
    else:
        all_time_all_metrics = {}

    if args.start_check_point and "all_losses" in checkpoint and args.load_all_losses:
        all_losses = checkpoint["all_losses"]
    else:
        all_losses = {}

    multi_loss = choice_loss(args, config)
    amp_enabled = bool(use_amp and device.type == 'cuda')
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)
    except (AttributeError, TypeError):  # PyTorch 2.0-2.2 compatibility
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    if args.set_per_process_memory_fraction:
        torch.cuda.set_per_process_memory_fraction(1.0)
    torch.cuda.empty_cache()

    safe_mode = args.safe_mode

    should_print = not dist.is_initialized() or dist.get_rank() == 0

    if should_print:
        if world_size:
            batch_size = config.training.batch_size
            ef_batch_size = batch_size * gradient_accumulation_steps * world_size
            num_gpu = world_size
        else:
            device_ids = args.device_ids
            batch_size = config.training.batch_size * len(device_ids)
            ef_batch_size = batch_size * gradient_accumulation_steps
            num_gpu = len(device_ids)

        print(
            f"Instruments: {config.training.instruments}\n"
            f"Metrics for training: {args.metrics}. Metric for scheduler: {args.metric_for_scheduler}\n"
            f"Patience: {config.training.patience} "
            f"Reduce factor: {config.training.reduce_factor}\n"
            f"Batch size: {batch_size} "
            f"Grad accum steps: {gradient_accumulation_steps} "
            f"Num gpus: {num_gpu} "
            f"Effective batch size: {ef_batch_size}\n"
            f"Dataset type: {args.dataset_type}\n"
            f"Optimizer: {config.training.optimizer}"
        )

        print(f'Train for: {config.training.num_epochs} epochs')
        log_model_info(model, args.results_path)

    for epoch in range(start_epoch, config.training.num_epochs):
        if ddp:
            train_loader.sampler.set_epoch(epoch)

        train_one_epoch(model, config, args, optimizer, device, device_ids, epoch,
                        use_amp, scaler, scheduler, gradient_accumulation_steps, train_loader, multi_loss, all_losses,
                        world_size, ema_model=ema_model, safe_mode=safe_mode)

        model_to_valid = ema_model if ema_model is not None else model

        if should_print:
            save_last_weights(args, model, device_ids, optimizer, epoch, all_time_all_metrics, all_losses, best_metric, scheduler)
        if ddp:
            metrics_avg, all_metrics = valid_multi_gpu(
                model, args, config, args.device_ids, verbose=False, epoch=epoch
            )
            if rank == 0:
                all_time_all_metrics[f"epoch_{epoch}"] = all_metrics
                best_metric = compute_epoch_metrics(
                    model=model,
                    args=args,
                    config=config,
                    device=device,
                    device_ids=device_ids,
                    best_metric=best_metric,
                    epoch=epoch,
                    scheduler=scheduler,
                    optimizer=optimizer,
                    all_time_all_metrics=all_time_all_metrics,
                    all_losses=all_losses,
                    world_size=world_size,
                    metrics_avg=metrics_avg,
                    all_metrics=all_metrics
                )
        else:
            best_metric = compute_epoch_metrics(
                model=model,
                args=args,
                config=config,
                device=device,
                device_ids=device_ids,
                best_metric=best_metric,
                epoch=epoch,
                scheduler=scheduler,
                optimizer=optimizer,
                all_time_all_metrics=all_time_all_metrics,
                all_losses=all_losses,
            )


if __name__ == "__main__":
    train_model(None)
