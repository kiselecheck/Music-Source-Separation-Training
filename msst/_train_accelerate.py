# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'
# Read more here:
# https://huggingface.co/docs/accelerate/index

import soundfile as sf
import numpy as np
import time
import glob
from tqdm.auto import tqdm
import os
import torch
import wandb
from torch.optim import Adam, AdamW, SGD, RAdam, RMSprop
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from accelerate import Accelerator

from msst.utils.dataset import MSSDataset, _training_collate_fn
from msst.utils.model_utils import demix, prefer_target_instrument, load_start_checkpoint
from msst.utils.metrics import sdr
from msst.utils.settings import (
    manual_seed,
    get_model_from_config,
    parse_args_train,
    validate_train_args,
    wandb_init,
)
from msst.utils.losses import choice_loss
from msst._training import compute_training_loss
from msst.checkpoints import build_checkpoint_metadata


def valid(model, valid_loader, args, config, device, verbose=False):
    instruments = prefer_target_instrument(config)

    all_sdr = dict()
    for instr in instruments:
        all_sdr[instr] = []

    all_mixtures_path = valid_loader
    if verbose:
        all_mixtures_path = tqdm(valid_loader)

    pbar_dict = {}
    for path_list in all_mixtures_path:
        path = path_list[0]
        mix, sr = sf.read(path)
        folder = os.path.dirname(path)
        res = demix(config, model, mix.T, device, model_type=args.model_type) # mix.T
        for instr in instruments:
            if instr != 'other' or config.training.other_fix is False:
                track, sr1 = sf.read(folder + '/{}.wav'.format(instr))
            else:
                # other is actually instrumental
                track, sr1 = sf.read(folder + '/{}.wav'.format('vocals'))
                track = mix - track
            # sf.write("{}.wav".format(instr), res[instr].T, sr, subtype='FLOAT')
            references = np.expand_dims(track, axis=0)
            estimates = np.expand_dims(res[instr].T, axis=0)
            sdr_val = sdr(references, estimates)[0]
            single_val = torch.from_numpy(np.array([sdr_val])).to(device)
            all_sdr[instr].append(single_val)
            pbar_dict['sdr_{}'.format(instr)] = sdr_val
        if verbose:
            all_mixtures_path.set_postfix(pbar_dict)

    return all_sdr


class MSSValidationDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        all_mixtures_path = []
        for valid_path in args.valid_path:
            part = sorted(glob.glob(valid_path + '/*/mixture.wav'))
            if len(part) == 0:
                print('No validation data found in: {}'.format(valid_path))
            all_mixtures_path += part

        self.list_of_files = all_mixtures_path

    def __len__(self):
        return len(self.list_of_files)

    def __getitem__(self, index):
        return self.list_of_files[index]


def train_model(args):
    accelerator = Accelerator()
    device = accelerator.device
    args = validate_train_args(parse_args_train(args))

    manual_seed(args.seed + int(time.time()))
    # torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False # Fix possible slow down with dilation convolutions
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError as error:
        if "context has already been set" not in str(error):
            raise

    model, config = get_model_from_config(
        args.model_type,
        args.config_path,
        pretrained=not bool(args.start_check_point),
    )
    if 'model_type' in config.training:
        args.model_type = config.training.model_type
    target_instrument = getattr(config.training, 'target_instrument', None)
    checkpoint_meta = build_checkpoint_metadata(
        model_type=args.model_type,
        instruments=list(config.training.instruments),
        target_instrument=target_instrument,
    )
    accelerator.print("Instruments: {}".format(config.training.instruments))

    os.makedirs(args.results_path, exist_ok=True)

    device_ids = args.device_ids
    batch_size = config.training.batch_size

    # wandb
    if accelerator.is_main_process:
        wandb_init(args, config, batch_size)
    else:
        wandb.init(mode = 'disabled')

    # Fix for num of steps
    config.training.num_steps *= accelerator.num_processes

    metadata_path = os.path.join(
        args.results_path, 'metadata_{}.pkl'.format(args.dataset_type)
    )
    if accelerator.is_main_process:
        MSSDataset.collect_metadata(
            config,
            args.data_path,
            metadata_path=metadata_path,
            dataset_type=args.dataset_type,
            verbose=accelerator.is_main_process,
            reuse=True,
        )
    accelerator.wait_for_everyone()
    metadata = MSSDataset.load_metadata(metadata_path)
    trainset = MSSDataset(
        config,
        args.data_path,
        batch_size=batch_size,
        metadata_path=metadata_path,
        dataset_type=args.dataset_type,
        verbose=accelerator.is_main_process,
        metadata=metadata,
    )

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        collate_fn=_training_collate_fn(config, args.dataset_type),
    )

    validset = MSSValidationDataset(args)
    valid_dataset_length = len(validset)

    valid_loader = DataLoader(
        validset,
        batch_size=1,
        shuffle=False,
    )

    valid_loader = accelerator.prepare(valid_loader)

    if args.start_check_point != '':
        accelerator.print('Start from checkpoint: {}'.format(args.start_check_point))
        if 1:
            checkpoint = torch.load(
                args.start_check_point,
                weights_only=False,
                map_location='cpu',
            )
            load_start_checkpoint(args, model, checkpoint, type_='train')
        else:
            model.load_state_dict(
                torch.load(args.start_check_point)
            )

    optim_params = dict()
    if 'optimizer' in config:
        optim_params = dict(config['optimizer'])
        accelerator.print('Optimizer params from config:\n{}'.format(optim_params))

    if config.training.optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=config.training.lr, **optim_params)
    elif config.training.optimizer == 'adamw':
        optimizer = AdamW(model.parameters(), lr=config.training.lr, **optim_params)
    elif config.training.optimizer == 'radam':
        optimizer = RAdam(model.parameters(), lr=config.training.lr, **optim_params)
    elif config.training.optimizer == 'rmsprop':
        optimizer = RMSprop(model.parameters(), lr=config.training.lr, **optim_params)
    elif config.training.optimizer == 'prodigy':
        from prodigyopt import Prodigy
        # you can choose weight decay value based on your problem, 0 by default
        # We recommend using lr=1.0 (default) for all networks.
        optimizer = Prodigy(model.parameters(), lr=config.training.lr, **optim_params)
    elif config.training.optimizer == 'adamw8bit':
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=config.training.lr, **optim_params)
    elif config.training.optimizer == 'sgd':
        accelerator.print('Use SGD optimizer')
        optimizer = SGD(model.parameters(), lr=config.training.lr, **optim_params)
    else:
        accelerator.print('Unknown optimizer: {}'.format(config.training.optimizer))
        exit()

    if accelerator.is_main_process:
        print('Processes GPU: {}'.format(accelerator.num_processes))
        print("Patience: {} Reduce factor: {} Batch size: {} Optimizer: {}".format(
            config.training.patience,
            config.training.reduce_factor,
            batch_size,
            config.training.optimizer,
        ))
    # Reduce LR if no SDR improvements for several epochs
    scheduler = ReduceLROnPlateau(
        optimizer,
        'max',
        # patience=accelerator.num_processes * config.training.patience, # This is strange place...
        patience=config.training.patience,
        factor=config.training.reduce_factor
    )

    multi_loss = choice_loss(args, config)
    use_amp = bool(getattr(config.training, 'use_amp', True))

    model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)

    ema_model = None
    if hasattr(config.training, 'ema_momentum') and config.training.ema_momentum > 0:
        accelerator.print(f"Initializing EMA with decay: {config.training.ema_momentum}")
        ema_model = AveragedModel(accelerator.unwrap_model(model), multi_avg_fn=get_ema_multi_avg_fn(config.training.ema_momentum))
        ema_model.to(device)

    if args.pre_valid:
        model_to_valid = ema_model if ema_model is not None else model
        sdr_list = valid(model_to_valid, valid_loader, args, config, device, verbose=accelerator.is_main_process)
        sdr_list = accelerator.gather(sdr_list)
        accelerator.wait_for_everyone()

        # print(sdr_list)

        sdr_avg = 0.0
        instruments = prefer_target_instrument(config)

        for instr in instruments:
            # print(sdr_list[instr])
            sdr_data = torch.cat(sdr_list[instr], dim=0).cpu().numpy()
            sdr_val = sdr_data.mean()
            accelerator.print("Valid length: {}".format(valid_dataset_length))
            accelerator.print("Instr SDR {}: {:.4f} Debug: {}".format(instr, sdr_val, len(sdr_data)))
            sdr_val = sdr_data[:valid_dataset_length].mean()
            accelerator.print("Instr SDR {}: {:.4f} Debug: {}".format(instr, sdr_val, len(sdr_data)))
            sdr_avg += sdr_val
        sdr_avg /= len(instruments)
        if len(instruments) > 1:
            accelerator.print('SDR Avg: {:.4f}'.format(sdr_avg))
        sdr_list = None

    accelerator.print('Train for: {}'.format(config.training.num_epochs))
    best_sdr = -100
    for epoch in range(config.training.num_epochs):
        model.train().to(device)
        accelerator.print('Train epoch: {} Learning rate: {}'.format(epoch, optimizer.param_groups[0]['lr']))
        loss_val = 0.
        total = 0

        pbar = tqdm(train_loader, disable=not accelerator.is_main_process)
        for i, data in enumerate(pbar):
            loss, _ = compute_training_loss(
                model,
                config,
                args,
                data,
                device,
                args.device_ids,
                multi_loss,
                use_amp,
            )

            accelerator.backward(loss)
            if config.training.grad_clip:
                accelerator.clip_grad_norm_(model.parameters(), config.training.grad_clip)

            optimizer.step()
            optimizer.zero_grad()

            if ema_model is not None:
                ema_model.update_parameters(accelerator.unwrap_model(model))

            li = loss.item()
            loss_val += li
            total += 1
            if accelerator.is_main_process:
                wandb.log({'loss': 100 * li, 'avg_loss': 100 * loss_val / (i + 1), 'total': total, 'loss_val': loss_val, 'i': i })
                pbar.set_postfix({'loss': 100 * li, 'avg_loss': 100 * loss_val / (i + 1)})

        if accelerator.is_main_process:
            print('Training loss: {:.6f}'.format(loss_val / total))
            wandb.log({'train_loss': loss_val / total, 'epoch': epoch})

        # Save last
        store_path = args.results_path + '/last_{}.ckpt'.format(args.model_type)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            if ema_model is not None:
                accelerator.save(
                    {"model_state_dict": ema_model.module.state_dict(), "meta": checkpoint_meta},
                    store_path,
                )
            else:
                unwrapped_model = accelerator.unwrap_model(model)
                accelerator.save(
                    {"model_state_dict": unwrapped_model.state_dict(), "meta": checkpoint_meta},
                    store_path,
                )

        # Validation
        model_to_valid = ema_model if ema_model is not None else model
        sdr_list = valid(model_to_valid, valid_loader, args, config, device, verbose=accelerator.is_main_process)
        sdr_list = accelerator.gather(sdr_list)
        accelerator.wait_for_everyone()

        sdr_avg = 0.0
        instruments = prefer_target_instrument(config)

        for instr in instruments:
            if accelerator.is_main_process and 0:
                print(sdr_list[instr])
            sdr_data = torch.cat(sdr_list[instr], dim=0).cpu().numpy()
            # sdr_val = sdr_data.mean()
            sdr_val = sdr_data[:valid_dataset_length].mean()
            if accelerator.is_main_process:
                print("Instr SDR {}: {:.4f} Debug: {}".format(instr, sdr_val, len(sdr_data)))
                wandb.log({ f'{instr}_sdr': sdr_val })
            sdr_avg += sdr_val
        sdr_avg /= len(instruments)
        if len(instruments) > 1:
            if accelerator.is_main_process:
                print('SDR Avg: {:.4f}'.format(sdr_avg))
                wandb.log({'sdr_avg': sdr_avg, 'best_sdr': best_sdr})

        if accelerator.is_main_process:
            if sdr_avg > best_sdr:
                store_path = args.results_path + '/model_{}_ep_{}_sdr_{:.4f}.ckpt'.format(args.model_type, epoch, sdr_avg)
                print('Store weights: {}'.format(store_path))
                if ema_model is not None:
                    accelerator.save(
                        {"model_state_dict": ema_model.module.state_dict(), "meta": checkpoint_meta},
                        store_path,
                    )
                else:
                    unwrapped_model = accelerator.unwrap_model(model)
                    accelerator.save(
                        {"model_state_dict": unwrapped_model.state_dict(), "meta": checkpoint_meta},
                        store_path,
                    )
                best_sdr = sdr_avg

            scheduler.step(sdr_avg)

        sdr_list = None
        accelerator.wait_for_everyone()


if __name__ == "__main__":
    train_model(None)
