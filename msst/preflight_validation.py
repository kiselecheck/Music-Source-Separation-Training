"""Validate metadata using the same track discovery as the validation runtime."""
from pathlib import Path

import soundfile as sf


def inspect_validation_metadata(args, config):
    from ._validation import _audio_file_extensions, _mixture_source_instruments, get_mixture_paths
    from .utils.model_utils import prefer_target_instrument

    extension = getattr(config.inference, 'extension', 'wav')
    extensions = _audio_file_extensions(extension)
    tracks = get_mixture_paths(args, False, config, extension)
    if not tracks:
        raise ValueError('No validation mixtures found for the configured paths or manifest')
    target = getattr(config.training, 'target_instrument', None)
    allow_negative = bool(target and getattr(config.training, 'single_target_from_mixture', False))
    instruments = prefer_target_instrument(config)
    negative_count = 0
    for track in tracks:
        path = Path(track)
        folder = path if path.is_dir() else path.parent
        if path.is_dir():
            rates = []
            for name in _mixture_source_instruments(config):
                source = next((folder / f'{name}.{ext}' for ext in extensions
                               if (folder / f'{name}.{ext}').is_file()), None)
                if source is None:
                    raise FileNotFoundError(f"Missing mixture component '{name}' in {folder}")
                rates.append(sf.info(source).samplerate)
            if not rates or len(set(rates)) != 1:
                raise ValueError(f'Invalid composed mixture sample rates in {folder}: {rates}')
        else:
            sf.info(path)
        for instrument in instruments:
            reference = next((folder / f'{instrument}.{ext}' for ext in extensions
                              if (folder / f'{instrument}.{ext}').is_file()), None)
            if reference is None:
                if instrument == target and allow_negative:
                    negative_count += 1
                    continue
                raise FileNotFoundError(f"Reference '{instrument}' is missing in {folder}")
            if instrument == 'other' and getattr(config.training, 'other_fix', False):
                reference = folder / f'vocals{reference.suffix}'
                if not reference.is_file():
                    raise FileNotFoundError(f"Reference '{reference.name}' is missing in {folder}")
            sf.info(reference)
    return {'tracks': len(tracks), 'target_absent': negative_count}
