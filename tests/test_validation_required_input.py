from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from ml_collections import ConfigDict

from msst._validation import get_mixture_paths, process_audio_files, _validation_manifest_for_args


def write(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.ones((64, 2), dtype=np.float32) * 0.1, 8000, subtype='FLOAT')


def config(**inference):
    return ConfigDict({'training': {'instruments': ['bass'], 'target_instrument': 'bass',
                        'mixture_source_instruments': ['bass', 'drums']},
                       'inference': {'batch_size': 1, 'num_overlap': 1, **inference}})


def test_valid_input_with_components_is_not_reconstructed(tmp_path):
    track = tmp_path / 'song'
    for name in ('mixture', 'bass', 'drums'):
        write(track / f'{name}.wav')
    paths = get_mixture_paths(Namespace(valid_path=[str(tmp_path)]), False, config(), 'wav')
    assert paths == [str(track / 'mixture.wav')]


def test_missing_input_errors_even_with_other_valid_tracks(tmp_path):
    write(tmp_path / 'good' / 'mixture.wav')
    for name in ('bass', 'drums'):
        write(tmp_path / 'missing' / f'{name}.wav')
    with pytest.raises(FileNotFoundError, match='missing') as exc:
        get_mixture_paths(Namespace(valid_path=[str(tmp_path)]), False, config(), 'wav')
    assert 'mixture' in str(exc.value)


def test_empty_track_errors_with_path(tmp_path):
    (tmp_path / 'empty_track').mkdir()
    with pytest.raises(FileNotFoundError, match='empty_track'):
        get_mixture_paths(Namespace(valid_path=[str(tmp_path)]), False, config(), 'wav')


def test_explicit_dereverb_name_is_accepted(tmp_path):
    track = tmp_path / 'song'
    write(track / 'reverb.wav')
    write(track / 'vocals.wav')
    paths = get_mixture_paths(Namespace(valid_path=[str(tmp_path)]), False,
                              config(input_name='reverb'), 'wav')
    assert paths == [str(track / 'reverb.wav')]
    assert get_mixture_paths(Namespace(valid_path=[str(tmp_path)]), False,
                             config(input_name='reverb.wav'), 'wav') == paths
    with pytest.raises(FileNotFoundError, match='song'):
        get_mixture_paths(Namespace(valid_path=[str(tmp_path)]), False, config(), 'wav')


def test_manifest_directory_resolves_input_and_keeps_key(tmp_path):
    track = tmp_path / 'valid' / 'song'
    write(track / 'reverb.wav')
    manifest = tmp_path / 'manifest.csv'
    manifest.write_text('key,mixture_path,group\ntrack,valid/song,holdout\n', encoding='utf-8')
    args = Namespace(valid_path=[str(tmp_path / 'valid')], validation_manifest=str(manifest))
    metadata = _validation_manifest_for_args(args)
    paths = get_mixture_paths(args, False, config(input_name='reverb'), 'wav', metadata)
    assert paths == [str(track / 'reverb.wav')]
    assert metadata[paths[0]].key == 'track' and metadata[paths[0]].group == 'holdout'


def test_manifest_directory_requires_input(tmp_path):
    track = tmp_path / 'valid' / 'song'
    write(track / 'bass.wav')
    manifest = tmp_path / 'manifest.csv'
    manifest.write_text('key,mixture_path\ntrack,valid/song\n', encoding='utf-8')
    args = Namespace(valid_path=[str(tmp_path / 'valid')], validation_manifest=str(manifest))
    with pytest.raises(FileNotFoundError, match='song'):
        get_mixture_paths(args, False, config(), 'wav')


def test_direct_processing_does_not_build_mixture(tmp_path):
    write(tmp_path / 'song' / 'bass.wav')
    args = Namespace(metrics=['sdr'], model_type='stub')
    with pytest.raises(ValueError, match='song'):
        process_audio_files([str(tmp_path / 'song')], torch.nn.Identity(), args,
                            config(), torch.device('cpu'), is_tqdm=False)


def test_accelerate_uses_same_input_discovery(tmp_path):
    pytest.importorskip('accelerate')
    from msst._train_accelerate import MSSValidationDataset

    write(tmp_path / 'song' / 'reverb.wav')
    args = Namespace(valid_path=[str(tmp_path)])
    assert MSSValidationDataset(args, config(input_name='reverb'))[0] == str(tmp_path / 'song' / 'reverb.wav')
    with pytest.raises(FileNotFoundError, match='song'):
        MSSValidationDataset(args, config())


def test_real_validation_uses_configured_input_and_reports_missing_track(tmp_path):
    from msst._validation import valid

    class Identity(torch.nn.Module):
        def forward(self, x):
            return x.unsqueeze(1)
    track = tmp_path / 'song'
    write(track / 'reverb.wav')
    write(track / 'bass.wav')
    cfg = config(input_name='reverb')
    cfg.audio = {'chunk_size': 32, 'sample_rate': 8000}
    cfg.inference.num_overlap = 2
    cfg.training.use_amp = False
    cfg.training.other_fix = False
    args = Namespace(valid_path=[str(tmp_path)], model_type='stub', metrics=['sdr'])
    average, details = valid(Identity(), args, cfg, torch.device('cpu'))
    assert average['sdr'] > 60 and len(details['sdr']['bass']) == 1
    write(tmp_path / 'broken' / 'bass.wav')
    with pytest.raises(FileNotFoundError, match='broken'):
        valid(Identity(), args, cfg, torch.device('cpu'))
