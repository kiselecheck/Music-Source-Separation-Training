from argparse import Namespace

import numpy as np
import pytest
import soundfile as sf
from ml_collections import ConfigDict

from msst.preflight_validation import inspect_validation_metadata


def audio(path, rate=8000):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros((128, 2), dtype='float32'), rate, subtype='FLOAT')


def config():
    return ConfigDict({'training': {'instruments': ['synth_bass', 'other'],
        'target_instrument': 'synth_bass', 'single_target_from_mixture': True,
        'mixture_source_instruments': ['bass', 'drums'], 'other_fix': False},
        'inference': {'batch_size': 1, 'num_overlap': 2}})


def test_composed_negative_and_target_tracks(tmp_path):
    for name in ['bass', 'drums']:
        audio(tmp_path / 'negative' / f'{name}.wav')
        audio(tmp_path / 'positive' / f'{name}.wav')
    audio(tmp_path / 'positive' / 'synth_bass.wav')
    result = inspect_validation_metadata(Namespace(valid_path=[str(tmp_path)]), config())
    assert result == {'tracks': 2, 'target_absent': 1}


def test_explicit_target_and_manifest_exclude_other_incomplete_tracks(tmp_path):
    audio(tmp_path / 'good' / 'mixture.wav')
    audio(tmp_path / 'good' / 'synth_bass.wav')
    audio(tmp_path / 'bad' / 'mixture.wav')
    manifest = tmp_path / 'manifest.csv'
    manifest.write_text('key,mixture_path\ngood,good/mixture.wav\n', encoding='utf-8')
    args = Namespace(valid_path=[str(tmp_path)], validation_manifest=str(manifest))
    assert inspect_validation_metadata(args, config()) == {'tracks': 1, 'target_absent': 0}


def test_missing_required_target_is_error_without_negative_mode(tmp_path):
    audio(tmp_path / 'track' / 'mixture.wav')
    cfg = config()
    cfg.training.single_target_from_mixture = False
    with pytest.raises(FileNotFoundError, match='synth_bass'):
        inspect_validation_metadata(Namespace(valid_path=[str(tmp_path)]), cfg)


def test_composed_mixture_with_mismatched_sample_rates_fails(tmp_path):
    audio(tmp_path / 'track' / 'bass.wav', rate=8000)
    audio(tmp_path / 'track' / 'drums.wav', rate=16000)
    with pytest.raises(ValueError, match='sample rates'):
        inspect_validation_metadata(Namespace(valid_path=[str(tmp_path)]), config())
