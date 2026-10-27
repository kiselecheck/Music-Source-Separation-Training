import random
from argparse import Namespace

import numpy as np
import pytest
import soundfile as sf
import torch
from ml_collections import ConfigDict

from msst.data_audit import inspect_training_pairs


def _case(tmp_path, target, mixture):
    track = tmp_path / 'track'
    track.mkdir()
    for name, value in [('vocals', target), ('mixture', mixture)]:
        sf.write(track / f'{name}.wav', np.full((1024, 2), value, dtype=np.float32),
                 8000, subtype='FLOAT')
    config = ConfigDict({'audio': {'chunk_size': 256, 'min_mean_abs': 0, 'sample_rate': 8000},
        'training': {'batch_size': 1, 'num_steps': 1, 'instruments': ['vocals', 'other'],
                     'target_instrument': 'vocals', 'single_target_from_mixture': True,
                     'read_metadata_procs': 1}, 'augmentations': {'enable': False}})
    args = Namespace(data_path=[str(tmp_path)], dataset_type=6)
    return config, args, [(str(track), 1024)], str(tmp_path / 'metadata.pkl')


@pytest.mark.parametrize('target,mixture,identity,silent', [(0.2, 0.2, 4, 0),
                                                            (0.2, 0.4, 0, 0),
                                                            (0.0, 0.4, 0, 4)])
def test_audit_real_dataset_pairs_and_preserves_rng(tmp_path, target, mixture, identity, silent):
    config, args, metadata, cache = _case(tmp_path, target, mixture)
    before_python = random.getstate()
    before_numpy = np.random.get_state()
    before_torch = torch.get_rng_state().clone()
    report = inspect_training_pairs(config, args, metadata, cache, 4)
    assert report['samples'] == 4
    assert report['identity_like'] == identity
    assert report['silent_targets'] == silent
    assert report['nonfinite'] == 0
    assert random.getstate() == before_python
    assert np.array_equal(np.random.get_state()[1], before_numpy[1])
    assert torch.equal(torch.get_rng_state(), before_torch)


def test_pair_audit_requires_single_target(tmp_path):
    config, args, metadata, cache = _case(tmp_path, 0.2, 0.4)
    config.training.target_instrument = None
    with pytest.raises(ValueError, match='single-target'):
        inspect_training_pairs(config, args, metadata, cache, 1)

def test_check_exposes_identity_warning_and_preserves_output_directory(tmp_path, monkeypatch, capsys):
    import msst
    import yaml
    from examples.custom_model import TinySeparator

    config, args, _, _ = _case(tmp_path, 0.2, 0.2)
    config.audio.num_channels = 2
    config.training.other_fix = False
    config.inference = {'batch_size': 1, 'num_overlap': 2}
    sf.write(tmp_path / 'track' / 'other.wav', np.zeros((1024, 2), dtype=np.float32),
             8000, subtype='FLOAT')
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config.to_dict()), encoding='utf-8')
    results = tmp_path / 'not_created'
    msst.register_model('audit_tiny', TinySeparator)
    try:
        report = msst.check(config_path=config_path, data_path=tmp_path,
                            valid_path=tmp_path, results_path=results,
                            model_type='audit_tiny', dataset_type=6,
                            sample_training_pairs=3)
        from msst.cli import check
        monkeypatch.setattr('sys.argv', ['msst check', '--config_path', str(config_path),
                            '--data_path', str(tmp_path), '--valid_path', str(tmp_path),
                            '--results_path', str(results), '--model_type', 'audit_tiny',
                            '--dataset_type', '6', '--sample_training_pairs', '2'])
        check()
        assert '[WARNING] training_pairs: Sampled 2 pairs: 2' in capsys.readouterr().out
    finally:
        msst.unregister_model('audit_tiny')
    item = next(item for item in report.items if item.name == 'training_pairs')
    assert item.status == 'warning'
    assert item.details['identity_like'] == 3
    assert report.ok
    assert not results.exists()
