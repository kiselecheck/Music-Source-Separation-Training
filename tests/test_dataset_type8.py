import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
from ml_collections import ConfigDict

from msst.utils.dataset import MSSDataset


def make_config(source_instruments=None):
    config = ConfigDict()
    config.audio = ConfigDict({
        'chunk_size': 32,
        'min_mean_abs': 0.0,
        'sample_rate': 44100,
    })
    config.training = ConfigDict({
        'batch_size': 1,
        'instruments': ['orig'],
        'num_steps': 1,
        'precompute_chunks': False,
        'target_instrument': 'orig',
        'read_metadata_procs': 1,
    })
    config.augmentations = ConfigDict({'enable': False})
    config.reverb = ConfigDict({
        'apply_probability': 0.0,
        'exclude_instruments': ['mixture', 'reverb', 'orig'],
        'method_weights': {'algorithmic': 1.0},
        'source_extensions': ['wav', 'flac'],
        'stem_loudness': False,
    })
    if source_instruments is not None:
        config.reverb.source_instruments = source_instruments
    return config


class DatasetType8SourcesTest(unittest.TestCase):
    def create_dataset(self, root, source_instruments=None):
        track_dir = root / 'track'
        track_dir.mkdir()
        sf.write(track_dir / 'alpha.wav', np.full((32, 2), 0.1, np.float32), 44100)
        sf.write(track_dir / 'beta.flac', np.full((32, 2), 0.2, np.float32), 44100)
        sf.write(track_dir / 'mixture.wav', np.full((32, 2), 0.9, np.float32), 44100)
        return MSSDataset(
            make_config(source_instruments),
            str(root),
            metadata_path=str(root / 'metadata.pkl'),
            dataset_type=8,
            verbose=False,
        )

    def test_auto_mode_loads_all_stems_and_excludes_service_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset = self.create_dataset(Path(tmp_dir))
            target, mixture, info = dataset.load_reverb_data(return_info=True)

        self.assertEqual(info['source_instruments'], ['alpha', 'beta'])
        np.testing.assert_allclose(target.numpy()[0], 0.3, atol=1e-4)
        np.testing.assert_allclose(mixture.numpy(), 0.3, atol=1e-4)

    def test_explicit_source_array_keeps_previous_behavior(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset = self.create_dataset(Path(tmp_dir), ['beta'])
            target, mixture, info = dataset.load_reverb_data(return_info=True)

        self.assertEqual(info['source_instruments'], ['beta'])
        np.testing.assert_allclose(target.numpy()[0], 0.2, atol=1e-4)
        np.testing.assert_allclose(mixture.numpy(), 0.2, atol=1e-4)

    def test_preferred_reverb_stem_is_added_to_random_selection(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset = self.create_dataset(Path(tmp_dir))
            dataset.reverb_config.apply_probability = 1.0
            dataset.reverb_config.none_probability = 0.0
            dataset.reverb_config.all_probability = 0.0
            dataset.reverb_config.min_reverb_stems = 1
            dataset.reverb_config.max_reverb_stems = 1
            dataset.reverb_config.preferred_reverb_stems = ['vocal']
            dataset.reverb_config.preferred_reverb_probability = 1.0

            with (
                patch('msst.utils.dataset.random.random', side_effect=[0.0, 0.5, 0.0]),
                patch('msst.utils.dataset.random.randint', return_value=1),
                patch('msst.utils.dataset.random.sample', return_value=[0]),
            ):
                selected = dataset._choose_reverb_stem_indices(
                    3,
                    ['other', 'lead_vocal', 'drums'],
                )

        self.assertEqual(selected, [0, 1])


if __name__ == '__main__':
    unittest.main()
