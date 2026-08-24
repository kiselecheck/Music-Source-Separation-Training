import unittest

import torch

from msst.utils.dataset import class_balanced_collate_fn
from msst.utils.losses import masked_loss, spec_masked_loss, spec_rmse_loss
from msst.models.scnet.scnet import SCNet, SUlayer


class MaskedLossRegressionTest(unittest.TestCase):
    def test_coarse_masked_loss_is_nonzero_for_batch_size_one(self):
        estimate = torch.zeros((1, 3, 2, 8), requires_grad=True)
        target = torch.ones_like(estimate)

        loss = masked_loss(estimate, target, q=0.95, coarse=True)
        loss.backward()

        self.assertGreater(loss.item(), 0.0)
        self.assertGreater(estimate.grad.abs().sum().item(), 0.0)


class DatasetType7CollateTest(unittest.TestCase):
    def test_variable_active_stems_are_collated_as_per_sample_mask(self):
        first = (torch.zeros(3, 2, 8), torch.zeros(2, 8), [0, 2])
        second = (torch.ones(3, 2, 8), torch.ones(2, 8), [1])

        stems, mixes, active_stem_mask = class_balanced_collate_fn([first, second])

        self.assertEqual(tuple(stems.shape), (2, 3, 2, 8))
        self.assertEqual(tuple(mixes.shape), (2, 2, 8))
        torch.testing.assert_close(
            active_stem_mask,
            torch.tensor([
                [True, False, True],
                [False, True, False],
            ]),
        )


class ActiveStemSpectralLossTest(unittest.TestCase):
    stft_config = {
        'n_fft': 8,
        'hop_length': 4,
        'win_length': 8,
        'center': True,
        'normalized': True,
    }

    def _assert_ignores_inactive_stems(self, loss_fn):
        estimate = torch.randn((2, 3, 2, 32), requires_grad=True)
        sources = torch.zeros_like(estimate)
        active_stem_mask = torch.tensor([
            [True, False, False],
            [False, True, False],
        ])

        with torch.no_grad():
            estimate[~active_stem_mask] = 1000.0

        masked = loss_fn(
            estimate,
            sources,
            self.stft_config,
            active_stem_mask=active_stem_mask,
        )
        selected = loss_fn(
            estimate[active_stem_mask],
            sources[active_stem_mask],
            self.stft_config,
        )

        torch.testing.assert_close(masked, selected)
        masked.backward()
        self.assertEqual(estimate.grad[~active_stem_mask].abs().sum().item(), 0.0)
        self.assertGreater(estimate.grad[active_stem_mask].abs().sum().item(), 0.0)

    def test_spec_rmse_ignores_inactive_sample_stem_pairs(self):
        self._assert_ignores_inactive_stems(spec_rmse_loss)

    def test_spec_masked_ignores_inactive_sample_stem_pairs(self):
        self._assert_ignores_inactive_stems(spec_masked_loss)


class SCNetActiveOutputTest(unittest.TestCase):
    band_configs = {
        'low': {'SR': 0.25, 'stride': 1, 'kernel': 1},
        'mid': {'SR': 0.25, 'stride': 1, 'kernel': 1},
        'high': {'SR': 0.50, 'stride': 1, 'kernel': 1},
    }

    def test_final_sparse_upsampler_matches_selected_full_channels(self):
        layer = SUlayer(3, 12, self.band_configs)
        x = torch.randn(2, 3, 6, 5)
        lengths = [2, 2, 2]
        original_lengths = [2, 2, 2]

        full = layer(x, lengths, original_lengths)
        selected_indices = [4, 5, 6, 7]
        selected = layer(
            x,
            lengths,
            original_lengths,
            output_channel_indices=selected_indices,
        )

        torch.testing.assert_close(selected, full[:, selected_indices])
        selected.square().mean().backward()
        inactive_indices = [index for index in range(12) if index not in selected_indices]
        for convtr in layer.convtrs:
            self.assertGreater(
                convtr.weight.grad[:, selected_indices].abs().sum().item(), 0.0
            )
            self.assertEqual(
                convtr.weight.grad[:, inactive_indices].abs().sum().item(), 0.0
            )

    def test_scnet_selected_stems_match_full_forward(self):
        model = SCNet(
            sources=['a', 'b', 'c'],
            dims=[4, 4],
            nfft=16,
            hop_size=4,
            win_size=16,
            band_SR=[0.25, 0.25, 0.5],
            band_stride=[1, 1, 1],
            band_kernel=[1, 1, 1],
            conv_depths=[1, 1, 1],
            num_dplayer=2,
        ).eval()
        mixture = torch.randn(1, 2, 64)

        with torch.no_grad():
            full = model(mixture)
            selected = model(mixture, active_stem_ids=[0, 2])

        torch.testing.assert_close(selected, full[:, [0, 2]], rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
