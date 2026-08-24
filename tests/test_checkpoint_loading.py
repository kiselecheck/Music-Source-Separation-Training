import unittest
import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from msst._inference import validate_checkpoint_metadata
from msst.checkpoints import CHECKPOINT_FORMAT_VERSION, build_checkpoint_metadata
from msst.errors import CheckpointError
from msst.utils.model_utils import load_not_compatible_weights, save_weights


class TinyMultiHeadModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        self.mask_estimators = nn.ModuleList([nn.Linear(2, 2) for _ in range(2)])


class CheckpointLoadingTest(unittest.TestCase):
    def test_excluded_prefix_keeps_new_heads(self):
        source = TinyMultiHeadModel()
        target = TinyMultiHeadModel()

        with torch.no_grad():
            for parameter in source.parameters():
                parameter.fill_(7.0)
            for parameter in target.parameters():
                parameter.fill_(3.0)

        load_not_compatible_weights(
            target,
            {"model_state_dict": source.state_dict()},
            excluded_prefixes=("mask_estimators.",),
        )

        for parameter in target.backbone.parameters():
            self.assertTrue(torch.all(parameter == 7.0))
        for head in target.mask_estimators:
            for parameter in head.parameters():
                self.assertTrue(torch.all(parameter == 3.0))


def _config(*, instruments=("vocals", "drums"), target=None):
    training = SimpleNamespace(instruments=list(instruments))
    if target is not None:
        training.target_instrument = target
    return SimpleNamespace(training=training)


def test_checkpoint_metadata_accepts_legacy_and_matching_metadata():
    path = Path("model.ckpt")
    validate_checkpoint_metadata(
        {"weight": torch.zeros(1)},
        checkpoint_path=path,
        model_type="bs_roformer",
        config=_config(),
    )
    validate_checkpoint_metadata(
        {
            "meta": {
                "model_type": "bs_roformer",
                "target_instrument": "vocals",
                "target_stem_count": 1,
            }
        },
        checkpoint_path=path,
        model_type="bs_roformer",
        config=_config(target="vocals"),
    )


def test_checkpoint_metadata_rejects_newer_format():
    with pytest.raises(CheckpointError, match="Upgrade msst"):
        validate_checkpoint_metadata(
            {"meta": {"format_version": CHECKPOINT_FORMAT_VERSION + 1}},
            checkpoint_path=Path("future.ckpt"),
            model_type="bs_roformer",
            config=_config(),
        )


def test_checkpoint_metadata_builder_versions_model_artifacts():
    metadata = build_checkpoint_metadata(
        model_type="bs_roformer",
        instruments=["vocals", "drums"],
        target_instrument="vocals",
    )

    assert metadata["format_version"] == CHECKPOINT_FORMAT_VERSION
    assert metadata["msst_version"]
    assert metadata["model_type"] == "bs_roformer"
    assert metadata["target_instrument"] == "vocals"
    assert metadata["target_stem_count"] == 1


@pytest.mark.parametrize(
    ("meta", "config", "message"),
    [
        ({"model_type": "scnet"}, _config(), "declares model_type"),
        (
            {"target_instrument": "piano", "target_stem_count": 1},
            _config(),
            "absent from training.instruments",
        ),
        (
            {"target_instrument": "vocals", "target_stem_count": 1},
            _config(),
            "no matching training.target_instrument",
        ),
        (
            {"target_stem_count": 1},
            _config(),
            "config selects 2",
        ),
        (
            {"target_stem_count": "many"},
            _config(),
            "expected a positive integer",
        ),
    ],
)
def test_checkpoint_metadata_reports_specific_mismatch(meta, config, message):
    with pytest.raises(CheckpointError, match=message):
        validate_checkpoint_metadata(
            {"meta": meta},
            checkpoint_path=Path("model.ckpt"),
            model_type="bs_roformer",
            config=config,
        )


def test_training_checkpoint_preserves_public_compatibility_metadata(tmp_path):
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    output = tmp_path / "training.ckpt"
    args = argparse.Namespace(
        train_lora_peft=False,
        train_lora_loralib=False,
        checkpoint_meta={
            "model_type": "bs_roformer",
            "target_instrument": "vocals",
            "target_stem_count": 1,
        },
    )

    save_weights(
        str(output),
        model,
        [0],
        optimizer,
        2,
        {},
        [],
        1.5,
        args,
    )
    checkpoint = torch.load(output, weights_only=False, map_location="cpu")

    assert checkpoint["meta"] == args.checkpoint_meta
    assert set(checkpoint["model_state_dict"]) == set(model.state_dict())


if __name__ == "__main__":
    unittest.main()
