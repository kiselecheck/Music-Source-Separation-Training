"""Lazy factories for MSST's built-in model architectures."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable


ModelBuilder = Callable[[Any, bool], Any]


@dataclass(frozen=True, slots=True)
class BuiltinModelFactory:
    builder: ModelBuilder
    config_section: str | None = "model"
    uses_internal_loss: bool = False

    def build(self, config: Any, *, pretrained: bool = False) -> Any:
        return self.builder(config, pretrained)


def _from_config(module_name: str, symbol_name: str) -> ModelBuilder:
    def build(config: Any, pretrained: bool) -> Any:
        del pretrained
        factory = getattr(import_module(module_name), symbol_name)
        return factory(config)

    return build


def _from_section(
    module_name: str,
    symbol_name: str,
    *,
    section_name: str = "model",
    pass_config: bool = False,
    pass_pretrained: bool = False,
) -> ModelBuilder:
    def build(config: Any, pretrained: bool) -> Any:
        factory = getattr(import_module(module_name), symbol_name)
        section = getattr(config, section_name)
        if pass_config:
            if pass_pretrained:
                return factory(config, pretrained=pretrained)
            return factory(config)
        if pass_pretrained:
            return factory(**dict(section), pretrained=pretrained)
        return factory(**section)

    return build


def _apollo(config: Any, pretrained: bool) -> Any:
    del pretrained
    base_model = getattr(import_module("msst.models.look2hear.models"), "BaseModel")
    return base_model.apollo(**config.model)


def _conformer(config: Any, pretrained: bool) -> Any:
    del pretrained
    module = import_module("msst.models.conformer_model")
    core = module.NeuralModel(**config.model)
    stft = config.stft
    return module.ConformerMSS(
        core=core,
        n_fft=stft.n_fft,
        hop_length=stft.hop_length,
        win_length=getattr(stft, "win_length", stft.n_fft),
        center=stft.center,
    )


# This mapping is the single source of built-in model names used both by the
# model factory and by the extension registry's collision checks.
BUILTIN_MODEL_FACTORIES: dict[str, BuiltinModelFactory] = {
    "mdx23c": BuiltinModelFactory(
        _from_config("msst.models.mdx23c_tfc_tdf_v3", "TFC_TDF_net")
    ),
    "htdemucs": BuiltinModelFactory(
        _from_config("msst.models.demucs4ht", "get_model"), config_section=None
    ),
    "segm_models": BuiltinModelFactory(
        _from_section(
            "msst.models.segm_models", "Segm_Models_Net",
            pass_config=True, pass_pretrained=True,
        )
    ),
    "torchseg": BuiltinModelFactory(
        _from_section(
            "msst.models.torchseg_models", "Torchseg_Net",
            pass_config=True, pass_pretrained=True,
        )
    ),
    "mel_band_roformer": BuiltinModelFactory(
        _from_section("msst.models.bs_roformer", "MelBandRoformer"),
        uses_internal_loss=True,
    ),
    "mel_band_conformer": BuiltinModelFactory(
        _from_section("msst.models.bs_roformer", "MelBandConformer"),
        uses_internal_loss=True,
    ),
    "mel_band_roformer_experimental": BuiltinModelFactory(
        _from_section(
            "msst.models.bs_roformer.mel_band_roformer_experimental",
            "MelBandRoformer",
        ),
        uses_internal_loss=True,
    ),
    "bs_roformer": BuiltinModelFactory(
        _from_section("msst.models.bs_roformer", "BSRoformer"),
        uses_internal_loss=True,
    ),
    "bs_conformer": BuiltinModelFactory(
        _from_section("msst.models.bs_roformer", "BSConformer"),
        uses_internal_loss=True,
    ),
    "bs_roformer_experimental": BuiltinModelFactory(
        _from_section(
            "msst.models.bs_roformer.bs_roformer_experimental", "BSRoformer"
        ),
        uses_internal_loss=True,
    ),
    "bs_mamba2": BuiltinModelFactory(
        _from_section(
            "msst.models.bs_mamba2_code.bs_mamba2", "BSMamba2Model"
        ),
        uses_internal_loss=True,
    ),
    "swin_upernet": BuiltinModelFactory(
        _from_section(
            "msst.models.upernet_swin_transformers", "Swin_UperNet_Model",
            pass_config=True, pass_pretrained=True,
        )
    ),
    "bandit": BuiltinModelFactory(
        _from_section(
            "msst.models.bandit.core.model", "MultiMaskMultiSourceBandSplitRNNSimple"
        )
    ),
    "bandit_v2": BuiltinModelFactory(
        _from_section(
            "msst.models.bandit_v2.bandit", "Bandit", section_name="kwargs"
        ),
        config_section="kwargs",
    ),
    "scnet_unofficial": BuiltinModelFactory(
        _from_section("msst.models.scnet_unofficial", "SCNet")
    ),
    "scnet": BuiltinModelFactory(
        _from_section("msst.models.scnet", "SCNet")
    ),
    "scnet_tran": BuiltinModelFactory(
        _from_section("msst.models.scnet.scnet_tran", "SCNet_Tran")
    ),
    "apollo": BuiltinModelFactory(_apollo),
    "experimental_mdx23c_stht": BuiltinModelFactory(
        _from_config("msst.models.mdx23c_tfc_tdf_v3_with_STHT", "TFC_TDF_net")
    ),
    "dttnet": BuiltinModelFactory(
        _from_config("msst.models.DTTNet", "DPTDFNet")
    ),
    "scnet_masked": BuiltinModelFactory(
        _from_section("msst.models.scnet.scnet_masked", "SCNet")
    ),
    "conformer": BuiltinModelFactory(_conformer),
    "moises_light": BuiltinModelFactory(
        _from_section("moises_light", "MoisesLight")
    ),
}

BUILTIN_MODEL_TYPES = frozenset(BUILTIN_MODEL_FACTORIES)


__all__ = ["BUILTIN_MODEL_FACTORIES", "BUILTIN_MODEL_TYPES", "BuiltinModelFactory"]
