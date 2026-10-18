"""Runtime registry for models supplied by MSST extensions.

Built-in architectures intentionally remain on the legacy factory for now.  The
registry is an extension seam: a third-party package can register a model in
its import path and use the normal MSST config, checkpoint, inference, and
standard-training workflows without editing this repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import entry_points
import inspect
import re
from typing import Any, Callable, TypeVar

import torch
from torch import nn

from msst.errors import ConfigurationError
from msst.models.builtin import BUILTIN_MODEL_TYPES


ModelFactory = Callable[..., nn.Module]
_ModelT = TypeVar("_ModelT", bound=nn.Module)

_MODEL_NAME = re.compile(r"^[a-z][a-z0-9_-]*$")
_MODEL_ENTRY_POINT_GROUP = "msst.models"


class MSSTBaseModel(nn.Module):
    """Small contract for third-party models used by generic MSST workflows.

    A model receives a parsed config and its ``forward`` must return a tensor
    shaped ``[batch, stems, channels, time]`` for a generic model.  The base
    class deliberately does not impose an architecture or tensor internals;
    it only supplies a predictable config factory and an explicit capability
    flag for models with their own training loss.

    Subclasses may set ``uses_internal_loss = True`` and implement the legacy
    internal-loss call ``forward(mix, target, active_stem_ids=...)``.  Most
    custom models should leave it false and use MSST's standard loss path.
    """

    uses_internal_loss = False

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def from_config(
        cls: type[_ModelT], config: Any, *, pretrained: bool = False
    ) -> _ModelT:
        """Construct ``cls`` from the complete parsed MSST config.

        The default assumes ``__init__(config)``.  A custom model can override
        this method when it needs to read a different config section or load a
        pretrained backbone.
        """

        del pretrained
        return cls(config)  # type: ignore[call-arg]

    def forward(
        self,
        mixture: torch.Tensor,
        target: torch.Tensor | None = None,
        *,
        active_stem_ids: Any = None,
    ) -> torch.Tensor:
        if target is None:
            return self.separate(mixture, active_stem_ids=active_stem_ids)
        return self.compute_loss(
            mixture, target, active_stem_ids=active_stem_ids
        )

    def separate(
        self, mixture: torch.Tensor, *, active_stem_ids: Any = None
    ) -> torch.Tensor:
        """Return ``[batch, stems, channels, time]`` predictions."""

        raise NotImplementedError

    def compute_loss(
        self,
        mixture: torch.Tensor,
        target: torch.Tensor,
        *,
        active_stem_ids: Any = None,
    ) -> torch.Tensor:
        """Return a scalar loss for models with an internal training loss."""

        del mixture, target, active_stem_ids
        raise NotImplementedError(
            "Use MSST's standard loss or implement compute_loss()"
        )


@dataclass(frozen=True, slots=True)
class ModelRegistration:
    """Validated description of one registered extension model."""

    name: str
    factory: ModelFactory
    uses_internal_loss: bool = False

    def build(self, config: Any, *, pretrained: bool = False) -> nn.Module:
        """Build and validate the extension model instance."""

        creator = self.factory
        try:
            parameters = inspect.signature(creator).parameters
        except (TypeError, ValueError):
            parameters = {}
        accepts_pretrained = (
            "pretrained" in parameters
            or any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )
        )
        model = (
            creator(config, pretrained=pretrained)
            if accepts_pretrained
            else creator(config)
        )
        if not isinstance(model, nn.Module):
            raise TypeError(
                f"Registered model {self.name!r} returned {type(model).__name__}; "
                "a torch.nn.Module is required"
            )
        return model


_REGISTRY: dict[str, ModelRegistration] = {}


def _normalize_name(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError(f"model name must be a string, got {type(name).__name__}")
    normalized = name.strip().lower()
    if not normalized or not _MODEL_NAME.fullmatch(normalized):
        raise ValueError(
            "model name must start with a letter and contain only lowercase "
            "letters, digits, '_' or '-'"
        )
    return normalized


def _lookup_name(name: str) -> str:
    """Normalize a lookup without turning legacy unknown keys into API errors."""

    return name.strip().lower() if isinstance(name, str) else str(name).strip().lower()


def _factory_for(model: type[nn.Module] | ModelFactory) -> ModelFactory:
    if not callable(model):
        raise TypeError("model must be a torch.nn.Module class or callable factory")

    from_config = getattr(model, "from_config", None)
    if callable(from_config):
        return from_config
    return model


def _has_internal_loss(model: Any) -> bool:
    if isinstance(model, type) and issubclass(model, MSSTBaseModel):
        if "uses_internal_loss" in model.__dict__:
            return bool(model.__dict__["uses_internal_loss"])
        return model.compute_loss is not MSSTBaseModel.compute_loss
    return bool(getattr(model, "uses_internal_loss", False))


def register_model(
    name: str,
    model: type[nn.Module] | ModelFactory,
    *,
    uses_internal_loss: bool | None = None,
    overwrite: bool = False,
) -> ModelRegistration:
    """Register an external model for normal MSST model construction.

    Typical usage from a third-party package::

        import msst
        from .model import MimoMelBand

        msst.register_model("mimo_mel_band", MimoMelBand)

    ``model`` may be an :class:`MSSTBaseModel` subclass, any ``nn.Module``
    class accepting the complete config, or a factory callable.  Registration
    is process-local and intentionally explicit; importing a package is the
    opt-in needed to make its model available.
    """

    normalized = _normalize_name(name)
    if normalized in BUILTIN_MODEL_TYPES:
        raise ValueError(
            f"{normalized!r} is a built-in MSST model type and cannot be replaced"
        )
    if normalized in _REGISTRY and not overwrite:
        raise ValueError(f"Model {normalized!r} is already registered")

    factory = _factory_for(model)
    if uses_internal_loss is None:
        uses_internal_loss = _has_internal_loss(model)
    registration = ModelRegistration(
        name=normalized,
        factory=factory,
        uses_internal_loss=bool(uses_internal_loss),
    )
    _REGISTRY[normalized] = registration
    return registration


def unregister_model(name: str) -> None:
    """Remove one process-local extension registration."""

    _REGISTRY.pop(_normalize_name(name), None)


def get_model_registration(name: str) -> ModelRegistration | None:
    """Return a registration, loading an installed plugin only when selected.

    Built-in architectures remain handled by the legacy factory. External
    packages can expose short names through the ``msst.models`` entry point
    group; their model code is imported only when its name is requested.
    """

    normalized = _lookup_name(name)
    registration = _REGISTRY.get(normalized)
    if registration is not None or normalized in BUILTIN_MODEL_TYPES:
        return registration
    return _load_model_entry_point(normalized)


def _get_model_entry_points() -> tuple[Any, ...]:
    discovered = entry_points()
    if hasattr(discovered, "select"):
        return tuple(discovered.select(group=_MODEL_ENTRY_POINT_GROUP))
    # Compatibility with older importlib.metadata implementations.
    return tuple(discovered.get(_MODEL_ENTRY_POINT_GROUP, ()))


def _load_model_entry_point(name: str) -> ModelRegistration | None:
    matches = [
        entry_point
        for entry_point in _get_model_entry_points()
        if _lookup_name(entry_point.name) == name
    ]
    if not matches:
        return None
    if len(matches) > 1:
        providers = ", ".join(
            sorted(
                getattr(getattr(item, "dist", None), "name", "<unknown>")
                for item in matches
            )
        )
        raise ConfigurationError(
            f"Multiple installed MSST model plugins use the name {name!r}: "
            f"{providers}. Choose a unique entry point name."
        )

    entry_point = matches[0]
    try:
        model_factory = entry_point.load()
        registration = register_model(name, model_factory)
    except Exception as error:
        raise ConfigurationError(
            f"Cannot load MSST model plugin {name!r} from "
            f"{entry_point.value!r}: {type(error).__name__}: {error}. "
            "Install the plugin package and its dependencies, and ensure its "
            "entry point name is valid and not reserved."
        ) from error
    return registration


def registered_models() -> tuple[str, ...]:
    """Return registered and installed extension names without importing them."""

    discovered_names = {
        _normalize_name(entry_point.name) for entry_point in _get_model_entry_points()
    }
    return tuple(sorted(set(_REGISTRY).union(discovered_names)))


def model_uses_internal_loss(name: str) -> bool:
    """Resolve the training-loss mode for built-ins and extensions."""

    normalized = _lookup_name(name)
    registration = _REGISTRY.get(normalized)
    if registration is not None:
        return registration.uses_internal_loss
    from msst.models.builtin import BUILTIN_MODEL_FACTORIES

    factory = BUILTIN_MODEL_FACTORIES.get(normalized)
    return factory.uses_internal_loss if factory is not None else False


__all__ = [
    "BUILTIN_MODEL_TYPES",
    "MSSTBaseModel",
    "ModelRegistration",
    "get_model_registration",
    "model_uses_internal_loss",
    "register_model",
    "registered_models",
    "unregister_model",
]
