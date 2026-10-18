"""Plugin discovery tests that need only MSST's base dependencies."""

from types import SimpleNamespace

import pytest

import msst
import msst.models.registry as registry
from examples.custom_model import TinySeparator
from msst.utils.settings import get_model_from_config


MODEL_NAME = "test_entry_point_separator"


class EntryPoints(list):
    def select(self, *, group):
        assert group == "msst.models"
        return self


@pytest.fixture(autouse=True)
def clean_registration():
    msst.unregister_model(MODEL_NAME)
    yield
    msst.unregister_model(MODEL_NAME)


def test_entry_point_is_lazy_and_works_through_config_loader(monkeypatch, tmp_path):
    class Plugin:
        name = MODEL_NAME
        value = "example_plugin.model:TinySeparator"
        dist = SimpleNamespace(name="example-plugin")
        loaded = False

        def load(self):
            self.loaded = True
            return TinySeparator

    plugin = Plugin()
    monkeypatch.setattr(registry, "entry_points", lambda: EntryPoints([plugin]))
    assert MODEL_NAME in msst.registered_models()
    assert not plugin.loaded

    config_path = tmp_path / "plugin.yaml"
    config_path.write_text(
        "audio:\n  num_channels: 2\n"
        "training:\n  instruments: [vocals, other]\n"
        "inference: {}\n",
        encoding="utf-8",
    )
    model, _config = get_model_from_config(MODEL_NAME, str(config_path))
    assert isinstance(model, TinySeparator)
    assert plugin.loaded


def test_broken_or_ambiguous_entry_points_fail_clearly(monkeypatch):
    class BrokenPlugin:
        name = MODEL_NAME
        value = "example_plugin.model:MissingModel"
        dist = SimpleNamespace(name="example-plugin")

        def load(self):
            raise ImportError("plugin dependency is missing")

    monkeypatch.setattr(
        registry, "entry_points", lambda: EntryPoints([BrokenPlugin()])
    )
    with pytest.raises(ValueError, match="plugin dependency is missing.*Install the plugin"):
        registry.get_model_registration(MODEL_NAME)

    second = SimpleNamespace(
        name=MODEL_NAME,
        value="other_plugin.model:Separator",
        dist=SimpleNamespace(name="other-plugin"),
    )
    monkeypatch.setattr(
        registry, "entry_points", lambda: EntryPoints([BrokenPlugin(), second])
    )
    with pytest.raises(ValueError, match="Multiple installed MSST model plugins"):
        registry.get_model_registration(MODEL_NAME)
