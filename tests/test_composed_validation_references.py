from argparse import Namespace
import numpy as np
import soundfile as sf
from ml_collections import ConfigDict


def _write_audio(path, value, frames=4096):
    sf.write(path, np.full((frames, 2), value, dtype=np.float32), 44100, subtype="FLOAT")


def _write_musdb_track(track):
    track.mkdir(parents=True)
    for instrument, value in {"bass": 0.1, "drums": 0.2, "other": 0.3, "vocals": 0.4}.items():
        _write_audio(track / f"{instrument}.wav", value)


def _config():
    return ConfigDict({"training": {"instruments": ["synth_bass", "other"],
        "target_instrument": "synth_bass", "single_target_from_mixture": True,
        "mixture_source_instruments": ["bass", "drums", "other", "vocals"]},
        "inference": {"mixture_names": ["mixture"]}})


def test_composed_validation_uses_reference_inside_track(tmp_path, monkeypatch):
    import torch
    import msst._validation as validation

    track = tmp_path / "track"
    _write_musdb_track(track)
    config = _config()
    config.training.instruments = ["bass", "other"]
    config.training.target_instrument = "bass"
    seen = []
    monkeypatch.setattr(validation, "demix", lambda config, model, mix, device, model_type: {"bass": mix * 0.1})
    def metrics(names, reference, estimate, mix, **kwargs):
        seen.append((reference.copy(), mix.copy()))
        return {"sdr": 7.0}
    monkeypatch.setattr(validation, "get_metrics", metrics)
    args = Namespace(valid_path=[str(tmp_path)], metrics=["sdr"], model_type="stub")
    result = validation.process_audio_files([str(track)], torch.nn.Identity(), args,
                                            config, torch.device("cpu"), is_tqdm=False)
    assert result["sdr"]["bass"] == [7.0]
    assert len(seen) == 1
    np.testing.assert_allclose(seen[0][0], 0.1, atol=1e-6)
    np.testing.assert_allclose(seen[0][1], 1.0, atol=1e-6)


def test_composed_targetless_track_uses_zero_reference_not_parent_stem(tmp_path, monkeypatch):
    import torch
    import msst._validation as validation

    track = tmp_path / "track"
    _write_musdb_track(track)
    _write_audio(tmp_path / "synth_bass.wav", 0.7)
    seen = []
    monkeypatch.setattr(validation, "demix", lambda config, model, mix, device, model_type: {"synth_bass": mix * 0.1})
    def metrics(names, reference, estimate, mix, **kwargs):
        seen.append(reference.copy())
        return {"sdr": 1.0}
    monkeypatch.setattr(validation, "get_metrics", metrics)
    args = Namespace(valid_path=[str(tmp_path)], metrics=["sdr"], model_type="stub")
    validation.process_audio_files([str(track)], torch.nn.Identity(), args, _config(),
                                   torch.device("cpu"), is_tqdm=False)
    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0], 0.0)
