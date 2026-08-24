import argparse
from pathlib import Path
from types import SimpleNamespace

from ml_collections import ConfigDict

import numpy as np
import pytest
import torch

from msst import _inference as inference
from msst import AudioFileError
from msst.utils.model_utils import demix


def _config(*, normalize=False, channels=2):
    return SimpleNamespace(
        audio=SimpleNamespace(sample_rate=44100, num_channels=channels),
        inference={"normalize": normalize},
        training=SimpleNamespace(instruments=["vocals"]),
    )


def _args(**overrides):
    values = {
        "model_type": "fake",
        "disable_detailed_pbar": True,
        "bigshifts": 1,
        "use_tta": False,
        "extract_instrumental": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_separate_audio_uses_inference_mode_and_expands_mono(monkeypatch):
    observed = []

    def fake_bigshifts(config, model, mix, device, **kwargs):
        observed.append((mix.copy(), torch.is_inference_mode_enabled()))
        return {"vocals": np.full_like(mix, 0.25)}

    monkeypatch.setattr(inference, "bigshifts_wrapper", fake_bigshifts)

    result = inference.separate_audio(
        object(),
        _args(),
        _config(),
        "cpu",
        np.zeros(16, dtype=np.float32),
    )

    assert observed[0][0].shape == (2, 16)
    assert observed[0][1] is True
    assert result["vocals"].shape == (2, 16)


def test_separate_audio_handles_silence_when_normalization_enabled(monkeypatch):
    def fake_bigshifts(config, model, mix, device, **kwargs):
        assert np.isfinite(mix).all()
        return {"vocals": mix.copy()}

    monkeypatch.setattr(inference, "bigshifts_wrapper", fake_bigshifts)

    result = inference.separate_audio(
        object(),
        _args(),
        _config(normalize=True),
        "cpu",
        np.zeros((2, 8), dtype=np.float32),
    )

    assert np.isfinite(result["vocals"]).all()


@pytest.mark.parametrize(
    "waveform, message",
    [
        (np.empty((2, 0)), "at least one sample"),
        (np.zeros((1, 2, 3)), "Audio must have shape"),
        (np.array([0.0, np.nan]), "NaN"),
    ],
)
def test_separate_audio_rejects_invalid_waveforms(waveform, message):
    with pytest.raises(ValueError, match=message):
        inference.separate_audio(
            object(),
            _args(),
            _config(),
            "cpu",
            waveform,
        )


def test_format_filename_rejects_output_traversal():
    with pytest.raises(ValueError, match="unsafe"):
        inference.format_filename("../{instr}", instr="vocals")

    assert inference.format_filename(
        "{file_name}/{instr}", file_name="song", instr="vocals"
    ) == (["song"], "vocals")


def test_run_folder_is_fail_fast_for_library_calls(monkeypatch, tmp_path):
    input_folder = tmp_path / "input"
    input_folder.mkdir()
    (input_folder / "broken.wav").write_bytes(b"not audio")
    output_folder = tmp_path / "output"
    args = _args(
        input_folder=str(input_folder),
        store_dir=str(output_folder),
        skip_errors=False,
    )
    monkeypatch.setattr(
        inference,
        "librosa",
        SimpleNamespace(
            load=lambda *args, **kwargs: (_ for _ in ()).throw(
                ValueError("decode failed")
            )
        ),
    )

    with pytest.raises(
        AudioFileError,
        match=r"Cannot process audio file .*broken\.wav: ValueError: decode failed",
    ) as caught:
        inference.run_folder(object(), args, _config(), "cpu", verbose=True)

    assert isinstance(caught.value.__cause__, ValueError)

    args.skip_errors = True
    assert inference.run_folder(object(), args, _config(), "cpu", verbose=True) == []


def test_write_stems_honors_flac_flag(monkeypatch, tmp_path):
    writes = []
    monkeypatch.setattr(
        inference.sf,
        "write",
        lambda path, data, sample_rate, subtype: writes.append(
            (Path(path), sample_rate, subtype)
        ),
    )
    args = _args(
        flac_file=True,
        pcm_type="PCM_24",
        filename_template="{file_name}/{instr}",
        start_check_point="model.ckpt",
        store_dir=str(tmp_path),
        draw_spectro=0,
    )

    paths = inference._write_stems(
        {"vocals": np.zeros((2, 10), dtype=np.float32)},
        args=args,
        sample_rate=44100,
        relative_path=Path("album/song.wav"),
        started_at=1,
    )

    assert paths == [str(tmp_path / "song" / "vocals.flac")]
    assert writes == [(tmp_path / "song" / "vocals.flac", 44100, "PCM_24")]


def test_format_filename_rejects_unknown_and_malformed_fields():
    with pytest.raises(ValueError, match="Unknown filename_template field"):
        inference.format_filename("{file_nam}/{instr}", file_name="song", instr="vocals")
    with pytest.raises(ValueError, match="Invalid filename_template"):
        inference.format_filename("{file_name", file_name="song")


def test_prepare_audio_scales_integer_pcm_and_validates_channels():
    prepared = inference._prepare_audio(
        np.array([0, 32767, -32768], dtype=np.int16),
        _config(channels=2),
    )
    assert prepared.shape == (2, 3)
    assert prepared.dtype == np.float32
    assert np.max(np.abs(prepared)) <= 1

    with pytest.raises(ValueError, match="expects 2 audio channel"):
        inference._prepare_audio(np.zeros((3, 8)), _config(channels=2))


def test_run_folder_filters_non_audio_and_preserves_relative_path(monkeypatch, tmp_path):
    input_folder = tmp_path / "input"
    (input_folder / "album-a").mkdir(parents=True)
    (input_folder / "album-b").mkdir()
    (input_folder / "album-a" / "song.wav").write_bytes(b"fake")
    (input_folder / "album-b" / "song.flac").write_bytes(b"fake")
    (input_folder / "notes.txt").write_text("ignore me")
    output_folder = tmp_path / "output"
    args = _args(
        input_folder=str(input_folder),
        store_dir=str(output_folder),
        skip_errors=False,
        flac_file=False,
        pcm_type="FLOAT",
        filename_template="{relative_path}/{instr}",
        start_check_point="model.ckpt",
        draw_spectro=0,
    )
    monkeypatch.setattr(
        inference.librosa,
        "load",
        lambda *args, **kwargs: (np.zeros((2, 8), dtype=np.float32), 44100),
    )
    monkeypatch.setattr(
        inference,
        "separate_audio",
        lambda *args, **kwargs: {"vocals": np.zeros((2, 8), dtype=np.float32)},
    )
    writes = []
    monkeypatch.setattr(
        inference.sf,
        "write",
        lambda path, *args, **kwargs: writes.append(Path(path)),
    )

    result = inference.run_folder(object(), args, _config(), "cpu", verbose=True)

    assert result == [
        str(output_folder / "album-a" / "song" / "vocals.wav"),
        str(output_folder / "album-b" / "song" / "vocals.wav"),
    ]
    assert writes == [Path(path) for path in result]


def test_run_folder_rejects_same_directory_and_empty_audio_set(tmp_path):
    input_folder = tmp_path / "input"
    input_folder.mkdir()
    (input_folder / "notes.txt").write_text("not audio")
    args = _args(input_folder=str(input_folder), store_dir=str(input_folder))
    with pytest.raises(ValueError, match="must be different"):
        inference.run_folder(object(), args, _config(), "cpu", verbose=True)

    args.store_dir = str(tmp_path / "output")
    with pytest.raises(AudioFileError, match="No supported audio files"):
        inference.run_folder(object(), args, _config(), "cpu", verbose=True)


def test_run_folder_rejects_custom_template_collisions(monkeypatch, tmp_path):
    input_folder = tmp_path / "input"
    (input_folder / "album-a").mkdir(parents=True)
    (input_folder / "album-b").mkdir()
    (input_folder / "album-a" / "song.wav").write_bytes(b"fake")
    (input_folder / "album-b" / "song.flac").write_bytes(b"fake")
    args = _args(
        input_folder=str(input_folder),
        store_dir=str(tmp_path / "output"),
        skip_errors=False,
        flac_file=False,
        pcm_type="FLOAT",
        filename_template="{file_name}/{instr}",
        start_check_point="model.ckpt",
        draw_spectro=0,
    )
    monkeypatch.setattr(
        inference.librosa,
        "load",
        lambda *args, **kwargs: (np.zeros((2, 8), dtype=np.float32), 44100),
    )
    monkeypatch.setattr(
        inference,
        "separate_audio",
        lambda *args, **kwargs: {"vocals": np.zeros((2, 8), dtype=np.float32)},
    )
    monkeypatch.setattr(inference.sf, "write", lambda *args, **kwargs: None)

    with pytest.raises(AudioFileError, match="maps multiple inputs"):
        inference.run_folder(object(), args, _config(), "cpu", verbose=True)


def test_run_folder_can_disambiguate_same_basename_with_extension(monkeypatch, tmp_path):
    input_folder = tmp_path / "input"
    input_folder.mkdir()
    (input_folder / "song.wav").write_bytes(b"fake")
    (input_folder / "song.flac").write_bytes(b"fake")
    args = _args(
        input_folder=str(input_folder),
        store_dir=str(tmp_path / "output"),
        skip_errors=False,
        flac_file=False,
        pcm_type="FLOAT",
        filename_template="{file_name}_{extension}/{instr}",
        start_check_point="model.ckpt",
        draw_spectro=0,
    )
    monkeypatch.setattr(
        inference.librosa,
        "load",
        lambda *args, **kwargs: (np.zeros((2, 8), dtype=np.float32), 44100),
    )
    monkeypatch.setattr(
        inference,
        "separate_audio",
        lambda *args, **kwargs: {"vocals": np.zeros((2, 8), dtype=np.float32)},
    )
    monkeypatch.setattr(inference.sf, "write", lambda *args, **kwargs: None)

    result = inference.run_folder(object(), args, _config(), "cpu", verbose=True)

    assert result == [
        str(tmp_path / "output" / "song_flac" / "vocals.wav"),
        str(tmp_path / "output" / "song_wav" / "vocals.wav"),
    ]

def test_resolve_device_rejects_invalid_cuda_ids(monkeypatch):
    monkeypatch.setattr(inference.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(inference.torch.cuda, "device_count", lambda: 2)
    with pytest.raises(RuntimeError, match="detected 2 CUDA device"):
        inference.resolve_device(_args(force_cpu=False, device_ids=[2]))
    with pytest.raises(RuntimeError, match="cannot be negative"):
        inference.resolve_device(_args(force_cpu=False, device_ids=[-1]))
    with pytest.raises(RuntimeError, match="must be unique"):
        inference.resolve_device(_args(force_cpu=False, device_ids=[0, 0]))


@pytest.mark.parametrize("supports_active_heads", [False, True])
def test_demix_selects_target_from_multistem_model(supports_active_heads):
    config = ConfigDict(
        {
            "audio": {"chunk_size": 8},
            "inference": {"batch_size": 1, "num_overlap": 1},
            "training": {
                "instruments": ["first", "target", "last"],
                "target_instrument": "target",
                "use_amp": False,
            },
        }
    )

    if supports_active_heads:
        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mask_estimators = torch.nn.ModuleList(
                    [torch.nn.Identity() for _ in range(3)]
                )
                self.active = None

            def forward(self, audio, active_stem_ids=None):
                self.active = active_stem_ids
                return torch.ones((audio.shape[0], 1, *audio.shape[1:]))
    else:
        class FakeModel(torch.nn.Module):
            def forward(self, audio):
                values = torch.arange(3, dtype=audio.dtype).view(1, 3, 1, 1)
                return values.expand(audio.shape[0], 3, *audio.shape[1:])

    model = FakeModel()
    result = demix(
        config,
        model,
        np.zeros((2, 8), dtype=np.float32),
        torch.device("cpu"),
        model_type="fake",
    )

    assert list(result) == ["target"]
    assert result["target"].shape == (2, 8)
    assert np.all(result["target"] == 1)
    if supports_active_heads:
        assert model.active == [1]


def test_demix_accepts_float_demucs_segment_and_rejects_zero_step():
    class FakeDemucs(torch.nn.Module):
        def forward(self, audio):
            return audio[:, None]

    config = ConfigDict(
        {
            "inference": {"batch_size": 1, "num_overlap": 1},
            "training": {
                "samplerate": 8,
                "segment": 1.0,
                "instruments": ["stem"],
                "use_amp": False,
            },
        }
    )
    result = demix(
        config,
        FakeDemucs(),
        np.zeros((2, 8), dtype=np.float32),
        torch.device("cpu"),
        model_type="htdemucs",
    )
    # Historical single-stem Demucs mode returns the source array directly.
    assert result.shape == (1, 2, 8)

    config.inference.num_overlap = 9
    with pytest.raises(ValueError, match="would make the inference step zero"):
        demix(
            config,
            FakeDemucs(),
            np.zeros((2, 8), dtype=np.float32),
            torch.device("cpu"),
            model_type="htdemucs",
        )
