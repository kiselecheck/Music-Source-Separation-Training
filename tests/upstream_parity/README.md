# Upstream functional-parity tests

These are source-repository tests, outside the installed `msst` runtime package.
They compare the MSST implementation with a separate ZFTurbo checkout pinned to
`050cae7345f4ac1e1e27e066c2c5cdc0a2cdb679`.

The fast suite checks:

- exact model-family registry and YAML-config parity;
- forward outputs for DTTNet, MDX23C, and SCNet with the same initialized weights;
- chunking, overlap/padding, BigShifts, and TTA for those model families;
- shared audio normalization/loading, training chunk reads (when the upstream
  augmentation dependencies are unavailable, harmless import placeholders are
  used because `load_chunk` does not call them), spectral-loss values and
  gradients, and one Adam optimizer update.

Run it from the MSST repository root in PowerShell:

```powershell
git worktree add --detach ..\zfturbo-upstream-parity 050cae7345f4ac1e1e27e066c2c5cdc0a2cdb679
$env:MSST_UPSTREAM_ROOT = (Resolve-Path ..\zfturbo-upstream-parity).Path
python -m pytest tests\upstream_parity -q
```

The release-checkpoint tests are opt-in. Set the matching config/checkpoint
variables for any of MDX23C, SCNet Masked, and BS Roformer. Audio and detailed reports are optional;
without an audio path the benchmark makes a deterministic stereo fixture. By
default it takes a short leading fragment so large CPU models remain practical.

```powershell
$env:MSST_PARITY_MDX23C_CONFIG = (Resolve-Path configs\config_musdb18_mdx23c.yaml).Path
$env:MSST_PARITY_MDX23C_CHECKPOINT = 'C:\weights\model_mdx23c_ep_168_sdr_7.0207.ckpt'
$env:MSST_PARITY_SCNET_MASKED_CONFIG = 'C:\weights\config_musdb18_scnet_small.yaml'
$env:MSST_PARITY_SCNET_MASKED_CHECKPOINT = 'C:\weights\model_scnet_masked_ep_156_sdr_8.8149.ckpt'
$env:MSST_PARITY_BS_ROFORMER_CONFIG = 'C:\weights\config_bs_roformer_384_8_2_485100.yaml'
$env:MSST_PARITY_BS_ROFORMER_CHECKPOINT = 'C:\weights\model_bs_roformer_ep_17_sdr_9.6568.ckpt'
$env:MSST_PARITY_AUDIO = 'C:\audio\mixture.wav'
$env:MSST_PARITY_AUDIO_SECONDS = '3'
$env:MSST_PARITY_REPEATS = '1'
$env:MSST_PARITY_REPORT_DIR = (Join-Path (Get-Location) 'tests\upstream_parity\results')
python -m pytest tests\upstream_parity\test_real_inference.py -q -vv
```

The end-to-end comparison loads the same checkpoint, runs the source tree's
folder-inference path, and compares stem filenames, sample rate, channel/frame
counts, WAV subtype, and every output sample. It records checkpoint/config/audio
hashes, code revisions and working-tree state, load time, processing time,
real-time factor, and numerical errors in JSON plus a short Markdown report.
The timing ratio is descriptive (not a flaky pass/fail threshold). For a manual
run, use `benchmark_real_inference.py --help`.

The release reports currently saved in `results/` use local release checkpoints
and a real audio fragment. The full 10-second source is not copied into this
repository. `PCM FLOAT` output and a shared filename template are selected in
both implementations to compare separation results without codec quantization
or naming-policy differences.

Known intentional behavior differences to keep distinct from model parity:

- MSST guards normalization of silent audio against division by zero; the
  upstream implementation does not. Normal non-silent inputs remain equivalent.
- MSST's public API defaults to preserving relative input subfolders in output
  names, while the legacy upstream CLI defaults to `{file_name}/{instr}`. Tests
  pass an explicit shared template.
- MSST rejects unsafe/colliding output templates and ignores unsupported files;
  upstream's older folder loop tries to decode every file and silently skips
  decode errors. Outputs for supported audio files are compared.
