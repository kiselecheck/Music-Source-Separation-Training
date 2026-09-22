# Real-model parity benchmark: bs_roformer

- Result: **PASS**
- Upstream revision: `050cae7345f4ac1e1e27e066c2c5cdc0a2cdb679`
- Package revision: `72224dff7e8159a009f7fb7bf0d6a5afbc3568a6`
- Audio duration: 3.000 s; input rate 44100 Hz
- Checkpoint SHA-256: `3e9daecd70aaed5b5a0d1f861cc4d77eaa45afb3fc6301b1cf32c1be0f5868fb`

| Stem | Maximum absolute error | RMSE | Pass |
|---|---:|---:|:---:|
| `audio/bass.wav` | 0 | 0 | True |
| `audio/drums.wav` | 0 | 0 | True |
| `audio/other.wav` | 0 | 0 | True |
| `audio/vocals.wav` | 0 | 0 | True |

| Implementation | Load (s) | Processing median (s) | Real-time factor |
|---|---:|---:|---:|
| Upstream | 3.281 | 83.210 | 0.036x |
| MSST | 3.389 | 83.560 | 0.036x |

MSST/upstream processing-time ratio: 1.004x; lower is faster. Timings are descriptive and are not pass/fail thresholds.