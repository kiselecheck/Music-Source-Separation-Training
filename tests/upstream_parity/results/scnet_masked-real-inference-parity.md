# Real-model parity benchmark: scnet_masked

- Result: **PASS**
- Upstream revision: `050cae7345f4ac1e1e27e066c2c5cdc0a2cdb679`
- Package revision: `72224dff7e8159a009f7fb7bf0d6a5afbc3568a6`
- Audio duration: 3.000 s; input rate 44100 Hz
- Checkpoint SHA-256: `dcd70804f21d97a63e32e246ef6a8fe32644cbae6399be857df3215b3288ece9`

| Stem | Maximum absolute error | RMSE | Pass |
|---|---:|---:|:---:|
| `audio/bass.wav` | 0 | 0 | True |
| `audio/drums.wav` | 0 | 0 | True |
| `audio/other.wav` | 0 | 0 | True |
| `audio/vocals.wav` | 0 | 0 | True |

| Implementation | Load (s) | Processing median (s) | Real-time factor |
|---|---:|---:|---:|
| Upstream | 0.086 | 10.874 | 0.276x |
| MSST | 0.463 | 10.798 | 0.278x |

MSST/upstream processing-time ratio: 0.993x; lower is faster. Timings are descriptive and are not pass/fail thresholds.