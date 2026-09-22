# Real-model parity benchmark: mdx23c

- Result: **PASS**
- Upstream revision: `050cae7345f4ac1e1e27e066c2c5cdc0a2cdb679`
- Package revision: `72224dff7e8159a009f7fb7bf0d6a5afbc3568a6`
- Audio duration: 3.000 s; input rate 44100 Hz
- Checkpoint SHA-256: `c1d31e3c47a98e8cdb12b3382a1a313c1a95c835da4310dd6350e7e1892f821c`

| Stem | Maximum absolute error | RMSE | Pass |
|---|---:|---:|:---:|
| `audio/bass.wav` | 0 | 0 | True |
| `audio/drums.wav` | 0 | 0 | True |
| `audio/other.wav` | 0 | 0 | True |
| `audio/vocals.wav` | 0 | 0 | True |

| Implementation | Load (s) | Processing median (s) | Real-time factor |
|---|---:|---:|---:|
| Upstream | 0.619 | 72.418 | 0.041x |
| MSST | 0.947 | 67.361 | 0.045x |

MSST/upstream processing-time ratio: 0.930x; lower is faster. Timings are descriptive and are not pass/fail thresholds.