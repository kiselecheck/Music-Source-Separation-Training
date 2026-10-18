 Dataset types for training

### Type 1 (MUSDB)

Different folders. Each folder contains all needed stems in format  
`<stem_name>.wav` (or `flac` in latest releases).

The structure is the same as in MUSDBHQ18.

Example:
```
--- Song 1:
------ vocals.wav
------ bass.wav
------ drums.wav
------ other.wav
--- Song 2:
------ vocals.wav
------ bass.wav
------ drums.wav
------ other.wav
--- Song 3:
...........
```

### Type 2 (Stems)

Each folder represents a single stem name.  
The folder contains audio files consisting only of that stem.

Example:
```
--- vocals:
------ vocals_1.wav
------ vocals_2.wav
------ vocals_3.wav
------ vocals_4.wav
------ ...
--- bass:
------ bass_1.wav
------ bass_2.wav
------ bass_3.wav
------ bass_4.wav
------ ...
...........
```

### Type 3 (CSV file)

You can provide a CSV file (or a list of CSV files) with the following structure:

```
instrum,path
vocals,/path/to/dataset/vocals_1.wav
vocals,/path/to/dataset2/vocals_v2.wav
vocals,/path/to/dataset3/vocals_some.wav
...
drums,/path/to/dataset/drums_good.wav
...
```

### Type 4 (MUSDB Aligned)

The same structure as Type 1, but during training all instruments are loaded from the same position of the song.

### Type 5 (Precomputed Chunks)

The same structure as Type 1, but all tracks are pre-split into chunks with 50% overlap.

### Type 6 (MUSDB Aligned + Explicit Mixture)

An extension of Type 4, designed for scenarios where the **mixture is treated as a separate signal**, rather than always being reconstructed as `sum(stems)`.

Structure is the same as Type 1 / Type 4, but it is recommended that each song folder contains `mixture.wav`.

Example:
```
--- Song 1:
------ vocals.wav
------ bass.wav
------ drums.wav
------ other.wav
------ mixture.wav
```

Key properties:
- All stems are loaded aligned from the same song position
- `mixture.wav` is loaded explicitly if present
- If `mixture.wav` is missing, mixture is computed as sum of stems
- Supports precomputed random chunks (same logic as Type 4)

Typical use cases:
- Teacher–student or distillation training
- Consistency losses
- Training with real mixes not equal to sum of stems

For sparse multi-stem datasets that already contain an explicit `mixture`, a
single-target model can reuse the original dataset without materializing a
separate `other` stem:

```yaml
training:
  instruments: ['synth_bass', 'other']
  target_instrument: synth_bass
  single_target_from_mixture: true
```

With this option, metadata contains only folders where `target_instrument`
exists by default. Missing non-target stems (such as `other`) are allowed, the
model input is read from the configured mixture, and the loss is computed only
for the configured target. This option is supported by dataset types 5 and 6.
Type 5 returns aligned precomputed chunks; type 6 samples aligned chunks from
full tracks.

For datasets without a stored mixture, set `training.mixture_source_instruments`
to the names of the component stem files that should be summed. On dataset type
6, target-absent tracks can be retained as negative examples with
`training.allow_missing_target: true`; their target is zero-filled while the
mixture remains the input. Set it to `false` for target-positive-only training.

Chunk filtering uses `audio.min_mean_abs`. On dataset type 6,
`training.precompute_chunks: true` builds a random chunk pool when this threshold
is greater than zero. `training.num_precompute_chunks` sets its size; if omitted,
it is derived from `num_steps`, batch size, and epoch count. Dataset type 5
continues to index the chunks already stored in its source dataset.


### Type 7 (Class-Balanced Aligned Dataset)

A class-balanced aligned dataset designed to reduce class frequency bias and improve learning of rare instruments.

Structure is the same as Type 6:
```
--- Song 1:
------ flute.wav
------ violin.wav
------ mixture.wav
```

How it works:
1. A random instrument (class) is selected
2. A random track containing this instrument is chosen
3. An aligned chunk is loaded from that track
4. The dataset returns which stems are actually present

Class frequency filtering:
- For each instrument, the ratio of tracks where it appears is computed
- Instruments appearing in more than `max_class_presence_ratio`
  (default: 0.4) of tracks are excluded
- Prevents dominant classes (e.g. vocals) from overwhelming training

Returned values:
- Stems tensor
- Mixture tensor (from `mixture.wav` if available, otherwise sum of stems)
- `active_stem_ids`: indices of instruments present in the current sample

Typical use cases:
- Training on sparse multi-instrument datasets
- Improving performance on rare instruments
- Conditional or multi-head source separation models

### Type 8 (Synthetic Dereverberation)

An aligned wet-to-dry dataset for dereverberation training.

The source dataset uses normal separated stems. Stem names may differ between
tracks:
```
--- Song 1:
------ drums.wav
------ bass.wav
------ other.wav
------ vocals.wav
--- Song 2:
------ violin_1.flac
------ violin_2.flac
------ cello.flac
```

The model target is a single dry stem named by `training.instruments`
and `training.target_instrument` (usually `orig`). Internally the dataset:
1. Chooses one track and an aligned chunk.
2. Loads all audio stems found in that track by default.
3. Builds `dry = sum(stems)`.
4. Randomly chooses no stems, all stems, or 1..N stems.
5. Adds reverb to the chosen stems using IR files and/or algorithmic reverb.
6. Returns `target=orig/dry` and `mixture=wet`.

Important config keys:
- `reverb.source_instruments`: omit it or set it to `all`, `auto`, or `*` to
  load every stem in each track. Set an explicit array such as
  `['drums', 'bass', 'other', 'vocals']` to keep the previous fixed-name
  behavior.
- `reverb.source_extensions`: extensions used for source discovery. Defaults
  to `['wav', 'flac']`.
- `reverb.exclude_instruments`: exact basenames ignored in automatic mode.
  Defaults to `['mixture', 'reverb', 'orig']`.
- `reverb.ir_paths`: folders or files with impulse responses.
- `reverb.method_weights`: weights for `ir`, `algorithmic`, and optional `pedalboard`.
- `reverb.none_probability`, `reverb.all_probability`, `reverb.min_reverb_stems`, `reverb.max_reverb_stems`: stem selection behavior.
- `reverb.wet_level_min/max`, `reverb.dry_level_min/max`: wet/dry blend for reverberated stems.

Typical use cases:
- `wet -> dry` dereverberation
- Training with on-the-fly randomized room coloration
- Avoiding overfitting to one specific IR dataset or one instrument class

### Dataset for validation

* The validation dataset must be the same structure as type 1 datasets (regardless of what type of dataset you're using for training), but also each folder must include `mixture.wav` for each song. `mixture.wav` - is the sum of all stems for song.

Example:
```
--- Song 1:
------ vocals.wav  
------ bass.wav 
------ drums.wav
------ other.wav
------ mixture.wav
--- Song 2:
------ vocals.wav  
------ bass.wav 
------ drums.wav
------ other.wav
------ mixture.wav
--- Song 3:
...........
```

For dereverberation validation, configs can set:
```
inference:
  mixture_names: ['reverb', 'mixture']
training:
  instruments: ['orig']
  target_instrument: orig
```
Then validation folders may contain `reverb.wav` as model input and `orig.wav` as the target.
