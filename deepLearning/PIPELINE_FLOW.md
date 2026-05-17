# End-to-end flow: audio input to annotation output

This document describes the **prepare → train → predict** call chain under `deepLearning/`, tensor/file meanings, and how they map to `data_formats.md`.

## 1. Overview

```mermaid
flowchart LR
  subgraph prepare["prepare"]
    W[WAV folder]
    A["*_annotations.csv"]
    D["dataset.build_npy_dataset"]
    N["*.npy dir\n(train/val/test + attrs)"]
    W --> D
    A --> D
    D --> N
  end
  subgraph train["train"]
    IO["io.load"]
    SEQ["utils.data.AudioSequence"]
    TC["utils.models TCN"]
    CK["*_model.h5 + YAML params"]
    N --> IO --> SEQ --> TC --> CK
  end
  subgraph predict["predict"]
    LB["librosa.load WAV"]
    INF["predict_core.run_inference"]
    MP["minimal_predict.run_minimal_predict"]
    OUT["*_annotations.csv / *_predict.h5"]
    CK --> INF
    LB --> INF --> MP --> OUT
  end
  N --> train
```

CLI entry: `python -m deepLearning.pipeline prepare|train|predict` (see `pipeline.py`).

---

## 2. Stage 1: Prepare data (audio + CSV → `.npy` dataset directory)

### 2.1 Input conventions

- **Audio**: `.wav` files paired with `*_annotations.csv` by shared core id via `dataset._resolve_wav_for_annotation` (optional `strip_prefix` on WAV names).
- **Annotation CSV**: `{core}_annotations.csv` with at least `name`, `start_seconds`, `stop_seconds` (see `data_formats.md` and common export tools). Placeholder rows (NaN `start`/`stop`) are skipped when building matrices.

### 2.2 Core code path

| Step | Files and roles |
|------|-----------------|
| Scan WAV–CSV pairs, global class list | `dataset.py`: `_collect_pairs`, `_global_class_names` |
| Per recording: load WAV, frame-level label matrix | `dataset.py`: `build_npy_dataset`; `make_dataset.make_annotation_matrix` marks intervals as 1, then `normalize_probabilities` so each frame sums to 1 (including `noise`) |
| Write to disk | `npy_dir.py`: `save` → directory name ends in `.npy`, contains `train/x.npy`, `train/y.npy`, `val/...`, `attrs`, etc. |

### 2.3 `attrs` and training

`build_npy_dataset` writes `data.attrs` including at least:

- `samplerate_x_Hz` / `samplerate_y_Hz`: match WAV sample rate.
- `class_names`: e.g. `["noise", "syllable_a", ...]` (**index 0 is noise**).
- `class_types`: currently **every class is `"segment"`** (interval/syllable task); extending `event` types later requires coordinated changes in `dataset._global_class_names` and `minimal_predict` event branches.

`train_core.run_classification_training` loads this via `io.load`, merges `attrs` into `params` for `AudioSequence` and model construction.

---

## 3. Stage 2: Train (`.npy` → TCN → checkpoint)

### 3.1 Load data

- **`io.load(data_dir)`** (`io.py`): path must end in `.npy`; `npy_dir.load` returns `DictClass`; each split uses keys `x` and `y`.

### 3.2 Training loop

- **`train_core.run_classification_training`** (`train_core.py`):
  - Fixed **frame classification** path: `with_y_hist=False`, `y_offset ≈ nb_hist/2`, `stride=1`.
  - Build train/val `AudioSequence` from `x`, `y` (shuffled / not shuffled).
  - `models.model_dict[model_name]` (default `tcn`) builds the graph; `utils.save_params` writes full `params` YAML beside weights.
  - `ModelCheckpoint` saves `save_path + "_model.h5"` (**for inference, `model_save_name` is the prefix without `_model.h5`**).

### 3.3 Tensor shapes (aligned with inference)

- `x`: `[time_samples, freq/feature_dim, channels]` or `[T, C]`; this pipeline writes `[T, 1]` mono waveform from `dataset`; `nb_freq` in `train_core` comes from `x.shape[1]`.
- `y`: `[T, nb_classes]`, each row is class probabilities summing to 1.

---

## 4. Stage 3: Predict (WAV → probability sequence → events/segments → CSV or H5)

### 4.1 Orchestration

- **`predict_core.run_inference`** (`predict_core.py`):
  1. `utils.load_model_and_params(model_save_name)` loads Keras model and YAML `params` (must match training: `samplerate_x_Hz`, `nb_hist`, `class_names`, `class_types`, etc.).
  2. `librosa.load` WAV; `x` shaped `[T, channels]` (multi-channel transposed to time × channel).
  3. Per file: **`minimal_predict.run_minimal_predict`**.

### 4.2 Forward pass and post-processing

- **`minimal_predict.run_minimal_predict`** (`minimal_predict.py`):
  - Optional bandpass, **resample to `params["samplerate_x_Hz"]`** (must match training rate or frames misalign).
  - Batched `AudioSequence` + `model.predict_on_batch` → **frame-level `class_probabilities` `[T, nb_classes]`** (same class dim as training `y`).
  - **`class_names[i]` ↔ column `i`** of the probability matrix; `noise` is usually index 0.
  - If `class_types` includes `"segment"`: `_predict_segments_numpy` thresholds segment dims, `fill_gaps` / `remove_short` → `onsets_seconds` / `offsets_seconds` / `sequence`, etc.
  - If `"event"`: `_predict_events_numpy` on those dims (with current all-`segment` `dataset`, events dict is often empty).

### 4.3 Write annotations

- **CSV**: `Events.from_predict(events, segments)` → `to_df()` → `to_csv`, default `{wav_stem}_annotations.csv` (compatible with prepare input).
- **H5**: `flammkuchen.save` stores `events`, `segments`, `class_probabilities`, `class_names`; default `<recording stem>_predict.h5`.

---

## 5. Classes and time axis (summary)

| Concept | Location |
|---------|----------|
| Class name list | `params["class_names"]` / dataset `attrs["class_names"]` |
| Frame probability for class `k` | `class_probabilities[:, k]` |
| Training target | `y[:, k]`, same `T` rows and sample rate |
| Output syllable interval names | `segments["sequence"]` aligned with `onsets_seconds` / `offsets_seconds`; merged into CSV `name` / `start_seconds` / `stop_seconds` |

---

## 6. Consistency notes

1. **`data_formats.md`** may use legacy names like `classnames`; Python code and `attrs` use **`class_names` / `class_types`** — cross-check `dataset.py` / `train_core.py`.
2. **`dataset.build_npy_dataset`** sets all **`class_types` to `"segment"`**; `make_dataset.infer_class_info` can distinguish `event` but that path is not wired. Pulse/event classes in annotations will not use the event branch until designed end-to-end.
3. **`pipeline.py` prepare** matches implementation: public entry is `dataset.build_npy_dataset`, which calls `make_dataset` and `npy_dir` internally.
4. **`utils/annot.py` `Events.from_df`** docs use column `stop_seconds`, matching implementation and CSV columns.

See `TRAINING.md` for training parameters; `data_formats.md` for on-disk layout.
