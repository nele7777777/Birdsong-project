# Deep learning pipeline (birdsong)

This folder contains a **self-contained** workflow: build a training dataset from manual annotations, train a frame-level TCN, then auto-label new WAV files. All model and data utilities live under `deepLearning.utils`; no separate segmentation package is required on `PYTHONPATH`.

The classic sklearn + MFCC flow lives under `traditionML/`; this path is for **frame-level / TCN** models on raw (or resampled) waveform input.

## Setup

From the repository root:

```bash
pip install tensorflow  # CPU or GPU build for your platform
# plus: librosa, numpy, pandas, pyyaml, scipy, tqdm, xarray, flammkuchen, … as imported by deepLearning
```

Install TensorFlow and other imports used by `deepLearning.utils.models` and `dataset.py` (see import lines or pin in your own `requirements.txt` if you add one).

## Data layout

- **WAV files** and **`{core}_annotations.csv`** with columns `name`, `start_seconds`, `stop_seconds` (same layout as many annotation GUIs). See `data_formats.md`.
- Optional **prefix** on WAV filenames (same idea as `traditionML/MFCC.py`): `--strip_prefix`.

## Commands

Run from repo root so imports resolve (`python -m` recommended):

### 1. Prepare dataset (`*.npy` directory)

Produces a training folder (e.g. `my_data.npy`) with `train/`, `val/`, `test/`, and `attrs` stored as `.npy` files under that directory name.

```bash
python -m deepLearning.pipeline prepare \
  --wav_dir /path/to/wavs \
  --annot_dir /path/to/annotations \
  --out_dataset /path/to/out/my_training.npy \
  --strip_prefix ""
```

### 2. Train

Uses **`deepLearning/train_core.py`** (`run_classification_training`): load npy → `AudioSequence` → TCN → `fit`. See **`TRAINING.md`** for the network flow.

```bash
python -m deepLearning.pipeline train \
  --data_dir /path/to/out/my_training.npy \
  --save_dir /path/to/runs \
  --save_prefix birds_ \
  --nb_epoch 200
```

Outputs include `{save_dir}/{save_prefix}{timestamp}_model.h5` and `_params.yaml`.  
Use the **path prefix without `_model.h5`** as `--model_prefix` for prediction (same string as the `save_name` prefix logged during training).

### 3. Predict (automatic syllable CSV)

Implemented in **`deepLearning/predict_core.py`** and **`minimal_predict.py`**.

```bash
python -m deepLearning.pipeline predict \
  --wav_path /path/to/new_recording.wav \
  --model_prefix /path/to/runs/birds_20260108_143022 \
  --save_format csv
```

For a folder of WAVs, omit `--save_csv`; each file gets `<stem>_annotations.csv`.  
With `--save_format h5`, the default sidecar name is `<stem>_predict.h5`.

## Relation to `traditionML`

| Step | Tradition ML | This pipeline |
|------|----------------|----------------|
| Labels | `.txt` per segment row | Time-aligned `_annotations.csv` |
| Features | MFCC CSV | Raw audio → learned frontend inside the TCN stack |
| Train | sklearn `classification_compare.py` | `pipeline train` → `train_core.py` |
| Inference | `model_predict.py` | `pipeline predict` → `predict_core.py` |

You can use both: e.g. this pipeline for boundary/syllable proposals, MFCC + SVM for refinement.
