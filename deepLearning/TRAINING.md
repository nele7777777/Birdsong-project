# Birdsong syllable training: flow and concepts

This document explains what training in `deepLearning/train_core.py` does, and what was trimmed compared to a full upstream training script.

## End-to-end flow

1. **Data** (from `prepare`): waveform \(x[t]\) and aligned label matrix \(y[t,c]\). Each time \(t\) is a row of class probabilities (including `noise`), built from manual `_annotations.csv` via frame-level one-hot / normalized probabilities (`deepLearning.make_dataset.normalize_probabilities`).
2. **Sliding window**: The network sees a context of **`nb_hist`** samples (default 1024), not a single sample. Each window predicts the class distribution at the **center** (or aligned) time — **frame-wise classification**.
3. **Frontend + TCN**: Default model name `tcn` maps to **`tcn_stft`** in `deepLearning.utils.models`: waveform → time–frequency representation, then **temporal convolutional network (TCN)** with dilated convolutions (e.g. 1,2,4,8,16) for large receptive field; easier to parallelize than LSTM for similar context.
4. **Loss**: Multi-class **`categorical_crossentropy`** (compiled in `deepLearning.utils.models`): predicted probabilities aligned with \(y\).
5. **Validation and early stopping**: `val_loss` on the validation set; `ModelCheckpoint` for best weights; `EarlyStopping` against overfitting.

**Inference** (`predict`): **`deepLearning.predict_core.run_inference`** calls **`deepLearning.minimal_predict.run_minimal_predict`** for forward pass and segment/event post-processing; outputs `_annotations.csv` or **`_predict.h5`**, paired with `_params.yaml` / `_model.h5`.

## Core concepts (why frame-level + TCN)

| Concept | Meaning |
|---------|---------|
| **Frame-level labels** | Each time point is noise or a syllable type; finer than one label per recording; suited to boundary detection. |
| **`nb_hist`** | Samples per decision window; larger → longer context, more memory and compute. |
| **`stride=1` (classification default)** | Windows advance one sample at a time (with center label) for dense predictions. |
| **`y_offset ≈ nb_hist/2`** | Label taken at window center so decisions align with the middle of acoustic events, not edges. |
| **TCN / dilated conv** | Stacked dilated convolutions see long context per frame; good for short structured sounds like birdsong. |

## How `train_core.py` differs from a heavy training script

**`run_classification_training`** keeps only the **classification + standard fit** path:

- **Kept**: `io.load`, `AudioSequence`, `models.model_dict`, `utils.save_params`, checkpoint / early stopping (same format as inference loading).
- **Omitted**: data augmentation, wandb/tensorboard, `post_opt` grid search, full test-set reports, etc. (fewer branches and dependencies).

**Dependencies**: TensorFlow and modules under `deepLearning/` only; no separate birdsong segmentation package required.

## Related files

- `deepLearning/dataset.py`: WAV + `*_annotations.csv` → `*.npy` dataset.
- `deepLearning/train_core.py`: streamlined training entry.
- `deepLearning/data_formats.md`: data dict and file layout.
