"""
Build a ``*.npy`` dataset directory from WAV + *_annotations.csv pairs.

Layout matches ``data_formats.md``: train/val splits with x, y and attrs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _resolve_wav_for_annotation(
    wav_dir: Path,
    core: str,
    strip_prefix: str,
) -> Path | None:
    """Same idea as traditionML/MFCC.resolve_wav_path: prefixed or bare core."""
    candidates = [
        wav_dir / f"{strip_prefix}{core}.wav",
        wav_dir / f"{core}.wav",
    ]
    for p in candidates:
        if p.is_file():
            return p.resolve()
    globs = sorted(wav_dir.glob(f"*{core}.wav")) + sorted(wav_dir.glob(f"*{core}.WAV"))
    uniq: list[Path] = []
    seen = set()
    for g in globs:
        try:
            r = g.resolve()
        except OSError:
            continue
        if r not in seen:
            seen.add(r)
            uniq.append(g)
    if len(uniq) == 1:
        return uniq[0]
    return None


def _collect_pairs(
    wav_dir: Path,
    annot_dir: Path,
    strip_prefix: str,
) -> List[Tuple[Path, Path, str]]:
    """List of (wav_path, annotation_csv_path, core_id)."""
    pairs: List[Tuple[Path, Path, str]] = []
    for ann in sorted(annot_dir.glob("*_annotations.csv"), key=lambda p: p.name.lower()):
        stem = ann.stem
        if not stem.endswith("_annotations"):
            continue
        core = stem[: -len("_annotations")]
        wav = _resolve_wav_for_annotation(wav_dir, core, strip_prefix)
        if wav is None:
            logger.warning("No WAV for annotation %s (core=%r)", ann.name, core)
            continue
        pairs.append((wav, ann, core))
    return pairs


def _global_class_names(pairs: List[Tuple[Path, Path, str]]) -> Tuple[List[str], List[str]]:
    names: set[str] = set()
    for _, ann, _ in pairs:
        df = pd.read_csv(ann)
        if "name" not in df.columns:
            raise ValueError(f"{ann}: need 'name' column")
        for n in df["name"].dropna():
            names.add(str(n).strip())
    class_names = sorted(names)
    class_types = ["segment"] * len(class_names)
    return class_names, class_types


def build_npy_dataset(
    wav_dir: Path | str,
    annot_dir: Path | str,
    out_dir: Path | str,
    *,
    strip_prefix: str = "",
    val_fraction: float = 0.2,
    seed: int = 42,
) -> Path:
    """
    Concatenate paired WAV + CSV into train/val npy-dir for ``train_core`` / ``io.load``.

    val_fraction: fraction of *recordings* used for val when >= 2 pairs; else time-split.
    """
    wav_dir = Path(wav_dir)
    annot_dir = Path(annot_dir)
    out_dir = Path(out_dir)

    from deepLearning.make_dataset import make_annotation_matrix, normalize_probabilities
    from deepLearning.npy_dir import DictClass, save as npy_save

    pairs = _collect_pairs(wav_dir, annot_dir, strip_prefix)
    if not pairs:
        raise ValueError(f"No WAV+annotation pairs under {wav_dir} / {annot_dir}")

    class_names, class_types = _global_class_names(pairs)
    logger.info("Classes (%d): %s", len(class_names), class_names)

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    metas: List[str] = []
    sr_ref: float | None = None

    for wav_path, ann_path, core in pairs:
        df = pd.read_csv(ann_path, encoding="utf-8-sig")
        for col in ("start_seconds", "stop_seconds"):
            if col not in df.columns:
                raise ValueError(f"{ann_path}: missing column {col}")

        y_audio, sr = librosa.load(str(wav_path), sr=None, mono=True)
        if sr_ref is None:
            sr_ref = float(sr)
        x = y_audio.astype(np.float32)[:, np.newaxis]
        nb_samples = x.shape[0]

        # Rows usable for matrix (skip nan-only placeholder rows from GUI)
        df_use = df.dropna(subset=["start_seconds", "stop_seconds"], how="all").copy()
        if "name" not in df_use.columns:
            raise ValueError(f"{ann_path}: need 'name' column")

        y_mat = make_annotation_matrix(df_use, nb_samples, sr, class_names=class_names)
        y_mat = normalize_probabilities(y_mat.astype(np.float64)).astype(np.float32)

        if y_mat.shape[0] != x.shape[0]:
            raise ValueError(f"{core}: len mismatch x vs y")

        xs.append(x)
        ys.append(y_mat)
        metas.append(core)
        logger.info("Loaded %s: %d samples @ %.1f Hz", core, nb_samples, sr)

    rng = np.random.default_rng(seed)

    if len(pairs) >= 2:
        n_val = max(1, int(round(len(pairs) * val_fraction)))
        if n_val >= len(pairs):
            n_val = len(pairs) - 1
        perm = np.arange(len(pairs))
        rng.shuffle(perm)
        val_idx = set(perm[:n_val].tolist())
        train_parts = [(xs[i], ys[i]) for i in range(len(pairs)) if i not in val_idx]
        val_parts = [(xs[i], ys[i]) for i in range(len(pairs)) if i in val_idx]
        if not train_parts:
            train_parts, val_parts = val_parts[:-1], val_parts[-1:]
        train_x = np.concatenate([p[0] for p in train_parts], axis=0)
        train_y = np.concatenate([p[1] for p in train_parts], axis=0)
        val_x = np.concatenate([p[0] for p in val_parts], axis=0)
        val_y = np.concatenate([p[1] for p in val_parts], axis=0)
    else:
        # Single recording: time-wise split
        x0, y0 = xs[0], ys[0]
        split = max(int(x0.shape[0] * (1.0 - val_fraction)), x0.shape[0] // 5)
        split = min(split, x0.shape[0] - 1024)
        if split <= 0:
            split = x0.shape[0] // 2
        train_x, val_x = x0[:split], x0[split:]
        train_y, val_y = y0[:split], y0[split:]
        logger.info("Single file: time split at sample %d / %d", split, x0.shape[0])

    assert sr_ref is not None
    data = DictClass()
    data.attrs = {
        "samplerate_x_Hz": sr_ref,
        "samplerate_y_Hz": sr_ref,
        "class_names": class_names,
        "class_types": class_types,
        "recording_ids": metas,
    }
    data["train"] = {"x": train_x, "y": train_y}
    data["val"] = {"x": val_x, "y": val_y}
    # Minimal test split (some evaluation helpers expect this key to exist)
    nb_hist = 1024
    if val_x.shape[0] >= nb_hist:
        data["test"] = {"x": val_x[:nb_hist].copy(), "y": val_y[:nb_hist].copy()}
    else:
        data["test"] = {"x": val_x.copy(), "y": val_y.copy()}

    out_dir.parent.mkdir(parents=True, exist_ok=True)

    npy_save(str(out_dir), data)
    logger.info("Wrote dataset -> %s", out_dir.resolve())
    logger.info("Shapes: train x=%s y=%s | val x=%s", train_x.shape, train_y.shape, val_x.shape)
    return out_dir
