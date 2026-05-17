"""Annotation matrix helpers (frame labels from interval tables)."""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def infer_class_info(df: pd.DataFrame):
    """Unique ``name`` values (sorted), no synthetic background class."""
    class_names = sorted({str(x).strip() for x in df["name"].dropna()})
    class_types: list[str] = []
    for name in class_names:
        sub = df[df["name"] == name].dropna(subset=["start_seconds", "stop_seconds"], how="all")
        if sub.empty:
            class_types.append("segment")
            continue
        fi = sub.index[0]
        if sub.loc[fi, "start_seconds"] == sub.loc[fi, "stop_seconds"]:
            class_types.append("event")
        else:
            class_types.append("segment")
    return class_names, class_types


def make_annotation_matrix(
    df: pd.DataFrame, nb_samples: int, samplerate: float, class_names: Optional[List[str]] = None
) -> np.ndarray:
    if class_names is None:
        class_names, _ = infer_class_info(df)
    class_matrix = np.zeros((nb_samples, len(class_names)))
    for _, row in df.iterrows():
        if row["name"] not in class_names:
            continue
        if np.all(np.isnan(row["start_seconds"])):
            continue
        class_index = class_names.index(row["name"])
        start_index = int(row["start_seconds"] * samplerate)
        stop_index = int(row["stop_seconds"] * samplerate + 1)
        if start_index < stop_index:
            class_matrix[start_index:stop_index, class_index] = 1
        else:
            logger.warning("%s should be greater than %s for row %s", start_index, stop_index, row)
    return class_matrix


def normalize_probabilities(p: np.ndarray) -> np.ndarray:
    """Each row sums to 1. Rows with no active class (all zeros) become uniform."""
    p = np.asarray(p, dtype=np.float64).copy()
    nb = p.shape[1]
    if nb == 0:
        return p
    row_sum = p.sum(axis=-1)
    bg = row_sum == 0
    if np.any(bg):
        p[bg, :] = 1.0 / nb
    fg = ~bg
    if np.any(fg):
        rs = p[fg].sum(axis=-1, keepdims=True)
        p[fg] = p[fg] / np.maximum(rs, 1e-12)
    return p
