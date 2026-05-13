"""Annotation matrix helpers (vendored from upstream DAS ``make_dataset``)."""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def infer_class_info(df: pd.DataFrame):
    class_names, first_indices = np.unique(df["name"], return_index=True)
    class_names = list(class_names)
    class_names.insert(0, "noise")

    class_types = ["segment"]
    for first_index in first_indices:
        if df.loc[first_index]["start_seconds"] == df.loc[first_index]["stop_seconds"]:
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
    p_song = np.sum(p[:, 1:], axis=-1)

    p[p_song > 1.0, 1:] = p[p_song > 1.0, 1:] / p_song[p_song > 1.0, np.newaxis]
    p[:, 0] = 1 - np.sum(p[:, 1:], axis=-1)
    return p
