"""Load training data from a ``.npy`` directory layout (numpy file tree under a ``*.npy`` path)."""

from __future__ import annotations

import os.path

from deepLearning import npy_dir


def load(location: str) -> npy_dir.DictClass:
    """Load npy-dir dataset produced by ``dataset.build_npy_dataset`` / ``npy_dir.save``."""
    location = os.path.normpath(location)
    if not location.endswith(".npy"):
        raise ValueError(
            f'Expected dataset path ending in ".npy" (directory layout), got: {location}. '
            "For .zarr / .h5 backends, extend deepLearning/io.py or convert to this ``*.npy`` layout."
        )
    return npy_dir.load(location)
