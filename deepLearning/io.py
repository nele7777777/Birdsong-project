"""Load training data from a ``.npy`` directory layout (vendored subset of upstream DAS ``io``)."""

from __future__ import annotations

import os.path

from deepLearning import npy_dir


def _select(data, x_suffix, y_suffix):
    for lvl in ["test", "val", "train"]:
        if lvl in data:
            if "y_" + y_suffix in data[lvl]:
                data[lvl]["y"] = data[lvl]["y_" + y_suffix]
                if "eventtimes_" + y_suffix in data[lvl]:
                    data[lvl]["eventtimes"] = data[lvl]["eventtimes_" + y_suffix]
            if "x_" + x_suffix in data[lvl]:
                data[lvl]["x"] = data[lvl]["x_" + x_suffix]
                if "eventtimes_" + x_suffix in data[lvl]:
                    data[lvl]["eventtimes"] = data[lvl]["eventtimes_" + x_suffix]

    if f"samplerate_x_{x_suffix}_Hz" in data.attrs:
        data.attrs["samplerate_x_Hz"] = data.attrs[f"samplerate_x_{x_suffix}_Hz"]

    if "class_names_" + y_suffix in data.attrs and "class_types_" + y_suffix in data.attrs:
        data.attrs["class_names"] = data.attrs["class_names_" + y_suffix]
        data.attrs["class_types"] = data.attrs["class_types_" + y_suffix]
    return data


def load(location: str, x_suffix: str = "", y_suffix: str = "") -> npy_dir.DictClass:
    """Load npy-dir dataset produced by ``dataset.build_npy_dataset`` / DAS ``npy_dir.save``."""
    location = os.path.normpath(location)
    if not location.endswith(".npy"):
        raise ValueError(
            f'Expected dataset path ending in ".npy" (directory layout), got: {location}. '
            "For .zarr / .h5 use upstream das.io or extend deepLearning/io.py."
        )
    data = npy_dir.load(location)
    if len(x_suffix) or len(y_suffix):
        data = _select(data, x_suffix, y_suffix)
    return data
