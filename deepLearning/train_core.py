"""
Training loop for syllable (frame) classification.

Loads a ``*.npy`` dataset directory → ``AudioSequence`` → TCN → ``fit`` → checkpoint + YAML.

Imports the training stack from ``deepLearning.utils`` (I/O, ``AudioSequence``, model zoo, utilities).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

logger = logging.getLogger(__name__)

# GPU memory growth (avoid allocating full VRAM up front)
try:
    for device in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(device, enable=True)
except Exception:
    logger.exception("GPU memory growth setup")


def run_classification_training(
    *,
    data_dir: str,
    save_dir: str,
    save_prefix: str = "",
    save_name: Optional[str] = None,
    model_name: str = "tcn",
    nb_filters: int = 16,
    kernel_size: int = 16,
    nb_conv: int = 3,
    nb_hist: int = 1024,
    batch_size: int = 32,
    nb_epoch: int = 200,
    verbose: int = 2,
    dilations: Optional[list] = None,
    ignore_boundaries: bool = True,
    nb_pre_conv: int = 0,
    pre_nb_dft: int = 64,
    reduce_lr: bool = False,
    reduce_lr_patience: int = 5,
    version_data: bool = True,
) -> Tuple[keras.Model, Dict[str, Any], keras.callbacks.History]:
    """
    Train a frame-wise classifier (``with_y_hist=False``) on a ``*.npy`` dataset directory.

    Omits: augmentation, wandb, tensorboard, post_opt grid search, full test-set report.
    """
    from deepLearning import data_hash, io
    from deepLearning.utils import data, models, utils

    if dilations is None:
        dilations = [1, 2, 4, 8, 16]

    # Classification settings (with_y_hist=False branch)
    with_y_hist = False
    return_sequences = False
    stride = 1
    y_offset = int(round(nb_hist / 2))
    upsample = False
    data_padding = 0
    sample_weight_mode = None
    output_stride = int(2**nb_pre_conv) if not upsample else 1

    if save_prefix:
        save_prefix = save_prefix.rstrip("_") + "_"
    else:
        save_prefix = ""

    params: Dict[str, Any] = {
        "data_dir": data_dir,
        "save_dir": save_dir,
        "model_name": model_name,
        "nb_filters": nb_filters,
        "kernel_size": kernel_size,
        "nb_conv": nb_conv,
        "use_separable": False,
        "nb_hist": nb_hist,
        "ignore_boundaries": ignore_boundaries,
        "batch_norm": True,
        "nb_pre_conv": nb_pre_conv,
        "pre_nb_dft": pre_nb_dft,
        "pre_kernel_size": 3,
        "pre_nb_filters": 16,
        "upsample": upsample,
        "dilations": dilations,
        "nb_lstm_units": 0,
        "verbose": verbose,
        "batch_size": batch_size,
        "nb_epoch": nb_epoch,
        "reduce_lr": reduce_lr,
        "reduce_lr_patience": reduce_lr_patience,
        "first_sample_train": 0,
        "last_sample_train": None,
        "first_sample_val": 0,
        "last_sample_val": None,
        "seed": None,
        "nb_stacks": 2,
        "with_y_hist": with_y_hist,
        "return_sequences": return_sequences,
        "stride": stride,
        "y_offset": y_offset,
        "data_padding": data_padding,
        "sample_weight_mode": sample_weight_mode,
        "output_stride": output_stride,
        "balance": False,
        "class_weights": None,
        "morph_nb_kernels": 0,
        "morph_kernel_duration": 32,
        "tmse_weight": 0.0,
        "resnet_compute": False,
        "resnet_train": False,
    }

    logger.info("Loading data from %s", data_dir)
    d = io.load(data_dir)
    params.update(dict(d.attrs))

    if version_data:
        params["data_hash"] = data_hash.hash_data(data_dir)
        logger.info("data_hash=%s", params["data_hash"])

    params.update(
        {
            "nb_freq": d["train"]["x"].shape[1],
            "nb_channels": d["train"]["x"].shape[-1],
            "nb_classes": len(params["class_names"]),
        }
    )

    first_sample_train = params["first_sample_train"]
    last_sample_train = params["last_sample_train"]
    first_sample_val = params["first_sample_val"]
    last_sample_val = params["last_sample_val"]

    logger.info("Building sequences (classification, stride=%s, y_offset=%s)", stride, y_offset)
    data_gen = data.AudioSequence(
        d["train"]["x"],
        d["train"]["y"],
        shuffle=True,
        shuffle_subset=None,
        first_sample=first_sample_train,
        last_sample=last_sample_train,
        nb_repeats=1,
        batch_processor=None,
        **params,
    )
    val_gen = data.AudioSequence(
        d["val"]["x"],
        d["val"]["y"],
        shuffle=False,
        shuffle_subset=None,
        first_sample=first_sample_val,
        last_sample=last_sample_val,
        **params,
    )
    logger.info("Train: %s", data_gen)
    logger.info("Val:   %s", val_gen)

    logger.info("Building model %r", model_name)
    try:
        model = models.model_dict[model_name](**params)
    except KeyError as e:
        raise ValueError(f"Unknown model_name={model_name!r}; allowed: {list(models.model_dict)}") from e

    logger.info(model.summary())

    os.makedirs(os.path.abspath(save_dir), exist_ok=True)
    if save_name is None:
        save_name = time.strftime("%Y%m%d_%H%M%S")
    save_path = f"{save_dir}/{save_prefix}{save_name}"
    params["save_name"] = save_path
    logger.info("Checkpoint prefix: %s", save_path)

    checkpoint_path = save_path + "_model.h5"
    callbacks = [
        ModelCheckpoint(
            checkpoint_path,
            save_best_only=True,
            save_weights_only=False,
            monitor="val_loss",
            verbose=1,
        ),
        EarlyStopping(monitor="val_loss", patience=20, verbose=1),
    ]
    if reduce_lr:
        callbacks.append(ReduceLROnPlateau(patience=reduce_lr_patience, verbose=1))

    utils.save_params(params, save_path)

    logger.info("Training (epochs=%s)", nb_epoch)
    history = model.fit(
        data_gen,
        epochs=nb_epoch,
        steps_per_epoch=min(len(data_gen), 1000),
        verbose=verbose,
        validation_data=val_gen,
        callbacks=callbacks,
    )

    tf.keras.backend.clear_session()
    return model, params, history
