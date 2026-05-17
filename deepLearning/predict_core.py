"""
Local inference orchestration.

Uses ``deepLearning.minimal_predict.run_minimal_predict`` for forward + postprocess,
``deepLearning.utils`` for model load, and ``deepLearning.utils.annot.Events`` for CSV/H5.
"""

from __future__ import annotations

import glob
import logging
import os
from typing import Optional

import flammkuchen
import librosa
import numpy as np


def run_inference(
    *,
    path: str,
    model_save_name: str,
    save_filename: Optional[str] = None,
    out_dir: Optional[str] = None,
    save_format: str = "csv",
    verbose: int = 1,
    batch_size: Optional[int] = None,
    event_thres: float = 0.5,
    event_dist: float = 0.01,
    event_dist_min: float = 0,
    event_dist_max: float = np.inf,
    segment_thres: float = 0.5,
    segment_use_optimized: bool = True,
    segment_minlen: Optional[float] = None,
    segment_fillgap: Optional[float] = None,
    bandpass_low_freq: Optional[float] = None,
    bandpass_up_freq: Optional[float] = None,
    resample: bool = True,
) -> None:
    """
    WAV file or folder → annotations CSV/H5 (CSV columns match ``*_annotations.csv`` used in prepare).

    ``model_save_name``: path prefix without ``_model.h5`` (same as training output).

    Output path:
      * Single WAV: ``save_filename`` if given; else ``out_dir / <stem>_annotations.csv`` if ``out_dir``;
        else ``<wav_stem>_annotations.csv`` beside the WAV (or ``_predict.h5`` when ``save_format='h5'``).
      * Folder of WAVs: ``save_filename`` is ignored; each file → ``out_dir / <stem>_…`` when ``out_dir`` is set,
        otherwise beside each WAV.
    """
    from deepLearning.minimal_predict import run_minimal_predict
    from deepLearning.utils.annot import Events
    from deepLearning.utils import utils

    if save_format not in ("csv", "h5"):
        raise ValueError("save_format must be 'csv' or 'h5'.")

    suffix = "_predict.h5" if save_format == "h5" else "_annotations.csv"

    is_dir = os.path.isdir(path)
    if is_dir:
        dir_path = os.path.abspath(path)
        filenames = sorted(glob.glob(os.path.join(dir_path, "*.wav")))
        filenames = [f for f in filenames if not os.path.isdir(f)]
    elif os.path.isfile(path):
        filenames = [path]
    else:
        raise FileNotFoundError(path)

    if is_dir and save_filename is not None:
        logging.warning("%s is a folder; ignoring save_filename=%s (use --out_dir for a common output folder).", path, save_filename)

    out_root: Optional[str] = None
    if out_dir is not None:
        out_root = os.path.abspath(os.path.expanduser(out_dir))
        os.makedirs(out_root, exist_ok=True)

    def resolve_out_path(recording_filename: str) -> str:
        if save_filename is not None and not is_dir:
            return str(save_filename)
        stem = os.path.splitext(os.path.basename(recording_filename))[0]
        if out_root is not None:
            return os.path.join(out_root, stem + suffix)
        return os.path.splitext(recording_filename)[0] + suffix

    logging.info("Loading model from %s", model_save_name)
    model, params = utils.load_model_and_params(model_save_name)

    for recording_filename in filenames:
        logging.info("Loading audio %s", recording_filename)
        try:
            x, fs_audio = librosa.load(recording_filename, sr=None, mono=False)
            x = x.T
            if x.ndim == 1:
                x = x[:, np.newaxis]

            events, segments, class_probabilities, class_names = run_minimal_predict(
                x,
                model,
                params,
                fs_audio=fs_audio,
                verbose=verbose,
                batch_size=batch_size,
                event_thres=event_thres,
                event_dist=event_dist,
                event_dist_min=event_dist_min,
                event_dist_max=event_dist_max,
                segment_thres=segment_thres,
                segment_use_optimized=segment_use_optimized,
                segment_minlen=segment_minlen,
                segment_fillgap=segment_fillgap,
                resample=resample,
                bandpass_low_freq=bandpass_low_freq,
                bandpass_up_freq=bandpass_up_freq,
            )

            if "event" in params["class_types"]:
                logging.info(
                    "Events: %s types %s",
                    len(events["seconds"]),
                    list(set(events["sequence"])),
                )
            if "segment" in params["class_types"]:
                logging.info(
                    "Segments: %s instances %s",
                    len(segments["onsets_seconds"]),
                    list(set(segments["sequence"])),
                )

            out_path = resolve_out_path(recording_filename)

            if save_format == "h5":
                payload = {
                    "events": events,
                    "segments": segments,
                    "class_probabilities": class_probabilities,
                    "class_names": class_names,
                }
                logging.info("Saving %s", out_path)
                flammkuchen.save(out_path, payload)
            else:
                evt = Events.from_predict(events, segments)
                logging.info("Saving %s", out_path)
                evt.to_df().to_csv(out_path)

        except Exception:
            logging.exception("Error processing %s", recording_filename)
