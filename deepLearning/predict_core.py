"""
Local inference orchestration (does not call ``das.predict.cli_predict`` or ``das.predict.predict``).

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
    WAV file or folder → annotations CSV/H5 (same layout as the upstream DAS CLI).

    ``model_save_name``: path prefix without ``_model.h5`` (same as training output).
    """
    from deepLearning.minimal_predict import run_minimal_predict
    from deepLearning.utils.annot import Events
    from deepLearning.utils import utils

    if save_format not in ("csv", "h5"):
        raise ValueError("save_format must be 'csv' or 'h5'.")

    if os.path.isdir(path) and save_filename is not None:
        logging.warning("%s is a folder; ignoring save_filename=%s", path, save_filename)

    if os.path.isdir(path):
        filenames = sorted(glob.glob(f"{path}/*.wav"))
        filenames = [f for f in filenames if not os.path.isdir(f)]
    elif os.path.isfile(path):
        filenames = [path]
    else:
        raise FileNotFoundError(path)

    logging.info("Loading model from %s", model_save_name)
    model, params = utils.load_model_and_params(model_save_name)

    explicit_out = None if os.path.isdir(path) else save_filename

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

            if save_format == "h5":
                payload = {
                    "events": events,
                    "segments": segments,
                    "class_probabilities": class_probabilities,
                    "class_names": class_names,
                }
                if explicit_out is None:
                    out_path = os.path.splitext(recording_filename)[0] + "_das.h5"
                else:
                    out_path = str(explicit_out)
                logging.info("Saving %s", out_path)
                flammkuchen.save(out_path, payload)
            else:
                evt = Events.from_predict(events, segments)
                if explicit_out is None:
                    out_path = os.path.splitext(recording_filename)[0] + "_annotations.csv"
                else:
                    out_path = str(explicit_out)
                logging.info("Saving %s", out_path)
                evt.to_df().to_csv(out_path)

        except Exception:
            logging.exception("Error processing %s", recording_filename)
