"""
Inference path: forward pass + postprocessing.

Uses ``deepLearning.utils.data`` (``AudioSequence`` / ``unpack_batches``) for batches,
``deepLearning.utils.utils`` for audio prep, ``segment_utils`` / ``event_utils`` for postprocessing,
and ``labels_from_probabilities`` for thresholded / argmax labels.
Outputs match the structures expected by ``deepLearning.utils.annot.Events.from_predict``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover

    def tqdm(iterable, **kwargs):  # type: ignore[misc]
        return iterable

from deepLearning.utils import data, event_utils, segment_utils, utils


def labels_from_probabilities(
    probabilities: np.ndarray,
    threshold: Optional[float] = None,
    indices: Optional[Union[Sequence[int], slice]] = None,
) -> np.ndarray:
    """Convert class-wise probabilities into labels."""

    if indices is None:
        indices = slice(None)

    if probabilities.ndim == 1:
        if threshold is None:
            threshold = 0.5
        labels = (probabilities[:, indices] > threshold).astype(np.intp)
    elif probabilities.ndim == 2:
        if threshold is None:
            labels = np.argmax(probabilities[:, indices], axis=1)
        else:
            thresholded_probabilities = probabilities[:, indices].copy()
            thresholded_probabilities[thresholded_probabilities < threshold] = 0
            labels = np.argmax(thresholded_probabilities > threshold, axis=1)
    else:
        raise ValueError(
            f"Probabilities have to many dimensions ({probabilities.ndim}). Can only be 1D or 2D."
        )

    return labels


def predict_probabilities_numpy(
    x: np.ndarray,
    model,
    params: Dict[str, Any],
    *,
    verbose: int = 1,
    prepend_data_padding: bool = True,
) -> np.ndarray:
    """
    Run the model over the full recording in batches and return a dense
    ``numpy.ndarray`` of shape ``[T, nb_classes]`` (no zarr/dask).
    """
    pred_gen = data.AudioSequence(x=x, y=None, shuffle=False, **params)
    nb_batches = len(pred_gen)
    chunks: List[np.ndarray] = []

    for batch_number, batch_data in tqdm(
        enumerate(pred_gen),
        total=nb_batches,
        disable=verbose < 1,
    ):
        y_pred_batch = model.predict_on_batch(batch_data)
        y_pred_unpacked_batch = data.unpack_batches(y_pred_batch, pred_gen.data_padding)

        if prepend_data_padding:
            pad_width = None
            if batch_number == 0:
                pad_width = ((params["data_padding"], 0), (0, 0))
            elif batch_number == nb_batches - 1:
                pad_width = ((0, params["data_padding"]), (0, 0))
            if pad_width is not None:
                y_pred_unpacked_batch = np.pad(
                    y_pred_unpacked_batch,
                    pad_width=pad_width,
                    mode="constant",
                    constant_values=0,
                )

        chunks.append(y_pred_unpacked_batch)

    return np.concatenate(chunks, axis=0)


def _predict_segments_numpy(
    class_probabilities: np.ndarray,
    samplerate: float,
    params: Dict[str, Any],
    *,
    segment_thres: float,
    segment_minlen: Optional[float],
    segment_fillgap: Optional[float],
    segment_ref_onsets: Optional[List[float]] = None,
    segment_ref_offsets: Optional[List[float]] = None,
    segment_labels_by_majority: bool = True,
) -> Dict[str, Any]:
    segment_dims = np.where([val == "segment" for val in params["class_types"]])[0]
    segment_names = [str(params["class_names"][int(d)]) for d in segment_dims]

    segments: Dict[str, Any] = {}
    if len(segment_dims) == 0:
        return segments

    segments["samplerate_Hz"] = samplerate
    segments["index"] = segment_dims
    segments["names"] = segment_names
    segments["probabilities"] = class_probabilities[:, segment_dims]

    labels = labels_from_probabilities(class_probabilities, segment_thres, segment_dims)

    legacy_noise_song = (
        len(segment_dims) == 2
        and len(segment_names) >= 2
        and str(segment_names[0]).lower() == "noise"
    )
    if legacy_noise_song:
        song_binary = (labels > 0).astype(np.int8)
    else:
        seg_p = class_probabilities[:, segment_dims]
        song_binary = (np.max(seg_p, axis=1) >= segment_thres).astype(np.int8)
    if segment_fillgap is not None:
        song_binary = segment_utils.fill_gaps(
            song_binary,
            gap_dur=int(segment_fillgap * samplerate),
        )
    if segment_minlen is not None:
        song_binary = segment_utils.remove_short(
            song_binary,
            min_len=int(segment_minlen * samplerate),
        )

    onsets = np.where(np.diff(song_binary, prepend=np.int8(0)) == 1)[0]
    offsets = np.where(np.diff(song_binary, append=np.int8(0)) == -1)[0]

    segments["onsets_seconds"] = onsets.astype(np.float64) / samplerate
    segments["offsets_seconds"] = offsets.astype(np.float64) / samplerate

    if legacy_noise_song:
        labels_out = song_binary
        sequence = [str(segment_names[1])] * len(segments["offsets_seconds"])
    elif len(segment_dims) >= 2 and segment_labels_by_majority:
        ref_o = segment_ref_onsets
        ref_f = segment_ref_offsets
        if ref_o is None:
            ref_o = segments["onsets_seconds"]
        if ref_f is None:
            ref_f = segments["offsets_seconds"]
        sequence, labels_out = segment_utils.label_syllables_by_majority(
            labels,
            ref_o,
            ref_f,
            samplerate,
        )
    else:
        labels_out = labels
        if len(segment_dims) == 1 and len(segments["offsets_seconds"]):
            sequence = [str(segment_names[0])] * len(segments["offsets_seconds"])
        else:
            sequence = []

    segments["samples"] = labels_out
    segments["sequence"] = sequence
    return segments


def _predict_events_numpy(
    class_probabilities: np.ndarray,
    samplerate: float,
    params: Dict[str, Any],
    *,
    event_thres: float,
    event_dist: float,
    event_dist_min: float,
    event_dist_max: float,
    events_offset: float = 0.0,
) -> Dict[str, Any]:
    event_dims = np.where([val == "event" for val in params["class_types"]])[0]
    event_names = [str(params["class_names"][int(d)]) for d in event_dims]

    events: Dict[str, Any] = {}
    if len(event_dims) == 0:
        return events

    events["samplerate_Hz"] = samplerate
    events["index"] = event_dims
    events["names"] = event_names
    events["seconds"] = []
    events["probabilities"] = []
    events["sequence"] = []

    min_dist = int(event_dist * samplerate)
    if event_dist_max is None:
        event_dist_max = np.inf

    for event_dim, event_name in zip(event_dims, event_names):
        event_indices, event_confidence = event_utils.detect_events(
            class_probabilities,
            thres=event_thres,
            min_dist=min_dist,
            index=int(event_dim),
        )
        events_seconds = event_indices.astype(np.float64) / samplerate + events_offset
        good = event_utils.event_interval_filter(
            events_seconds,
            event_dist_min,
            event_dist_max,
        )
        events["seconds"].extend(events_seconds[good])
        ec = np.asarray(event_confidence, dtype=np.float64)
        events["probabilities"].extend(ec[good].tolist())
        events["sequence"].extend([event_name] * int(np.sum(good)))

    return events


def run_minimal_predict(
    x: np.ndarray,
    model,
    params: Dict[str, Any],
    *,
    fs_audio: Optional[float] = None,
    verbose: int = 1,
    batch_size: Optional[int] = None,
    event_thres: float = 0.5,
    event_dist: float = 0.01,
    event_dist_min: float = 0.0,
    event_dist_max: float = np.inf,
    segment_thres: float = 0.5,
    segment_use_optimized: bool = True,
    segment_minlen: Optional[float] = None,
    segment_fillgap: Optional[float] = None,
    pad: bool = True,
    prepend_data_padding: bool = True,
    bandpass_low_freq: Optional[float] = None,
    bandpass_up_freq: Optional[float] = None,
    resample: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any], np.ndarray, List[str]]:
    """
    Full-recording inference returning numpy probabilities (no dask/zarr temporary stores).

    Returns:
        events, segments, class_probabilities (np.ndarray), class_names
    """
    fs_model = params["samplerate_x_Hz"]

    if fs_audio is not None:
        fs_probs = fs_audio
        if bandpass_low_freq is not None or bandpass_up_freq is not None:
            logging.info(
                "   Filtering audio between %sHz and %sHz.",
                bandpass_low_freq,
                bandpass_up_freq,
            )
            x = utils.bandpass_filter_song(x, fs_audio, bandpass_low_freq, bandpass_up_freq)

        if resample and fs_audio != fs_model:
            logging.info(
                "   Resampling: audio rate is %s Hz; model expects %s Hz.",
                fs_audio,
                fs_model,
            )
            x = utils.resample(x, fs_audio, fs_model)
            fs_probs = fs_model
    else:
        fs_probs = fs_model

    if segment_use_optimized and "post_opt" in params and isinstance(params["post_opt"], dict):
        if segment_minlen is None:
            segment_minlen = params["post_opt"]["min_len"]
        if segment_fillgap is None:
            segment_fillgap = params["post_opt"]["gap_dur"]

    params_infer = dict(params)
    if batch_size is not None:
        params_infer["batch_size"] = batch_size

    x_len_original = len(x)
    pad_len = 0
    if pad:
        batch_len = params_infer["batch_size"] * params_infer["nb_hist"] + params_infer["nb_hist"]
        if np.remainder(len(x), batch_len) > 0:
            pad_len = batch_len - np.remainder(len(x), batch_len)
            x = np.pad(x, ((0, pad_len), (0, 0)), mode="edge")

    class_probabilities = predict_probabilities_numpy(
        x,
        model,
        params_infer,
        verbose=verbose,
        prepend_data_padding=prepend_data_padding,
    )

    if pad and pad_len > 0:
        nb_c = class_probabilities.shape[1]
        class_probabilities[-pad_len:, :] = 1.0 / max(nb_c, 1)
        class_probabilities = class_probabilities[:x_len_original, :].copy()

    segments = _predict_segments_numpy(
        class_probabilities,
        fs_probs,
        params,
        segment_thres=segment_thres,
        segment_minlen=segment_minlen,
        segment_fillgap=segment_fillgap,
    )

    events = _predict_events_numpy(
        class_probabilities,
        fs_probs,
        params,
        event_thres=event_thres,
        event_dist=event_dist,
        event_dist_min=event_dist_min,
        event_dist_max=event_dist_max,
    )

    class_names = list(params["class_names"])
    return events, segments, class_probabilities, class_names
