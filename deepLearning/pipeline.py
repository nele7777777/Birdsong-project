#!/usr/bin/env python3
"""
Birdsong deep-learning pipeline (deepLearning/* — prepare / train / predict).

Subcommands:
  prepare   Build *.npy dataset from WAV + *_annotations.csv (``dataset.build_npy_dataset`` → make_dataset + npy_dir)
  train     Core training loop in deepLearning/train_core.py
  predict   Inference in deepLearning/predict_core.py + minimal_predict

Requires TensorFlow and the Python packages imported by deepLearning.utils / dataset.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def cmd_prepare(args: argparse.Namespace) -> None:
    from deepLearning.dataset import build_npy_dataset

    build_npy_dataset(
        args.wav_dir,
        args.annot_dir,
        args.out_dataset,
        strip_prefix=args.strip_prefix or "",
        val_fraction=args.val_fraction,
        seed=args.seed,
    )


def cmd_train(args: argparse.Namespace) -> None:
    from deepLearning.train_core import run_classification_training

    run_classification_training(
        data_dir=str(args.data_dir),
        save_dir=str(args.save_dir),
        save_prefix=args.save_prefix or "",
        save_name=args.save_name,
        model_name=args.model_name,
        nb_epoch=args.nb_epoch,
        batch_size=args.batch_size,
        nb_hist=args.nb_hist,
        verbose=args.verbose,
        version_data=not args.no_version_data,
    )


def cmd_predict(args: argparse.Namespace) -> None:
    from deepLearning.predict_core import run_inference

    run_inference(
        path=str(args.wav_path),
        model_save_name=str(args.model_prefix),
        save_filename=str(args.save_csv) if args.save_csv else None,
        out_dir=str(args.out_dir.expanduser().resolve()) if args.out_dir else None,
        save_format=args.save_format,
        verbose=args.verbose,
        batch_size=args.batch_size,
        segment_thres=args.segment_thres,
        segment_use_optimized=not args.no_segment_post_opt,
        segment_minlen=args.segment_minlen,
        segment_fillgap=args.segment_fillgap,
        event_thres=args.event_thres,
        resample=args.resample,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Birdsong deep-learning pipeline (prepare → train → predict).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_prep = sub.add_parser(
        "prepare",
        help="Build *.npy dataset directory from WAVs and *_annotations.csv files.",
    )
    p_prep.add_argument("--wav_dir", type=Path, required=True)
    p_prep.add_argument("--annot_dir", type=Path, required=True)
    p_prep.add_argument(
        "--out_dataset",
        type=Path,
        required=True,
        help="Output folder path ending with .npy (e.g. ./data/my_training.npy)",
    )
    p_prep.add_argument(
        "--strip_prefix",
        type=str,
        default="",
        help='Optional WAV filename prefix before id (e.g. "a b c d e f i ").',
    )
    p_prep.add_argument("--val_fraction", type=float, default=0.2)
    p_prep.add_argument("--seed", type=int, default=42)
    p_prep.set_defaults(func=cmd_prepare)

    p_tr = sub.add_parser(
        "train",
        help="Train TCN (frame classification) via deepLearning/train_core.py — see TRAINING.md.",
    )
    p_tr.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Dataset directory from prepare (*.npy folder).",
    )
    p_tr.add_argument("--save_dir", type=Path, required=True)
    p_tr.add_argument("--save_prefix", type=str, default="")
    p_tr.add_argument(
        "--save_name",
        type=str,
        default=None,
        help="Run name (default: timestamp). Final prefix is save_dir/save_prefix+save_name.",
    )
    p_tr.add_argument("--model_name", type=str, default="tcn")
    p_tr.add_argument("--nb_epoch", type=int, default=200)
    p_tr.add_argument("--batch_size", type=int, default=32)
    p_tr.add_argument("--nb_hist", type=int, default=1024)
    p_tr.add_argument("--verbose", type=int, default=2)
    p_tr.add_argument(
        "--no_version_data",
        action="store_true",
        help="Skip hashing data_dir (faster for huge datasets).",
    )
    p_tr.set_defaults(func=cmd_train)

    p_pr = sub.add_parser(
        "predict",
        help="Inference via deepLearning/predict_core.py + minimal_predict.",
    )
    p_pr.add_argument(
        "--wav_path",
        type=Path,
        required=True,
        help="One .wav file or a folder of .wav files.",
    )
    p_pr.add_argument(
        "--model_prefix",
        type=Path,
        required=True,
        help=(
            "Stem/path without _model.h5 — same as save_name prefix written by train "
            "(e.g. .../save_dir/prefix_20260101_120000)."
        ),
    )
    p_pr.add_argument(
        "--save_csv",
        type=Path,
        default=None,
        help="Output path (single WAV only). Overrides --out_dir for that case. Default: beside WAV.",
    )
    p_pr.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help=(
            "Write outputs here: folder of WAVs → <out_dir>/<stem>_annotations.csv for each file; "
            "single WAV without --save_csv → <out_dir>/<stem>_annotations.csv."
        ),
    )
    p_pr.add_argument("--save_format", type=str, default="csv", choices=("csv", "h5"))
    p_pr.add_argument("--verbose", type=int, default=1)
    p_pr.add_argument("--batch_size", type=int, default=None)
    p_pr.add_argument("--segment_thres", type=float, default=0.5)
    p_pr.add_argument(
        "--segment_minlen",
        type=float,
        default=None,
        help=(
            "Min voiced segment length in seconds after thresholding; shorter runs are removed. "
            "If set, overrides *_params.yaml post_opt min_len when --segment-post-opt is on."
        ),
    )
    p_pr.add_argument(
        "--segment_fillgap",
        type=float,
        default=None,
        help=(
            "Bridge silence gaps shorter than this (seconds) inside a voiced region. "
            "If set, overrides *_params.yaml post_opt gap_dur when --segment-post-opt is on."
        ),
    )
    p_pr.add_argument(
        "--no-segment-post-opt",
        action="store_true",
        help="Do not read post_opt min_len/gap_dur from params; only CLI values (if any) apply.",
    )
    p_pr.add_argument("--event_thres", type=float, default=0.5)
    p_pr.add_argument("--resample", action="store_true", default=True)
    p_pr.add_argument("--no_resample", action="store_false", dest="resample")
    p_pr.set_defaults(func=cmd_predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
