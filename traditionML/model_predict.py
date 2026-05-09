#!/usr/bin/env python3
"""
Load a joblib bundle saved by classification_compare.py (--cv_folds --save_model),
predict syllables from *_MFCC.csv + matching *_segments.csv.

Writes one CSV per clip: columns name,start_seconds,stop_seconds,channel
(same layout as *_annotations.csv).

Naming: <MFCC_stem_without__MFCC>_predict.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import joblib
from sklearn.preprocessing import LabelEncoder

_pkg_dir = Path(__file__).resolve().parent
if str(_pkg_dir) not in sys.path:
    sys.path.insert(0, str(_pkg_dir))

from classification_compare import load_feature_matrix, label_basename_from_feature_csv_stem


def load_segments_rows(seg_csv: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(seg_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            try:
                t0 = float(row["t_start_s"])
                t1 = float(row["t_end_s"])
            except (KeyError, ValueError):
                continue
            rows.append({"t_start_s": t0, "t_end_s": t1})
    return rows


def is_mfcc_features_csv(path: Path) -> bool:
    return path.suffix.lower() == ".csv" and path.stem.endswith("_MFCC")


def predict_directory(
    *,
    mfcc_dir: Path,
    segments_dir: Path,
    model_path: Path,
    out_dir: Path,
) -> None:
    bundle = joblib.load(model_path)
    model = bundle["model"]
    le: LabelEncoder = bundle["label_encoder"]
    n_feat = int(bundle["n_features"])

    mfcc_dir = Path(mfcc_dir)
    segments_dir = Path(segments_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mfcc_files = sorted(
        (p for p in mfcc_dir.glob("*.csv") if is_mfcc_features_csv(p)),
        key=lambda p: p.name.lower(),
    )
    if not mfcc_files:
        raise ValueError(f"No *_MFCC.csv under {mfcc_dir}")

    print(f"Using model: {bundle.get('best_model_name', '?')} ({model_path})")

    for mfcc_csv in mfcc_files:
        stem = mfcc_csv.stem
        core = label_basename_from_feature_csv_stem(stem)
        seg_csv = segments_dir / f"{core}_segments.csv"
        if not seg_csv.is_file():
            print(f"SKIP (no segments): {mfcc_csv.name} -> missing {seg_csv.name}")
            continue

        segments = load_segments_rows(seg_csv)
        X = load_feature_matrix(mfcc_csv)
        if X.shape[1] != n_feat:
            raise ValueError(
                f"{mfcc_csv.name}: feature columns {X.shape[1]} != trained {n_feat}"
            )
        if len(X) != len(segments):
            raise ValueError(
                f"{mfcc_csv.name}: {len(X)} MFCC rows vs {len(segments)} segment rows "
                f"in {seg_csv.name}"
            )

        y_enc = model.predict(X)
        labels = le.inverse_transform(y_enc)

        out_path = out_dir / f"{core}_predict.csv"
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["name", "start_seconds", "stop_seconds", "channel"])
            for seg, lab in zip(segments, labels):
                w.writerow(
                    [
                        str(lab).strip().lower(),
                        seg["t_start_s"],
                        seg["t_end_s"],
                        0.0,
                    ]
                )
        print(f"OK {mfcc_csv.name} -> {out_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Predict annotation-style CSV from MFCC + segments using a bundle from "
            "classification_compare.py (--cv_folds --save_model)."
        )
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        required=True,
        help="joblib path written by classification_compare --save_model.",
    )
    parser.add_argument(
        "--mfcc_dir",
        type=Path,
        required=True,
        help="Folder containing *_MFCC.csv files.",
    )
    parser.add_argument(
        "--segments_dir",
        type=Path,
        required=True,
        help="Folder containing matching *_segments.csv (stem matches MFCC without _MFCC).",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Output folder for <core>_predict.csv",
    )
    args = parser.parse_args()
    predict_directory(
        mfcc_dir=args.mfcc_dir,
        segments_dir=args.segments_dir,
        model_path=args.model_path,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
