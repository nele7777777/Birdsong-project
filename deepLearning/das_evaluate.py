#!/usr/bin/env python3
"""
Run DAS ``evaluate()`` — **frame-level** metrics aligned with training (same ``AudioSequence``, ``nb_hist``, labels from ``y``).

Use this when you want **F1 / confusion matrix** that match how the TCN was trained. CSV-based
``compare_annotation_csvs.py`` uses a separate time raster and will often **disagree** with these numbers.

Outputs go to ``deepLearning/das_eval_outputs/<run_tag>/`` by default (CSV/JSON/H5 + readable summary).

Examples (repo root)::

    python deepLearning/das_evaluate.py \\
        --stem /path/to/run/20260227_184334

    python deepLearning/das_evaluate.py \\
        --stem /path/to/run/20260227_184334 \\
        --out-dir /path/to/custom_out

Requirements: TensorFlow/Keras compatible with your ``*_model.h5``, plus DAS deps
(zarr<3, tqdm, scipy, …). Package ``das`` is loaded from ``<repo>/das/src`` automatically.

If ``_params.yaml`` still lists a ``data_dir`` from another machine (e.g. ``D:/...``), pass
``--data-dir /abs/path/to/your_dataset.npy`` to override.

If you see skipping because ``test/x`` has fewer frames than ``nb_hist`` (~1024 by default),
use ``--split val`` or ``--split train``, or rely on ``--split auto`` (default).

Unless ``--quiet``, the script prints **confusion matrix + per-class F1** to stdout after the run.

Use ``--strip-noise-pred`` to forbid the model from choosing class ``noise`` at each frame (argmax over other classes only);
true labels still include ``noise`` — useful to see syllable confusions without noise dominating the predicted column.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

# NumPy 2 removed ``np.unicode_`` / ``np.string_``; older flammkuchen expects them during HDF5 save.
if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_  # type: ignore[misc]
if not hasattr(np, "string_"):
    np.string_ = np.bytes_  # type: ignore[misc]

# -----------------------------------------------------------------------------
# Bootstrap: locate repo root and prepend das/src **before** ``import das``
# -----------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DAS_SRC = _REPO_ROOT / "das" / "src"
if _DAS_SRC.is_dir():
    sys.path.insert(0, str(_DAS_SRC))
else:
    raise RuntimeError(f"Expected DAS package at {_DAS_SRC}; clone or symlink janclemenslab/das there.")

DEFAULT_OUT_PARENT = Path(__file__).resolve().parent / "das_eval_outputs"

logger = logging.getLogger(__name__)


def _run_tag(stem: str) -> str:
    p = Path(stem.rstrip("/"))
    return p.name if p.name else p.resolve().name


def _save_confusion_csv(path: Path, cm: np.ndarray, class_names: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "," + ",".join(class_names)
    lines = [header]
    for i, row_name in enumerate(class_names):
        lines.append(",".join([row_name] + [str(int(x)) for x in cm[i]]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _report_dict_to_readable_text(report: dict) -> str:
    """Approximate sklearn text layout from classification_report(..., output_dict=True)."""
    lines: list[str] = []
    # class rows (skip metrics keys that are aggregates)
    skip = {"accuracy", "macro avg", "weighted avg"}
    for key, vals in report.items():
        if key in skip:
            continue
        if isinstance(vals, dict) and "precision" in vals:
            lines.append(
                f"{key:>12}  precision={vals['precision']:.3f}  recall={vals['recall']:.3f}  "
                f"f1={vals['f1-score']:.3f}  support={int(vals['support'])}"
            )
    lines.append("")
    for agg in ("macro avg", "weighted avg"):
        if agg in report and isinstance(report[agg], dict):
            v = report[agg]
            lines.append(
                f"{agg:>12}  precision={v['precision']:.3f}  recall={v['recall']:.3f}  "
                f"f1={v['f1-score']:.3f}  support={int(v['support'])}"
            )
    if "accuracy" in report:
        lines.append(f"{'accuracy':>12}  {report['accuracy']:.4f}")
    return "\n".join(lines) + "\n"


def _fmt_confusion_matrix(cm: np.ndarray, class_names: list[str]) -> str:
    """Pretty-print sklearn-style CM: rows = true, cols = pred."""
    cm = np.asarray(cm)
    rows: list[str] = []
    if cm.size == 0:
        return "(empty confusion matrix)\n"
    ncol = cm.shape[1]
    colnames = list(class_names)[:ncol]
    if len(colnames) < ncol:
        colnames.extend([str(i) for i in range(len(colnames), ncol)])

    tw = max(10, max(len(str(t)) for t in colnames), max(len(str(t)) for t in class_names) if class_names else 8)
    cell_w = max(6, len(str(int(cm.max()))) + 1)

    hdr = "".join(str(c)[:tw].rjust(cell_w) for c in colnames)
    rows.append("true \\ pred →" + hdr)
    for i in range(cm.shape[0]):
        tn = str(class_names[i]) if i < len(class_names) else str(i)
        row_vals = "".join(str(int(cm[i, j])).rjust(cell_w) for j in range(cm.shape[1]))
        rows.append(tn[:tw].ljust(max(14, tw + 4)) + row_vals)
    return "\n".join(rows) + "\n"


def _build_metrics_summary(
    *,
    stem: str,
    report: dict,
    class_names: list[str],
    eval_split_used: str | None,
    data_dir_effective: str | None,
    params_snapshot: dict,
    n_frames: int,
) -> dict:
    skip = {"accuracy", "macro avg", "weighted avg"}
    per_class: dict[str, dict[str, float]] = {}
    for k, v in report.items():
        if k in skip or not isinstance(v, dict) or "f1-score" not in v:
            continue
        per_class[str(k)] = {
            "precision": float(v["precision"]),
            "recall": float(v["recall"]),
            "f1": float(v["f1-score"]),
            "support": int(v["support"]),
        }
    out: dict = {
        "model_stem": stem,
        "eval_split_used": eval_split_used,
        "data_dir_effective": data_dir_effective,
        "n_eval_frames": n_frames,
        "nb_hist": params_snapshot.get("nb_hist"),
        "batch_size_eval": params_snapshot.get("batch_size"),
        "samplerate_x_Hz": params_snapshot.get("samplerate_x_Hz"),
        "class_names": list(class_names),
        "accuracy": float(report["accuracy"]) if "accuracy" in report else None,
        "macro_avg": (
            {kk: float(vv) for kk, vv in report["macro avg"].items()} if "macro avg" in report else None
        ),
        "weighted_avg": (
            {kk: float(vv) for kk, vv in report["weighted avg"].items()} if "weighted avg" in report else None
        ),
        "per_class": per_class,
        "note": "Frame-level labels from probabilities (same pipeline as train val). Rows=true, cols=pred in CSV.",
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a DAS run; save CM + report under deepLearning/.")
    parser.add_argument(
        "--stem",
        type=str,
        required=True,
        help="Prefix path (no _model.h5): same stem as *_model.h5 and *_params.yaml.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=f"Directory for outputs (default: {DEFAULT_OUT_PARENT}/<stem_basename>).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override params data_dir (same layout as DAS train: directory ending in .npy with train/val/test).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="auto",
        choices=("auto", "test", "val", "train"),
        help="Which data split to evaluate (default auto: test→val→train until len(x)>=nb_hist).",
    )
    parser.add_argument("--verbose", type=int, default=1, help="verbosity for DAS prediction (0/1).")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print confusion matrix / report to stdout (files are still written).",
    )
    parser.add_argument(
        "--strip-noise-pred",
        action="store_true",
        help=(
            "Never assign predicted class 'noise': argmax is taken over syllable dims only "
            "(noise logit column forced to -inf). True labels unchanged; confusion matrix noise *column* near zero."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    stem = str(Path(args.stem).expanduser().resolve())
    tag = _run_tag(stem)
    out_dir = args.out_dir if args.out_dir is not None else (DEFAULT_OUT_PARENT / tag)
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    from das import evaluate as ev
    import flammkuchen as fl

    data_dir_arg = str(args.data_dir.expanduser().resolve()) if args.data_dir is not None else None

    logger.info("Evaluating stem=%s", stem)
    conf_mat, report, results = ev.evaluate(
        stem,
        full_output=True,
        verbose=args.verbose,
        custom_objects=None,
        data_dir=data_dir_arg,
        split=args.split,
        strip_noise_from_pred=args.strip_noise_pred,
    )

    if conf_mat is None or report is None:
        logger.warning(
            "No evaluation (no split with len(x) >= nb_hist, or missing x). Nothing written."
        )
        return

    class_names = list(results["params"].get("class_names", []))
    n_frames = int(np.asarray(results["labels_test"]).size)
    params_snap = dict(results["params"])
    params_snap.pop("data_splits", None)

    stem_meta = {
        "model_stem": stem,
        "repo_das_src": str(_DAS_SRC),
        "out_dir": str(out_dir),
        "eval_split_requested": args.split,
        "eval_split_used": results.get("eval_split"),
        "strip_noise_from_pred": bool(args.strip_noise_pred),
        "noise_class_index_applied": results.get("noise_class_index"),
        "data_dir_override": data_dir_arg,
        "data_dir_effective": results.get("params", {}).get("data_dir") if results else None,
        "nb_hist": results["params"].get("nb_hist"),
        "samplerate_x_Hz": results["params"].get("samplerate_x_Hz"),
        "class_names": class_names,
        "n_eval_frames": n_frames,
    }
    meta_path = out_dir / "eval_meta.json"
    meta_path.write_text(json.dumps(stem_meta, indent=2, default=str), encoding="utf-8")

    h5_out = out_dir / f"{tag}_results.h5"
    fl.save(str(h5_out), results)

    cm_arr = np.asarray(conf_mat)
    cm_path = out_dir / f"{tag}_confusion_matrix.csv"
    _save_confusion_csv(cm_path, cm_arr, class_names)

    report_json = out_dir / f"{tag}_classification_report.json"

    def _json_default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(repr(o))

    report_json.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")

    txt_path = out_dir / f"{tag}_classification_report.txt"
    txt_path.write_text(_report_dict_to_readable_text(report), encoding="utf-8")

    summary = _build_metrics_summary(
        stem=stem,
        report=report,
        class_names=class_names,
        eval_split_used=results.get("eval_split"),
        data_dir_effective=stem_meta.get("data_dir_effective"),
        params_snapshot=params_snap,
        n_frames=n_frames,
    )
    summary["strip_noise_from_pred"] = bool(results.get("strip_noise_from_pred", args.strip_noise_pred))
    summary["noise_class_index"] = results.get("noise_class_index")

    summary_path = out_dir / f"{tag}_metrics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    strip_note = ""
    if summary.get("strip_noise_from_pred"):
        strip_note = (
            f"strip_noise_from_pred=True (noise class index {summary.get('noise_class_index')})\n"
        )
    summary_txt = (
        "Frame-level evaluation (matches training AudioSequence / label conversion)\n"
        f"Stem: {stem}\nSplit: {summary.get('eval_split_used')}  |  Frames: {n_frames}\n"
        f"{strip_note}"
        f"nb_hist={summary.get('nb_hist')} samplerate_x_Hz={summary.get('samplerate_x_Hz')}\n"
        f"data_dir: {summary.get('data_dir_effective')}\n\n"
        "Confusion matrix (rows=true, cols=pred)\n"
        + _fmt_confusion_matrix(cm_arr, class_names)
        + "\nClassification report\n"
        + _report_dict_to_readable_text(report)
    )
    summary_txt_path.write_text(summary_txt, encoding="utf-8")

    if not args.quiet:
        print()
        print("=" * 72)
        print("FRAME-LEVEL METRICS (from model + YAML + *.npy; not CSV raster comparison)")
        print("=" * 72)
        print(summary_txt.strip())
        print("=" * 72)
        print()

    logger.info("Wrote:")
    logger.info("  %s", h5_out)
    logger.info("  %s", cm_path)
    logger.info("  %s", report_json)
    logger.info("  %s", txt_path)
    logger.info("  %s", summary_path)
    logger.info("  %s", summary_txt_path)
    logger.info("  %s", meta_path)


if __name__ == "__main__":
    main()
