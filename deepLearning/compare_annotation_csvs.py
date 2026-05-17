#!/usr/bin/env python3
"""
Compare two syllable-level segment CSVs (same layout as ``*_annotations.csv``).

1) **Segment IoU** (default): greedy matching per syllable class → TP/FP/FN → P/R/F1
   (good for counting how many syllables were detected).

2) **Frame raster** (``--frames``): discretize time into short steps, assign a label per frame
   (syllable covering that time, or silence), then build a **confusion matrix** and
   frame-level accuracy / per-class P/R/F1 (closer to classic classification metrics).

**Paths**: ``--truth`` and ``--pred`` must both be files, or both be directories containing
matching filenames ``*_annotations.csv`` (same basename in each folder).

Examples (repo root)::

    # Single pair
    python deepLearning/compare_annotation_csvs.py \\
        --truth Wav/.../rec_annotations.csv \\
        --pred /path/pred_annotations.csv \\
        --iou 0.25 --frames --step-ms 10

    # Two folders of CSVs (matched by filename)
    python deepLearning/compare_annotation_csvs.py \\
        --truth Wav/manual_annot_dir/ \\
        --pred preds/out_dir/ \\
        --frames --export-dir ./eval_batch/

    # Export matrix + sklearn-style report text
    python deepLearning/compare_annotation_csvs.py --truth ... --pred ... --frames \\
        --export-dir ./eval_out/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

# Token for frames where no syllable interval covers the time bin.
_SIL = "__SIL__"


def _load_intervals_csv(path: Path) -> dict[str, list[tuple[float, float]]]:
    out: dict[str, list[tuple[float, float]]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: empty CSV")
        canon = {h.strip(): h for h in reader.fieldnames}
        missing = {"name", "start_seconds", "stop_seconds"} - canon.keys()
        if missing:
            raise ValueError(f"{path}: need columns name,start_seconds,stop_seconds; missing {missing}")
        k_name, k_s, k_e = canon["name"], canon["start_seconds"], canon["stop_seconds"]

        for row in reader:
            name = str(row.get(k_name, "")).strip()
            try:
                s = float(row[k_s])
                e = float(row[k_e])
            except (KeyError, TypeError, ValueError):
                continue
            if name == "" or name.lower() == "nan":
                continue
            out.setdefault(name, []).append((s, e))
    return out


def segment_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    s1, e1 = a
    s2, e2 = b
    inter = max(0.0, min(e1, e2) - max(s1, s2))
    if inter <= 0.0:
        return 0.0
    u = (e1 - s1) + (e2 - s2) - inter
    return inter / u if u > 0 else 0.0


def _greedy_match(
    truth: list[tuple[float, float]],
    pred: list[tuple[float, float]],
    *,
    min_iou: float,
):
    pairs: list[tuple[float, int, int]] = []
    for i, ti in enumerate(truth):
        for j, pj in enumerate(pred):
            iou = segment_iou(ti, pj)
            if iou >= min_iou:
                pairs.append((iou, i, j))
    pairs.sort(key=lambda t: -t[0])

    used_t = set()
    used_p = set()
    tp = 0
    for _, i, j in pairs:
        if i in used_t or j in used_p:
            continue
        used_t.add(i)
        used_p.add(j)
        tp += 1

    fn = len(truth) - tp
    fp = len(pred) - tp
    return tp, fp, fn


def _flatten_intervals(seg_map: dict[str, list[tuple[float, float]]]) -> list[tuple[float, float, str]]:
    rows: list[tuple[float, float, str]] = []
    for name, segs in seg_map.items():
        for s, e in segs:
            rows.append((float(s), float(e), name))
    rows.sort(key=lambda r: r[0])
    return rows


def _label_at_sample(t: float, flat: list[tuple[float, float, str]]) -> str:
    """One label per instant: syllable with latest start among those covering ``t`` (half-open [s,e))."""
    best: str | None = None
    best_s = -math.inf
    for s, e, name in flat:
        if s <= t < e and s >= best_s:
            best_s = s
            best = name
    return best if best is not None else _SIL


def _frame_label_sequences(
    t_map: dict[str, list[tuple[float, float]]],
    p_map: dict[str, list[tuple[float, float]]],
    *,
    step_s: float,
) -> tuple[list[str], list[str], float]:
    flat_t = _flatten_intervals(t_map)
    flat_p = _flatten_intervals(p_map)
    t_max = max((e for _s, e, _n in flat_t + flat_p), default=0.0)
    if t_max <= 0.0:
        t_max = 1.0

    max_frames = 2_000_000
    n_estimate = int(math.ceil(t_max / step_s))
    if n_estimate > max_frames:
        raise ValueError(
            f"Too many frames (~{n_estimate}) for step_s={step_s}; increase --step-ms (current step ~{step_s * 1000:.3g} ms)."
        )

    y_t: list[str] = []
    y_p: list[str] = []
    t = 0.5 * step_s  # bin center-ish: sample middle of each step
    while t < t_max:
        y_t.append(_label_at_sample(t, flat_t))
        y_p.append(_label_at_sample(t, flat_p))
        t += step_s
    return y_t, y_p, t_max


def _confusion_counts(y_true: list[str], y_pred: list[str]) -> tuple[list[str], list[list[int]]]:
    classes = sorted(set(y_true) | set(y_pred))
    ci = {c: i for i, c in enumerate(classes)}
    n = len(classes)
    cm = [[0 for _ in range(n)] for _ in range(n)]
    for yt, yp in zip(y_true, y_pred):
        cm[ci[yt]][ci[yp]] += 1
    return classes, cm


def _metrics_from_cm(classes: list[str], cm: list[list[int]]) -> dict:
    """Per-class precision/recall/F1 + micro accuracy over all frames."""
    n = len(classes)
    totals_p = [sum(cm[i][j] for i in range(n)) for j in range(n)]  # pred col
    totals_t = [sum(cm[i][j] for j in range(n)) for i in range(n)]  # true row
    tot = sum(sum(row) for row in cm)

    per: dict[str, dict[str, float]] = {}
    for i, c in enumerate(classes):
        tp = cm[i][i]
        tp_fp = totals_p[i]
        tp_fn = totals_t[i]
        prec = tp / tp_fp if tp_fp else (1.0 if tp_fn == 0 else 0.0)
        rec = tp / tp_fn if tp_fn else 1.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per[c] = {"precision": prec, "recall": rec, "f1": f1, "support": float(tp_fn)}

    correct = sum(cm[i][i] for i in range(n))
    accuracy = correct / tot if tot else 0.0

    # macro F1 over syllables only (exclude silence token)
    non_sil = [c for c in classes if c != _SIL]
    if non_sil:
        macro_f1 = sum(per[c]["f1"] for c in non_sil) / len(non_sil)
    else:
        macro_f1 = 0.0

    return {"accuracy": accuracy, "micro_correct": correct, "n_frames": tot, "per_class": per, "macro_f1_sylla": macro_f1}


def _save_confusion_csv(path: Path, classes: list[str], cm: list[list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + classes)
        for i, rn in enumerate(classes):
            w.writerow([rn] + [str(int(cm[i][j])) for j in range(len(classes))])


def _gather_dir_pairs(truth_dir: Path, pred_dir: Path) -> tuple[list[tuple[Path, Path]], list[str], list[str]]:
    def index_folder(d: Path) -> dict[str, Path]:
        out: dict[str, Path] = {}
        for p in sorted(d.glob("*_annotations.csv")):
            if p.is_file():
                out[p.name] = p.resolve()
        return out

    ti = index_folder(truth_dir)
    pi = index_folder(pred_dir)
    common = sorted(set(ti.keys()) & set(pi.keys()))
    only_t = sorted(set(ti.keys()) - set(pi.keys()))
    only_p = sorted(set(pi.keys()) - set(ti.keys()))
    return [(ti[n], pi[n]) for n in common], only_t, only_p


def _merge_confusion_counts(blocks: list[dict]) -> tuple[list[str], list[list[int]], dict]:
    """Sum confusion counts over files into one global CM (union of labels)."""
    all_labels: set[str] = set()
    for b in blocks:
        for lab in b["confusion_matrix"]["labels"]:
            all_labels.add(lab)
    classes = sorted(all_labels)
    ci = {c: i for i, c in enumerate(classes)}
    n = len(classes)
    merged = [[0 for _ in range(n)] for _ in range(n)]
    for b in blocks:
        loc_labels = b["confusion_matrix"]["labels"]
        loc_cm = b["confusion_matrix"]["counts"]
        li = {lbl: idx for idx, lbl in enumerate(loc_labels)}
        for i_li, yt in enumerate(loc_labels):
            gi = ci[yt]
            for j_li, pj in enumerate(loc_labels):
                gj = ci[pj]
                merged[gi][gj] += int(loc_cm[i_li][j_li])
    metrics = _metrics_from_cm(classes, merged)
    return classes, merged, metrics


def _aggregate_segment_detection(results: list[dict]) -> dict:
    sum_tp = sum_fp = sum_fn = 0
    agg_counts: dict[str, dict[str, float]] = {}
    for out in results:
        mic = out["segment_iou_detection"]["micro"]
        sum_tp += int(mic["tp"])
        sum_fp += int(mic["fp"])
        sum_fn += int(mic["fn"])
        for lab, stat in out["segment_iou_detection"]["per_class"].items():
            ac = agg_counts.setdefault(lab, {"tp": 0.0, "fp": 0.0, "fn": 0.0})
            ac["tp"] += stat["tp"]
            ac["fp"] += stat["fp"]
            ac["fn"] += stat["fn"]
    micro_p = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) else 0.0
    micro_r = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0.0
    per: dict[str, dict[str, float]] = {}
    for lab in sorted(agg_counts.keys()):
        tp, fp, fn = agg_counts[lab]["tp"], agg_counts[lab]["fp"], agg_counts[lab]["fn"]
        prec = tp / (tp + fp) if (tp + fp) else (1.0 if fn == 0 else 0.0)
        rec = tp / (tp + fn) if (tp + fn) else 1.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per[lab] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        }
    return {
        "micro": {
            "tp": sum_tp,
            "fp": sum_fp,
            "fn": sum_fn,
            "precision": micro_p,
            "recall": micro_r,
            "f1": micro_f1,
        },
        "per_class": per,
    }


def compare_one_pair(
    truth_path: Path,
    pred_path: Path,
    *,
    iou: float,
    frames: bool,
    step_ms: float,
    export_parent: Path | None,
    export_base_stem: str,
) -> dict:
    truth_path = truth_path.expanduser().resolve()
    pred_path = pred_path.expanduser().resolve()
    t_map = _load_intervals_csv(truth_path)
    p_map = _load_intervals_csv(pred_path)
    labels = sorted(set(t_map.keys()) | set(p_map.keys()))

    per: dict[str, dict[str, float]] = {}
    sum_tp = sum_fp = sum_fn = 0

    for lab in labels:
        tr = sorted(t_map.get(lab, []), key=lambda x: x[0])
        pr = sorted(p_map.get(lab, []), key=lambda x: x[0])
        tp, fp, fn = _greedy_match(tr, pr, min_iou=iou)
        sum_tp += tp
        sum_fp += fp
        sum_fn += fn
        prec = tp / (tp + fp) if (tp + fp) else (1.0 if fn == 0 else 0.0)
        rec = tp / (tp + fn) if (tp + fn) else 1.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per[lab] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        }

    micro_p = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) else 0.0
    micro_r = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0.0

    out: dict = {
        "truth_csv": str(truth_path),
        "pred_csv": str(pred_path),
        "iou_threshold": iou,
        "segment_iou_detection": {
            "micro": {
                "tp": sum_tp,
                "fp": sum_fp,
                "fn": sum_fn,
                "precision": micro_p,
                "recall": micro_r,
                "f1": micro_f1,
            },
            "per_class": per,
        },
    }

    frame_block: dict | None = None
    y_t: list[str] = []
    y_p: list[str] = []
    classes: list[str] = []

    if frames:
        step_s = max(step_ms / 1000.0, 1e-9)
        y_t, y_p, t_max = _frame_label_sequences(t_map, p_map, step_s=step_s)
        classes, cm = _confusion_counts(y_t, y_p)
        m = _metrics_from_cm(classes, cm)
        frame_block = {
            "step_ms": step_ms,
            "duration_s_approx": t_max,
            "n_frames": m["n_frames"],
            "accuracy": m["accuracy"],
            "macro_f1_syllables_excl_silence": m["macro_f1_sylla"],
            "confusion_matrix": {"labels": classes, "counts": cm},
            "per_class": m["per_class"],
        }
        out["frame_raster"] = frame_block

        if export_parent is not None:
            ex = export_parent.expanduser().resolve()
            ex.mkdir(parents=True, exist_ok=True)
            stem = export_base_stem
            cm_path = ex / f"{stem}_frame_confusion_matrix.csv"
            _save_confusion_csv(cm_path, classes, cm)
            (ex / f"{stem}_frame_metrics.json").write_text(json.dumps(frame_block, indent=2), encoding="utf-8")
            try:
                import sklearn.metrics as skm

                rep = skm.classification_report(
                    y_t,
                    y_p,
                    labels=classes,
                    digits=3,
                    zero_division=0,
                )
                (ex / f"{stem}_classification_report.txt").write_text(rep, encoding="utf-8")
            except ImportError:
                pass

    return out


def _print_human_one(out: dict, frames: bool, step_ms: float) -> None:
    print(f"truth: {out['truth_csv']}")
    print(f"pred : {out['pred_csv']}")
    seg = out["segment_iou_detection"]
    mic = seg["micro"]
    iou_th = float(out["iou_threshold"])
    print(f"\n=== Segment IoU (IoU >= {iou_th}) ===")
    print(
        f"Micro: P={mic['precision']:.3f} R={mic['recall']:.3f} F1={mic['f1']:.3f}  "
        f"(TP={int(mic['tp'])} FP={int(mic['fp'])} FN={int(mic['fn'])})\n"
    )
    for lab in sorted(seg["per_class"].keys()):
        r = seg["per_class"][lab]
        print(
            f"  [{lab}] P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} "
            f"(tp={int(r['tp'])} fp={int(r['fp'])} fn={int(r['fn'])})"
        )

    if not frames or "frame_raster" not in out:
        return
    frame_block = out["frame_raster"]
    print(f"\n=== Frame raster (step={step_ms} ms) ===")
    print(f"accuracy={frame_block['accuracy']:.4f}  n_frames={int(frame_block['n_frames'])}")
    print(f"macro-F1 (syllables, excl. {_SIL})={frame_block['macro_f1_syllables_excl_silence']:.4f}\n")

    lbls = frame_block["confusion_matrix"]["labels"]
    mat = frame_block["confusion_matrix"]["counts"]
    if len(lbls) <= 16:
        w = max(len(str(c)) for c in lbls) if lbls else 6
        head = "".join(str(c).rjust(w + 1) for c in lbls)
        print("pred->" + head)
        for i, rn in enumerate(lbls):
            row_s = "".join(str(int(mat[i][j])).rjust(w + 1) for j in range(len(lbls)))
            print(f"{rn!s:>{w}} |{row_s}")

    pc = frame_block["per_class"]
    print("\nPer-class (frames):")
    for c in lbls:
        r = pc[c]
        print(
            f"  [{c}] P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} support={int(r['support'])}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Segment IoU / frame metrics between annotation CSVs (single pair or two folders).",
    )
    parser.add_argument(
        "--truth",
        type=Path,
        required=True,
        help="Reference CSV or directory of *_annotations.csv files.",
    )
    parser.add_argument(
        "--pred",
        type=Path,
        required=True,
        help="Prediction CSV or directory of *_annotations.csv files (same basenames as truth when using dirs).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.25,
        help="Minimum IoU to count a TP for a syllable interval pair (default 0.25).",
    )
    parser.add_argument(
        "--frames",
        action="store_true",
        help="Also compute frame-level confusion matrix and accuracy / P/R/F1 (needs time-aligned CSVs).",
    )
    parser.add_argument(
        "--step-ms",
        type=float,
        default=10.0,
        help="Raster step in milliseconds when --frames is set (default 10).",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help="Write per-file frame artifacts; batch mode also writes batch_summary.json here.",
    )
    parser.add_argument("--json", action="store_true", help="Print one JSON object to stdout.")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="With batch dirs: only print summary (one line per file still skipped; use --json for details).",
    )
    args = parser.parse_args()

    truth_r = args.truth.expanduser().resolve()
    pred_r = args.pred.expanduser().resolve()

    if truth_r.is_dir() != pred_r.is_dir():
        parser.error("Either both --truth and --pred are files, or both are directories.")

    pairs: list[tuple[Path, Path]]
    if truth_r.is_dir():
        pairs, only_t, only_p = _gather_dir_pairs(truth_r, pred_r)
        for n in only_t:
            print(f"WARNING: only in truth dir (no pred match): {n}", file=sys.stderr)
        for n in only_p:
            print(f"WARNING: only in pred dir (no truth match): {n}", file=sys.stderr)
        if not pairs:
            parser.error(f"No matching *_annotations.csv pairs between {truth_r} and {pred_r}.")
    else:
        if not truth_r.is_file() or not pred_r.is_file():
            parser.error("Paths must be existing files or directories.")
        pairs = [(truth_r, pred_r)]

    results: list[dict] = []
    for tp, pp in pairs:
        export_base = tp.stem
        out = compare_one_pair(
            tp,
            pp,
            iou=args.iou,
            frames=args.frames,
            step_ms=args.step_ms,
            export_parent=args.export_dir,
            export_base_stem=export_base,
        )
        results.append(out)
        if len(pairs) == 1 and not args.json and not args.quiet:
            _print_human_one(out, args.frames, args.step_ms)
            if args.export_dir:
                print(f"\nExported under {args.export_dir.expanduser().resolve()}")

    if len(pairs) > 1:
        summary: dict = {
            "n_pairs": len(pairs),
            "iou_threshold": args.iou,
            "segment_iou_detection_aggregated": _aggregate_segment_detection(results),
        }
        if args.frames:
            frame_blocks = [r["frame_raster"] for r in results if "frame_raster" in r]
            if frame_blocks:
                glabels, gcm, gmet = _merge_confusion_counts(frame_blocks)
                summary["frame_raster_merged"] = {
                    "step_ms": args.step_ms,
                    "n_frames": gmet["n_frames"],
                    "accuracy": gmet["accuracy"],
                    "macro_f1_syllables_excl_silence": gmet["macro_f1_sylla"],
                    "confusion_matrix": {"labels": glabels, "counts": gcm},
                    "per_class": gmet["per_class"],
                }
        summary["per_file"] = results

        if args.export_dir is not None and args.frames and "frame_raster_merged" in summary:
            ex = args.export_dir.expanduser().resolve()
            ex.mkdir(parents=True, exist_ok=True)
            _save_confusion_csv(ex / "batch_frame_confusion_matrix.csv", glabels, gcm)
            (ex / "batch_frame_metrics.json").write_text(
                json.dumps(summary["frame_raster_merged"], indent=2),
                encoding="utf-8",
            )

        if args.export_dir is not None:
            (args.export_dir.expanduser().resolve() / "batch_summary.json").write_text(
                json.dumps(summary, indent=2, default=str),
                encoding="utf-8",
            )

        if args.json:
            print(json.dumps(summary, indent=2, default=str))
        elif not args.quiet:
            agg = summary["segment_iou_detection_aggregated"]
            mic = agg["micro"]
            print(f"\n=== Batch: {len(pairs)} file pairs (matched by filename) ===")
            print(f"Aggregated segment IoU: P={mic['precision']:.3f} R={mic['recall']:.3f} F1={mic['f1']:.3f}")
            if args.frames and "frame_raster_merged" in summary:
                fr = summary["frame_raster_merged"]
                print(
                    f"Merged frame raster (sum of CMs): acc={fr['accuracy']:.4f} "
                    f"n_frames={int(fr['n_frames'])} macro-F1(syl)={fr['macro_f1_syllables_excl_silence']:.4f}"
                )
            if args.export_dir:
                print(f"Exported per-file + batch summary under {args.export_dir.expanduser().resolve()}")


if __name__ == "__main__":
    main()
