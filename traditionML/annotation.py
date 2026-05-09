#!/usr/bin/env python3
"""
Align manual annotation CSV (time ranges + syllable names) with segmentation CSV,
produce per-segment labels for MFCC/training and a rich *_segments_info.csv.

Manual files: *{core}_annotations.csv  (e.g. "a b c d e f i 246_05_annotations.csv")
Segment files: {core}_segments.csv

Outputs:
  - <output_txt_dir>/{core}.txt  — one lowercase label per segment row
  - <segments_dir>/{core}_segments_info.csv — segment columns + label + remark
  - warnings appended to log file for unmatched / odd cases
  - <segments_dir>/annotation_alignment_stats.csv — per-recording & total missed-syllable counts vs manual total
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
from pathlib import Path
from typing import Any


DEFAULT_TIME_TOL = 0.2
# One segmentation segment covering multiple manual syllables (merged boundary error)
MULTI_OVERLAP_MIN_SEC = 0.02
MULTI_OVERLAP_MIN_FRAC = 0.12


def segments_core_from_filename(seg_path: Path) -> str | None:
    """246_05_segments.csv -> 246_05"""
    stem = seg_path.stem
    if stem.endswith("_segments"):
        return stem[: -len("_segments")]
    return None


def find_annotation_csv(annotated_dir: Path, core: str) -> Path | None:
    """Match *{core}_annotations.csv in annotated_dir."""
    pattern = f"{re.escape(core)}_annotations.csv"
    candidates = list(annotated_dir.glob(f"*{core}_annotations.csv"))
    # Prefer filename ending exactly with core_annotations.csv
    exact_suffix = f"{core}_annotations.csv"
    for p in candidates:
        if p.name.endswith(exact_suffix) or p.name == exact_suffix:
            return p
    return candidates[0] if len(candidates) == 1 else None


def load_manual_annotations(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return rows
        name_col = next(
            (c for c in reader.fieldnames if c.strip().lower() in ("name", "label", "syllable")),
            reader.fieldnames[0],
        )
        start_col = next(
            (
                c
                for c in reader.fieldnames
                if c.strip().lower() in ("start_seconds", "start", "t_start", "t_start_s")
            ),
            None,
        )
        stop_col = next(
            (
                c
                for c in reader.fieldnames
                if c.strip().lower()
                in ("stop_seconds", "stop", "end_seconds", "t_end", "t_end_s")
            ),
            None,
        )
        if start_col is None or stop_col is None:
            raise ValueError(
                f"{path}: need start/stop columns (start_seconds, stop_seconds, ...)"
            )

        for row in reader:
            name = (row.get(name_col) or "").strip()
            if not name:
                continue
            try:
                ts = float(row[start_col])
                te = float(row[stop_col])
            except (TypeError, ValueError):
                continue
            rows.append({"name": name, "start": ts, "stop": te})
    rows.sort(key=lambda r: r["start"])
    return rows


def load_segments_csv(path: Path) -> list[dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            try:
                sid = row.get("unit_id", "").strip()
                t0 = float(row["t_start_s"])
                t1 = float(row["t_end_s"])
            except (KeyError, ValueError):
                continue
            rec = {
                "unit_id": sid if sid != "" else str(len(out)),
                "t_start_s": t0,
                "t_end_s": t1,
                "duration_s": row.get("duration_s", ""),
                "f_low_hz": row.get("f_low_hz", ""),
                "f_high_hz": row.get("f_high_hz", ""),
            }
            out.append(rec)
    return out


def _swallowed_tail_remark(
    i: int,
    ts: float,
    te: float,
    manuals: list[dict[str, Any]],
    tol: float,
) -> str:
    """
    After matching manual index i: following syllables that overlap (ts, te] beyond
    manual i's end (segment ate into / swallowed them). Requires next syllable to
    start before segment ends (not after te).
    """
    swallowed: list[str] = []
    j = i + 1
    while j < len(manuals):
        m = manuals[j]
        if m["start"] >= te - 1e-9:
            break
        ov_end = min(te, m["stop"])
        ov_start = max(ts, m["start"], manuals[i]["stop"] - tol)
        if ov_end <= ov_start + 1e-9:
            j += 1
            continue
        if m["stop"] <= te + tol:
            swallowed.append(m["name"].strip().lower())
            j += 1
            continue
        if m["start"] < te < m["stop"]:
            swallowed.append(m["name"].strip().lower())
            break
        break
    if len(swallowed) == 1:
        return f"less {swallowed[0]}"
    if len(swallowed) > 1:
        return "miss " + " ".join(swallowed)
    return ""


def significant_manual_overlap_indices(
    ts: float,
    te: float,
    manuals: list[dict[str, Any]],
) -> list[int]:
    """
    Manual rows whose interval has substantial overlap with [ts, te].
    Used to detect one segment swallowing multiple syllables (e.g. d+e merged).
    """
    idxs: list[int] = []
    for i, m in enumerate(manuals):
        ov_s = max(ts, m["start"])
        ov_e = min(te, m["stop"])
        ov = ov_e - ov_s
        if ov <= 1e-9:
            continue
        dur_m = m["stop"] - m["start"]
        if dur_m <= 1e-9:
            continue
        frac = ov / dur_m
        if ov >= MULTI_OVERLAP_MIN_SEC and frac >= MULTI_OVERLAP_MIN_FRAC:
            idxs.append(i)
    return idxs


def segment_overlaps_manual(ts: float, te: float, ms: float, me: float) -> bool:
    return min(te, me) - max(ts, ms) > 1e-9


def count_missed_syllables_from_remark(remark: str) -> int:
    """Remark 'miss a b' / 'less x' counts syllables not assigned their own labeled row."""
    r = (remark or "").strip().lower()
    if r.startswith("miss "):
        return len(r[5:].split())
    if r.startswith("less "):
        return 1
    return 0


def compute_pair_alignment_stats(
    manuals: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    labels_and_remarks: list[tuple[str, str]],
) -> dict[str, float | int]:
    """
    missed_merge_remark: syllables folded into another segment (remark miss/less).
    missed_empty_overlap: manual syllables overlapping a segment that got no label.
    missed_orphan_no_segment: manual intervals not overlapping any segment at all.
    """
    total_manual = len(manuals)
    missed_merge = sum(
        count_missed_syllables_from_remark(rm) for _lab, rm in labels_and_remarks
    )

    missed_empty_overlap = 0
    for seg, (lab, _rm) in zip(segments, labels_and_remarks):
        if lab.strip():
            continue
        ts = float(seg["t_start_s"])
        te = float(seg["t_end_s"])
        for m in manuals:
            if segment_overlaps_manual(ts, te, m["start"], m["stop"]):
                missed_empty_overlap += 1

    missed_orphan = 0
    for m in manuals:
        ms, me = m["start"], m["stop"]
        if not any(
            segment_overlaps_manual(float(s["t_start_s"]), float(s["t_end_s"]), ms, me)
            for s in segments
        ):
            missed_orphan += 1

    missed_total = missed_merge + missed_empty_overlap + missed_orphan
    ratio = (missed_total / total_manual) if total_manual else 0.0
    return {
        "total_manual_syllables": total_manual,
        "missed_merge_remark": missed_merge,
        "missed_empty_segment_overlap": missed_empty_overlap,
        "missed_orphan_no_segment": missed_orphan,
        "missed_total": missed_total,
        "missed_ratio": ratio,
    }


def assign_segment_label(
    ts: float,
    te: float,
    manuals: list[dict[str, Any]],
    tol: float,
) -> tuple[str, str, list[str]]:
    """
    Returns (label, remark, warnings) — label/remark lowercase where applicable.
    """
    warns: list[str] = []
    if not manuals:
        return "", "", ["no manual rows"]

    # Prefer chronological alignment when one segment clearly covers ≥2 manual syllables
    # (avoids picking the last syllable by tight endpoint match only).
    sig_idxs = significant_manual_overlap_indices(ts, te, manuals)
    if len(sig_idxs) >= 2:
        ordered = sorted(sig_idxs, key=lambda i: manuals[i]["start"])
        first_i = ordered[0]
        lab = manuals[first_i]["name"].strip().lower()
        rest_names = [manuals[j]["name"].strip().lower() for j in ordered[1:]]
        remark = "miss " + " ".join(rest_names)
        warns.append(
            "segment_merged_multiple_manual_syllables: "
            f"[{ts:.4f},{te:.4f}] -> label={lab!r}, {remark!r}"
        )
        return lab, remark, warns

    best_i = min(
        range(len(manuals)),
        key=lambda i: max(abs(ts - manuals[i]["start"]), abs(te - manuals[i]["stop"])),
    )
    best_d = max(
        abs(ts - manuals[best_i]["start"]),
        abs(te - manuals[best_i]["stop"]),
    )

    if best_d <= tol:
        lab = manuals[best_i]["name"].strip().lower()
        remark = _swallowed_tail_remark(best_i, ts, te, manuals, tol)
        return lab, remark, warns

    best_s = min(range(len(manuals)), key=lambda i: abs(ts - manuals[i]["start"]))
    if abs(ts - manuals[best_s]["start"]) <= tol:
        lab = manuals[best_s]["name"].strip().lower()
        remark = _swallowed_tail_remark(best_s, ts, te, manuals, tol)
        return lab, remark, warns

    overlap_idxs = [
        i
        for i, m in enumerate(manuals)
        if min(te, m["stop"]) - max(ts, m["start"]) > -1e-9
    ]
    if overlap_idxs:
        ordered = sorted(overlap_idxs, key=lambda i: manuals[i]["start"])
        first_i = ordered[0]
        lab = manuals[first_i]["name"].strip().lower()
        rest_i = [i for i in ordered if i != first_i]
        if rest_i:
            rest_names = [manuals[i]["name"].strip().lower() for i in rest_i]
            if len(rest_names) == 1:
                remark = f"less {rest_names[0]}"
            else:
                remark = "miss " + " ".join(rest_names)
            return lab, remark, warns
        remark = _swallowed_tail_remark(first_i, ts, te, manuals, tol)
        return lab, remark, warns

    warns.append(
        f"no match: segment [{ts:.6f},{te:.6f}] vs manuals "
        f"[{manuals[0]['start']:.3f},{manuals[-1]['stop']:.3f}]"
    )
    return "", "", warns


def process_pair(
    seg_csv: Path,
    ann_csv: Path,
    out_txt_dir: Path,
    segments_dir: Path,
    tol: float,
    log: logging.Logger,
) -> dict[str, float | int] | None:
    manuals = load_manual_annotations(ann_csv)
    segments = load_segments_csv(seg_csv)
    core = segments_core_from_filename(seg_csv)
    if core is None:
        log.warning("Skip bad segment filename: %s", seg_csv)
        return None

    labels_lines: list[str] = []
    info_rows: list[list[Any]] = []
    labels_and_remarks: list[tuple[str, str]] = []

    for row in segments:
        ts = float(row["t_start_s"])
        te = float(row["t_end_s"])
        lab, remark, warns = assign_segment_label(ts, te, manuals, tol)
        for w in warns:
            log.warning("%s | unit ~%s | %s", core, row.get("unit_id"), w)
        if not lab:
            log.warning(
                "%s | empty label for segment unit_id=%s [%s,%s]",
                core,
                row.get("unit_id"),
                ts,
                te,
            )

        labels_lines.append(lab if lab else "")
        labels_and_remarks.append((lab, remark))

        info_rows.append(
            [
                row["unit_id"],
                f"{ts:.6f}",
                f"{te:.6f}",
                row.get("duration_s", f"{te - ts:.6f}"),
                row.get("f_low_hz", ""),
                row.get("f_high_hz", ""),
                lab,
                remark,
            ]
        )

    out_txt_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_txt_dir / f"{core}.txt"
    txt_path.write_text("\n".join(labels_lines) + ("\n" if labels_lines else ""), encoding="utf-8")

    info_path = segments_dir / f"{core}_segments_info.csv"
    with open(info_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "unit_id",
                "t_start_s",
                "t_end_s",
                "duration_s",
                "f_low_hz",
                "f_high_hz",
                "label",
                "remark",
            ]
        )
        w.writerows(info_rows)

    stats = compute_pair_alignment_stats(manuals, segments, labels_and_remarks)
    log.info(
        "OK %s: %d segments -> %s , %s",
        core,
        len(segments),
        txt_path,
        info_path,
    )
    log.info(
        "STATS %s: manual_syllables=%d missed_total=%d "
        "(merge_remark=%d empty_seg_overlap=%d orphan=%d) missed_ratio=%.4f",
        core,
        stats["total_manual_syllables"],
        stats["missed_total"],
        stats["missed_merge_remark"],
        stats["missed_empty_segment_overlap"],
        stats["missed_orphan_no_segment"],
        stats["missed_ratio"],
    )
    out_stats = dict(stats)
    out_stats["core"] = core
    return out_stats


def setup_logging(log_path: Path, *, file_append: bool = False) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("annotation_align")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(
        log_path, encoding="utf-8", mode="a" if file_append else "w"
    )
    fh.setLevel(logging.WARNING)
    fh.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def main() -> None:
    root = Path("/home/yuxuan.li/workspace/Birdsong-project/Wav/finch")
    parser = argparse.ArgumentParser(description="Align manual annotations with segment CSVs.")
    parser.add_argument(
        "--annotated_dir",
        type=Path,
        default=root / "246_O_annotated_excel",
        help="Folder with *_annotations.csv",
    )
    parser.add_argument(
        "--segments_dir",
        type=Path,
        default=root / "246_O_seg",
        help="Folder with *_segments.csv",
    )
    parser.add_argument(
        "--output_txt_dir",
        type=Path,
        default=root / "246_O_anno_y",
        help="Write {core}.txt per recording",
    )
    parser.add_argument(
        "--time_tol",
        type=float,
        default=DEFAULT_TIME_TOL,
        help="Max |Δstart| and |Δend| for tight match (seconds)",
    )
    parser.add_argument(
        "--log_file",
        type=Path,
        default=None,
        help="Warning log. Default: <segments_dir>/annotation_align_warnings.log",
    )
    parser.add_argument(
        "--append_log",
        action="store_true",
        help="Append to log file instead of overwriting each run.",
    )
    parser.add_argument(
        "--stats_csv",
        type=Path,
        default=None,
        help=(
            "Write per-recording alignment stats CSV. "
            "Default: <segments_dir>/annotation_alignment_stats.csv"
        ),
    )
    args = parser.parse_args()

    annotated_dir = Path(args.annotated_dir)
    segments_dir = Path(args.segments_dir)
    log_path = (
        Path(args.log_file)
        if args.log_file
        else segments_dir / "annotation_align_warnings.log"
    )
    log = setup_logging(log_path, file_append=args.append_log)

    if not annotated_dir.is_dir():
        raise SystemExit(f"Not a directory: {annotated_dir}")
    if not segments_dir.is_dir():
        raise SystemExit(f"Not a directory: {segments_dir}")

    seg_files = sorted(segments_dir.glob("*_segments.csv"))
    seg_files = [p for p in seg_files if not p.name.endswith("_segments_info.csv")]

    if not seg_files:
        raise SystemExit(f"No *_segments.csv in {segments_dir}")

    stats_csv_path = (
        Path(args.stats_csv)
        if args.stats_csv is not None
        else segments_dir / "annotation_alignment_stats.csv"
    )
    all_stats: list[dict[str, float | int | str]] = []

    for seg_csv in seg_files:
        core = segments_core_from_filename(seg_csv)
        if not core:
            continue
        ann = find_annotation_csv(annotated_dir, core)
        if ann is None:
            log.warning("No annotation file for core=%r (segment %s)", core, seg_csv.name)
            continue
        try:
            row = process_pair(
                seg_csv,
                ann,
                Path(args.output_txt_dir),
                segments_dir,
                args.time_tol,
                log,
            )
            if row is not None:
                all_stats.append(row)
        except Exception as e:
            log.warning("Failed %s: %s", seg_csv.name, e, exc_info=True)

    if all_stats:
        tot_m = sum(int(s["total_manual_syllables"]) for s in all_stats)
        tot_miss = sum(int(s["missed_total"]) for s in all_stats)
        mm = sum(int(s["missed_merge_remark"]) for s in all_stats)
        me = sum(int(s["missed_empty_segment_overlap"]) for s in all_stats)
        mo = sum(int(s["missed_orphan_no_segment"]) for s in all_stats)
        overall_ratio = (tot_miss / tot_m) if tot_m else 0.0
        log.info(
            "=== OVERALL (all paired recordings) === "
            "manual_syllables=%d missed_total=%d "
            "(merge_remark=%d empty_seg_overlap=%d orphan=%d) "
            "missed_ratio=%.6f",
            tot_m,
            tot_miss,
            mm,
            me,
            mo,
            overall_ratio,
        )

        stats_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_csv_path, "w", encoding="utf-8", newline="") as sf:
            w = csv.writer(sf)
            w.writerow(
                [
                    "core",
                    "total_manual_syllables",
                    "missed_merge_remark",
                    "missed_empty_segment_overlap",
                    "missed_orphan_no_segment",
                    "missed_total",
                    "missed_ratio",
                ]
            )
            for s in sorted(all_stats, key=lambda r: str(r["core"]).lower()):
                w.writerow(
                    [
                        s["core"],
                        s["total_manual_syllables"],
                        s["missed_merge_remark"],
                        s["missed_empty_segment_overlap"],
                        s["missed_orphan_no_segment"],
                        s["missed_total"],
                        f"{float(s['missed_ratio']):.6f}",
                    ]
                )
            w.writerow([])
            w.writerow(
                [
                    "_TOTAL_",
                    tot_m,
                    mm,
                    me,
                    mo,
                    tot_miss,
                    f"{overall_ratio:.6f}",
                ]
            )
        log.info("Wrote alignment stats: %s", stats_csv_path.resolve())


if __name__ == "__main__":
    main()
