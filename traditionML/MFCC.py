import argparse
import csv
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

import librosa
import numpy as np


def load_segments_from_txt(txt_path):
    segments = []

    # 允许以下形式（含可选的 "|" 分隔）：
    # Unit 0: 1.78s - 1.83s | 2412-2498 Hz
    # Unit 0: 1.78s - 1.83s 2412-2498 Hz
    line_pattern = re.compile(
        r"^Unit\s+\d+:\s*([\d.]+)s\s*-\s*([\d.]+)s\s*\|?\s*([\d.]+)\s*-\s*([\d.]+)\s*Hz\s*$"
    )

    with open(txt_path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            match = line_pattern.match(line)
            if match is None:
                # 跳过无法解析的行，避免单行异常导致流程中断
                continue

            t_start = float(match.group(1))
            t_end = float(match.group(2))
            f_low = float(match.group(3))
            f_high = float(match.group(4))

            # 过滤非法区间，确保后续处理稳定
            if t_end <= t_start or f_high <= f_low:
                continue

            segments.append({
                "t_start": t_start,
                "t_end": t_end,
                "f_low": f_low,
                "f_high": f_high
            })

    return segments


def load_segments_from_csv(csv_path):
    """
    Load segments from segmentation.py CSV export:
      unit_id, t_start_s, t_end_s, duration_s, f_low_hz, f_high_hz
    Flexible headers: also accepts t_start / f_low_hz style aliases.
    """
    segments = []
    path = Path(csv_path)
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return segments

        def col(row, *aliases):
            for a in aliases:
                if a in row and row[a] is not None and str(row[a]).strip():
                    return str(row[a]).strip()
            return None

        for row in reader:
            ts = col(row, "t_start_s", "t_start")
            te = col(row, "t_end_s", "t_end")
            fl = col(row, "f_low_hz", "f_low")
            fh = col(row, "f_high_hz", "f_high")
            if ts is None or te is None or fl is None or fh is None:
                continue

            try:
                t_start = float(ts)
                t_end = float(te)
                f_low = float(fl)
                f_high = float(fh)
            except ValueError:
                continue

            if t_end <= t_start or f_high <= f_low:
                continue

            segments.append({
                "t_start": t_start,
                "t_end": t_end,
                "f_low": f_low,
                "f_high": f_high,
            })

    return segments


def load_segments(segment_path):
    """Auto: .csv -> CSV; otherwise legacy Unit ... txt."""
    segment_path = Path(segment_path)
    suffix = segment_path.suffix.lower()
    if suffix == ".csv":
        return load_segments_from_csv(segment_path)
    return load_segments_from_txt(segment_path)


# librosa.feature.delta 默认 width=9；此处 cap 同值，帧数不足时降为 ≤帧数的最大奇数（≥3）
MFCC_DELTA_DEFAULT_WIDTH = 9
MFCC_DELTA_MIN_FRAMES = 3


def pick_n_fft_for_segment(n_samples: int, n_fft: int) -> int:
    """Use min(requested n_fft, segment length), rounded down to a power of two (no extra knobs)."""
    n = max(int(n_samples), 1)
    cap = min(int(n_fft), n)
    if cap < 2:
        return 2
    return int(2 ** int(np.floor(np.log2(cap))))


def mfcc_delta_filter_width(n_frames: int, cap: int = MFCC_DELTA_DEFAULT_WIDTH) -> int | None:
    """Largest odd width <= min(n_frames, cap), at least 3; None if n_frames < 3."""
    if n_frames < MFCC_DELTA_MIN_FRAMES:
        return None
    m = min(int(n_frames), int(cap))
    if m % 2 == 0:
        m -= 1
    if m < MFCC_DELTA_MIN_FRAMES:
        return None
    return m


def extract_segment_waveform(y, sr, segment):
    start_sample = int(segment["t_start"] * sr)
    end_sample   = int(segment["t_end"]   * sr)
    return y[start_sample:end_sample]

def extract_mfcc_features(
    y_seg,
    sr,
    n_mfcc=13,
    n_fft=760,
    hop_length=256,
    delta_width: int | None = None,
):
    """
    delta_width: 显式传给 librosa delta 的奇数宽度；默认 None 时按帧数自适应（最大 cap=MFCC_DELTA_DEFAULT_WIDTH）。
    帧数 < MFCC_DELTA_MIN_FRAMES 时返回 None。
    """
    n_fft_eff = pick_n_fft_for_segment(len(y_seg), n_fft)

    mfcc = librosa.feature.mfcc(
        y=y_seg,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft_eff,
        hop_length=hop_length
    )

    n_frames = mfcc.shape[1]
    if delta_width is None:
        width = mfcc_delta_filter_width(n_frames)
        if width is None:
            return None
    else:
        width = int(delta_width)
        if width % 2 == 0:
            width = max(width - 1, MFCC_DELTA_MIN_FRAMES)
        if n_frames < width:
            return None

    delta  = librosa.feature.delta(mfcc, width=width)
    delta2 = librosa.feature.delta(mfcc, order=2, width=width)

    # 统计汇总（传统做法）
    features = []

    for M in (mfcc, delta, delta2):
        features.extend(np.mean(M, axis=1))
        features.extend(np.std(M, axis=1))

    return np.array(features)

def extract_duration_bandwidth(segment):
    duration = segment["t_end"] - segment["t_start"]
    bandwidth = segment["f_high"] - segment["f_low"]
    return np.array([duration, bandwidth])

def extract_spectral_stats(
    y_seg,
    sr,
    n_fft=760,
    hop_length=256
):
    n_fft_eff = pick_n_fft_for_segment(len(y_seg), n_fft)
    centroid = librosa.feature.spectral_centroid(
        y=y_seg, sr=sr, n_fft=n_fft_eff, hop_length=hop_length
    )
    bandwidth = librosa.feature.spectral_bandwidth(
        y=y_seg, sr=sr, n_fft=n_fft_eff, hop_length=hop_length
    )
    rolloff = librosa.feature.spectral_rolloff(
        y=y_seg, sr=sr, n_fft=n_fft_eff, hop_length=hop_length
    )

    features = []
    for F in (centroid, bandwidth, rolloff):
        features.append(F.mean())
        features.append(F.std())

    return np.array(features)

def extract_segment_spectrogram(
    y_seg,
    sr,
    n_fft=760,
    hop_length=256
):
    n_fft_eff = pick_n_fft_for_segment(len(y_seg), n_fft)
    S = np.abs(librosa.stft(
        y_seg,
        n_fft=n_fft_eff,
        hop_length=hop_length
    ))
    return S

def extract_shape_features(S):
    energy_mask = S > np.percentile(S, 75)

    total_pixels = energy_mask.sum()
    height, width = energy_mask.shape

    aspect_ratio = width / max(height, 1)

    return np.array([
        total_pixels,
        aspect_ratio
    ])

def extract_segment_features(y, sr, segment):
    y_seg = extract_segment_waveform(y, sr, segment)

    if len(y_seg) < 0.01 * sr:   # 极短片段直接跳过
        return None

    mfcc_feat = extract_mfcc_features(y_seg, sr)
    if mfcc_feat is None:
        return None

    dur_bw    = extract_duration_bandwidth(segment)
    spec_stat = extract_spectral_stats(y_seg, sr)

    S_seg = extract_segment_spectrogram(y_seg, sr)
    shape = extract_shape_features(S_seg)

    feature_vector = np.concatenate([
        mfcc_feat,
        dur_bw,
        spec_stat,
        shape
    ])

    return feature_vector

def _skipped_segment_line(
    wav_path: str | Path,
    seg_index: int,
    seg: dict,
    *,
    reason: str,
    n_frames: int | None = None,
    skip_batch_summary: str | None = None,
    wave_samples: int | None = None,
    sample_rate: int | None = None,
) -> str:
    """
    skip_batch_summary: same格式 as控制台一行，
    「segments.csv + wav.wav -> xxx_MFCC.csv」，便于辨认是哪条任务里被 skip。

    第N段 = 按 segments CSV 行顺序从 1 数；segment_0based_index = Python/enumerate 下标。
    """
    wav_name = Path(wav_path).name
    ts, te = seg.get("t_start"), seg.get("t_end")
    dur = float(te) - float(ts)
    ord_1 = seg_index + 1

    fl = seg.get("f_low")
    fh = seg.get("f_high")
    freq_s = ""
    if fl is not None and fh is not None:
        freq_s = (
            f" f_low_hz={float(fl):.2f} f_high_hz={float(fh):.2f}"
        )

    wave_s = ""
    if wave_samples is not None:
        wave_s = f" wave_samples={int(wave_samples)}"
        if sample_rate is not None:
            wave_s += f" sr_hz={int(sample_rate)}"

    mfcc_s = ""
    if n_frames is not None:
        mfcc_s = (
            f" mfcc_time_frames={int(n_frames)}"
            f" (min_for_delta={MFCC_DELTA_MIN_FRAMES})"
        )

    detail = (
        f"第{ord_1}段 segment_0based_index={seg_index} |"
        f" t_start_s={float(ts):.6f} t_end_s={float(te):.6f} duration_s={dur:.6f}"
        f"{freq_s}{wave_s}{mfcc_s} | {reason}"
    )
    if skip_batch_summary:
        return (
            f"{skip_batch_summary} | SKIP | {detail}"
        )
    return f"  SKIP {wav_name} | {detail}"


def _log_skipped_segment(
    wav_path: str | Path,
    seg_index: int,
    seg: dict,
    *,
    reason: str,
    n_frames: int | None = None,
    print_to_stdout: bool = True,
    skip_log_fp: TextIO | None = None,
    skip_batch_summary: str | None = None,
    wave_samples: int | None = None,
    sample_rate: int | None = None,
    skip_lines_accumulator: list[str] | None = None,
):
    line = _skipped_segment_line(
        wav_path,
        seg_index,
        seg,
        reason=reason,
        n_frames=n_frames,
        skip_batch_summary=skip_batch_summary,
        wave_samples=wave_samples,
        sample_rate=sample_rate,
    )
    if skip_lines_accumulator is not None:
        skip_lines_accumulator.append(line)
    if print_to_stdout:
        print(line, flush=True)
    if skip_log_fp is not None:
        skip_log_fp.write(line + "\n")
        skip_log_fp.flush()


def extract_features_from_segments(
    wav_path,
    segments,
    log_skipped_segments=False,
    skip_log_fp: TextIO | None = None,
    skip_batch_summary: str | None = None,
    skip_lines_accumulator: list[str] | None = None,
):
    y, sr = librosa.load(wav_path, sr=None)

    X = []
    valid_segments = []

    for idx, seg in enumerate(segments):
        y_seg = extract_segment_waveform(y, sr, seg)
        short_wav = len(y_seg) < 0.01 * sr

        feat = extract_segment_features(y, sr, seg)

        if feat is None:
            if (
                log_skipped_segments
                or skip_log_fp is not None
                or skip_lines_accumulator is not None
            ):
                if short_wav:
                    _log_skipped_segment(
                        wav_path,
                        idx,
                        seg,
                        reason="waveform shorter than minimum (0.01 s)",
                        print_to_stdout=log_skipped_segments,
                        skip_log_fp=skip_log_fp,
                        skip_batch_summary=skip_batch_summary,
                        wave_samples=len(y_seg),
                        sample_rate=sr,
                        skip_lines_accumulator=skip_lines_accumulator,
                    )
                else:
                    mfcc_quick = librosa.feature.mfcc(
                        y=y_seg,
                        sr=sr,
                        n_mfcc=13,
                        n_fft=pick_n_fft_for_segment(len(y_seg), 760),
                        hop_length=256,
                    )
                    _log_skipped_segment(
                        wav_path,
                        idx,
                        seg,
                        reason="insufficient MFCC frames for delta",
                        n_frames=mfcc_quick.shape[1],
                        print_to_stdout=log_skipped_segments,
                        skip_log_fp=skip_log_fp,
                        skip_batch_summary=skip_batch_summary,
                        wave_samples=len(y_seg),
                        sample_rate=sr,
                        skip_lines_accumulator=skip_lines_accumulator,
                    )
            continue

        X.append(feat)
        valid_segments.append(seg)

    if not X:
        return np.empty((0, 0)), valid_segments

    return np.vstack(X), valid_segments

def get_feature_names(n_mfcc=13):
    names = []
    for prefix in ("mfcc", "delta", "delta2"):
        names.extend([f"{prefix}_mean_{i}" for i in range(1, n_mfcc + 1)])
        names.extend([f"{prefix}_std_{i}" for i in range(1, n_mfcc + 1)])

    names.extend([
        "duration",
        "bandwidth",
        "spectral_centroid_mean",
        "spectral_centroid_std",
        "spectral_bandwidth_mean",
        "spectral_bandwidth_std",
        "spectral_rolloff_mean",
        "spectral_rolloff_std",
        "shape_total_pixels",
        "shape_aspect_ratio"
    ])
    return names


def mfcc_csv_name_from_segments_csv(segments_csv_path: Path) -> str:
    """246_05_segments.csv -> 246_05_MFCC.csv（文件名里 _segments 换成 _MFCC）"""
    stem = segments_csv_path.stem
    if stem.endswith("_segments"):
        base = stem[: -len("_segments")]
        return f"{base}_MFCC.csv"
    return f"{stem}_MFCC.csv"


def core_from_segments_csv_stem(stem: str) -> str:
    """Stem of segments CSV → id used for matching wav (without _segments)."""
    return stem[:-len("_segments")] if stem.endswith("_segments") else stem


def is_segments_table_csv(path: Path) -> bool:
    """
    Batch MFCC only uses segment manifest CSVs (*_segments.csv), not sidecars like
    *_segments_info*.csv (stem must end with _segments and not contain _segments_info).
    """
    stem = path.stem
    if not stem.endswith("_segments"):
        return False
    if "_segments_info" in stem:
        return False
    return True


def resolve_wav_path(wav_dir: Path, core: str, strip_prefix: str) -> Path:
    """Match segmentation naming: prefixed wav or bare core, else unique *core*.wav"""
    wav_dir = Path(wav_dir)
    candidates = [
        wav_dir / f"{strip_prefix}{core}.wav",
        wav_dir / f"{core}.wav",
    ]
    for p in candidates:
        if p.is_file():
            return p.resolve()
    globs = sorted(wav_dir.glob(f"*{core}.wav")) + sorted(wav_dir.glob(f"*{core}.WAV"))
    uniq = []
    seen = set()
    for g in globs:
        try:
            r = g.resolve()
        except OSError:
            continue
        if r not in seen:
            seen.add(r)
            uniq.append(g)
    if len(uniq) == 1:
        return uniq[0].resolve()
    if len(uniq) > 1:
        raise FileNotFoundError(
            f"Multiple WAVs matching *{core}* in {wav_dir}: {[p.name for p in uniq]}"
        )
    raise FileNotFoundError(
        f"No WAV found for core={core!r} in {wav_dir} "
        f"(tried '{strip_prefix}{core}.wav' and '{core}.wav')"
    )


def save_segments_csv_pruned(segments: list, csv_path: Path) -> None:
    """
    Same columns as segmentation.save_segments_csv; unit_id renumbered 0..n-1 for kept rows only.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "unit_id",
                "t_start_s",
                "t_end_s",
                "duration_s",
                "f_low_hz",
                "f_high_hz",
            ]
        )
        for i, seg in enumerate(segments):
            t0, t1 = float(seg["t_start"]), float(seg["t_end"])
            w.writerow(
                [
                    i,
                    f"{t0:.6f}",
                    f"{t1:.6f}",
                    f"{t1 - t0:.6f}",
                    f"{float(seg['f_low']):.2f}",
                    f"{float(seg['f_high']):.2f}",
                ]
            )


def save_mfcc_csv(X: np.ndarray, csv_path: Path, n_mfcc: int = 13) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    feature_names = get_feature_names(n_mfcc=n_mfcc)
    if X.ndim != 2 or X.shape == (0, 0):
        X = np.empty((0, len(feature_names)), dtype=float)
    elif X.shape[1] != len(feature_names):
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    np.savetxt(
        csv_path,
        X,
        delimiter=",",
        fmt="%.6f",
        header=",".join(feature_names),
        comments="",
    )


def batch_mfcc_from_segments_dir(
    segments_dir: Path,
    wav_dir: Path,
    out_dir: Path,
    strip_prefix: str = "a b c d e f i ",
    n_mfcc: int = 13,
    log_skipped_segments: bool = True,
    verbose: bool = False,
    *,
    write_pruned_segments_csv: bool = False,
    overwrite_segments_csv: bool = False,
) -> None:
    """
    Writes skipped_segments.txt directly under <out_dir>/ for each batch run (overwritten).
    After all WAVs, appends an ALL SKIPPED SEGMENTS block listing every skip line again.

    Optional write_pruned_segments_csv / overwrite_segments_csv: drop MFCC-skipped rows from
    segment CSVs (same schema as segmentation.save_segments_csv); unit_id is renumbered.

    By default (verbose=False) stdout only shows SKIP lines when they occur; per-file
    progress and "Skip log saved" are suppressed. Use verbose=True to restore full prints.
    """
    segments_dir = Path(segments_dir)
    wav_dir = Path(wav_dir)
    out_dir = Path(out_dir)
    csv_files = sorted(
        (p for p in segments_dir.glob("*.csv") if is_segments_table_csv(p)),
        key=lambda p: p.name.lower(),
    )
    if not csv_files:
        raise ValueError(
            f"No *_segments.csv manifest files in {segments_dir} "
            f"(excluding *_segments_info*.csv)"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    skip_log_path = out_dir / "skipped_segments.txt"
    # Always replace the whole file (no append) so a new run does not keep old skip lines.
    skip_log_path.unlink(missing_ok=True)
    skip_fp = open(skip_log_path, "w", encoding="utf-8")
    skip_lines_all: list[str] = []
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    skip_fp.write(
        f"MFCC — skipped segments log\n"
        f"# This file is replaced entirely on each batch run (not appended).\n"
        f"time_utc={stamp}\n"
        f"segments_dir={segments_dir.resolve()}\n"
        f"wav_dir={wav_dir.resolve()}\n"
        f"out_dir={out_dir.resolve()}\n"
        f"# Lines with SKIP: 第N段 = 1-based row order in each *_segments.csv; "
        f"segment_0based_index = enumerate index; t_* = seconds in WAV timeline; "
        f"f_*_hz from segment CSV; wave_samples = clipped waveform length; "
        f"mfcc_time_frames = MFCC columns after crop (when relevant).\n"
    )

    try:
        for seg_csv in csv_files:
            mfcc_name = mfcc_csv_name_from_segments_csv(seg_csv)
            out_path = out_dir / mfcc_name
            core = core_from_segments_csv_stem(seg_csv.stem)

            wav_path = resolve_wav_path(wav_dir, core, strip_prefix)
            segments_list = load_segments(seg_csv)

            skip_fp.write(f"\n=== {seg_csv.name} + {wav_path.name} -> {mfcc_name} ===\n")
            skip_fp.flush()

            batch_skip_summary = (
                f"{seg_csv.name} + {wav_path.name} -> {mfcc_name}"
            )
            if verbose:
                print(batch_skip_summary + f" -> {out_path}", end=" ")
            X, valid_segments = extract_features_from_segments(
                str(wav_path),
                segments_list,
                log_skipped_segments=log_skipped_segments,
                skip_log_fp=skip_fp,
                skip_batch_summary=batch_skip_summary,
                skip_lines_accumulator=skip_lines_all,
            )
            if verbose:
                if X.size == 0:
                    print("(no features; empty CSV)")
                else:
                    print(f"shape {X.shape}")
            save_mfcc_csv(X, out_path, n_mfcc=n_mfcc)

            if write_pruned_segments_csv or overwrite_segments_csv:
                if write_pruned_segments_csv:
                    save_segments_csv_pruned(valid_segments, out_dir / seg_csv.name)
                if overwrite_segments_csv:
                    save_segments_csv_pruned(valid_segments, seg_csv)
                if verbose:
                    where = []
                    if write_pruned_segments_csv:
                        where.append(f"{out_dir / seg_csv.name}")
                    if overwrite_segments_csv:
                        where.append(f"{seg_csv.resolve()}")
                    print(f"  pruned segments CSV -> {', '.join(where)}")
        skip_fp.write("\n=== ALL SKIPPED SEGMENTS (batch summary, end of run) ===\n")
        skip_fp.write(f"total_skipped={len(skip_lines_all)}\n")
        for ln in skip_lines_all:
            skip_fp.write(ln + "\n")
        skip_fp.flush()
    finally:
        skip_fp.close()

    resolved_skip_log = skip_log_path.resolve()
    print(
        f"MFCC skipped_segments.txt: {len(skip_lines_all)} skipped segment(s) -> {resolved_skip_log}",
        flush=True,
    )
    if verbose:
        print(f"Skip log saved to {resolved_skip_log}")


if __name__ == "__main__":
    base = Path("/home/yuxuan.li/workspace/Birdsong-project/Wav/finch")
    default_segments = base / "246_O_seg"
    default_wav = base / "246_O"
    default_out = base / "246_O_MFCC"
    default_strip = "a b c d e f i "

    parser = argparse.ArgumentParser(
        description=(
            "Batch MFCC: segment CSVs -> *_MFCC.csv. "
            "Default: stdout only SKIP lines; use --verbose for per-file progress."
        )
    )
    parser.add_argument(
        "--segments_dir",
        type=Path,
        default=default_segments,
        help=(
            f"Folder of *_segments.csv manifests only; *_segments_info*.csv ignored "
            f"(default: {default_segments})"
        ),
    )
    parser.add_argument(
        "--wav_dir",
        type=Path,
        default=default_wav,
        help=f"Folder of WAV files (default: {default_wav})",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=default_out,
        help=f"Output folder for *_MFCC.csv (default: {default_out})",
    )
    parser.add_argument(
        "--strip_prefix",
        type=str,
        default=default_strip,
        help='WAV filename prefix before id (default: "a b c d e f i ")',
    )
    parser.add_argument("--n_mfcc", type=int, default=13, help="MFCC coefficients (default 13).")
    parser.add_argument(
        "--quiet_skips",
        action="store_true",
        help="Do not print per-segment MFCC skip lines (still saved to <out_dir>/skipped_segments.txt).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-file progress (path, shape) and 'Skip log saved' line. Default: quiet except SKIP lines.",
    )
    parser.add_argument(
        "--write_pruned_segments_csv",
        action="store_true",
        help=(
            "After MFCC, write *_segments.csv containing only rows that produced features "
            "(same filenames as inputs) into --out_dir alongside *_MFCC.csv."
        ),
    )
    parser.add_argument(
        "--overwrite_segments_csv",
        action="store_true",
        help=(
            "Destructive: overwrite each source CSV in --segments_dir with MFCC-valid segments only "
            "(same layout as segmentation.py)."
        ),
    )
    args = parser.parse_args()

    batch_mfcc_from_segments_dir(
        args.segments_dir,
        args.wav_dir,
        args.out_dir,
        strip_prefix=args.strip_prefix,
        n_mfcc=args.n_mfcc,
        log_skipped_segments=not args.quiet_skips,
        verbose=args.verbose,
        write_pruned_segments_csv=args.write_pruned_segments_csv,
        overwrite_segments_csv=args.overwrite_segments_csv,
    )