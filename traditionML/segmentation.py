import argparse
import csv
from pathlib import Path

import numpy as np
import librosa
import librosa.display
from scipy.ndimage import binary_opening, binary_closing
import matplotlib.pyplot as plt

def compute_spectrogram(
    wav_path,
    sr=None, # keep original sampling rate for better high-frequency recall
    n_fft=1024, # window size for STFT
    hop_length=256
):
    y, sr = librosa.load(wav_path, sr=sr)
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, center=False))**2 # power spectrogram
    S_db = librosa.power_to_db(S, ref=np.median(S) * 100)
    #S_db = librosa.power_to_db(S, ref=np.max) #log-scaled spectrogram
    return S_db, sr

def band_limit_spectrogram(S_db, sr, fmin=2000, fmax=6000):
    n_fft = (S_db.shape[0] - 1) * 2  # derive n_fft from S_db row count
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]
    return S_db[idx, :], freqs[idx]

def energy_threshold_mask(S_db, threshold_db=-30):
    """
    threshold_db: relative to max (e.g. -30 dB)
    """
    max_db = S_db.max()
    mask = S_db > (max_db + threshold_db)
    return mask

def clean_mask(mask, min_size_time=2, min_size_freq=2):
    structure = np.ones((min_size_freq, min_size_time)) # structuring element for morphological operations
    mask = binary_opening(mask, structure=structure)
    mask = binary_closing(mask, structure=structure)
    return mask

def extract_segments(
    mask,
    freqs,
    sr,
    hop_length=256,
    min_duration=0.01,
    min_active_bins=2,
    min_silence_gap=0.01
):
    segments = []
    if mask.size == 0:
        return segments

    # 将二维时频掩码压到时间轴，得到每一帧是否“有足够能量”
    active_t = np.sum(mask, axis=0) >= min_active_bins
    if not np.any(active_t):
        return segments

    # 合并极短静音空隙，避免一个音符被切成多个碎段
    gap_frames = max(int(round(min_silence_gap * sr / hop_length)), 0)
    if gap_frames > 0:
        false_idx = np.where(~active_t)[0]
        if false_idx.size > 0:
            gap_start = false_idx[0]
            prev = false_idx[0]
            for idx in false_idx[1:]:
                if idx != prev + 1:
                    if (
                        (prev - gap_start + 1) <= gap_frames
                        and gap_start > 0
                        and prev < len(active_t) - 1
                        and active_t[gap_start - 1]
                        and active_t[prev + 1]
                    ):
                        active_t[gap_start:prev + 1] = True
                    gap_start = idx
                prev = idx
            if (
                (prev - gap_start + 1) <= gap_frames
                and gap_start > 0
                and prev < len(active_t) - 1
                and active_t[gap_start - 1]
                and active_t[prev + 1]
            ):
                active_t[gap_start:prev + 1] = True

    # 在时间轴上提取连续 True 区间（保证时段互不重叠）
    active_int = active_t.astype(np.int8)
    changes = np.diff(np.pad(active_int, (1, 1)))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    for t_start_idx, t_end_idx in zip(starts, ends):
        duration = (t_end_idx - t_start_idx) * hop_length / sr
        if duration < min_duration:
            continue

        # 在该时间段内统计频率上下界
        local_mask = mask[:, t_start_idx:t_end_idx]
        active_f = np.sum(local_mask, axis=1) > 0
        if not np.any(active_f):
            continue
        f_idx = np.where(active_f)[0]
        f_start_idx, f_end_idx = f_idx[0], f_idx[-1]

        segments.append({
            "t_start": t_start_idx * hop_length / sr,
            "t_end":   t_end_idx * hop_length / sr,
            "f_low":   freqs[f_start_idx],
            "f_high":  freqs[f_end_idx]
        })

    return segments


def save_segments_csv(segments, csv_path):
    """Write segments as CSV with header for spreadsheets / downstream tools."""
    csv_path = Path(csv_path) if not isinstance(csv_path, Path) else csv_path
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
            t0, t1 = seg["t_start"], seg["t_end"]
            w.writerow(
                [
                    i,
                    f"{t0:.6f}",
                    f"{t1:.6f}",
                    f"{t1 - t0:.6f}",
                    f"{seg['f_low']:.2f}",
                    f"{seg['f_high']:.2f}",
                ]
            )


def segments_csv_basename(stem: str, strip_prefix: str) -> str:
    """e.g. stem 'a b c d e f i 246_05' + prefix 'a b c d e f i ' → '246_05_segments.csv'"""
    s = stem
    if strip_prefix and stem.startswith(strip_prefix):
        s = stem[len(strip_prefix) :].lstrip()
    if not s:
        s = stem
    safe = "".join(c if c not in '\\/:*?"<>|' else "_" for c in s)
    return f"{safe}_segments.csv"


def segment_vocal_units(
    wav_path,
    sr=None,
    n_fft=1024,
    hop_length=64,
    threshold_db=-45,
    fmin=1500,
    fmax=10000,
    min_size_time=2,
    min_size_freq=2,
    min_duration=0.008,
    min_active_bins=1,
    min_silence_gap=0.02,
    plot=True,
):
    S_db, sr = compute_spectrogram(wav_path, sr=sr, n_fft=n_fft, hop_length=hop_length)
    S_band, freqs = band_limit_spectrogram(S_db, sr, fmin=fmin, fmax=fmax)

    mask = energy_threshold_mask(S_band, threshold_db=threshold_db)
    mask = clean_mask(mask, min_size_time=min_size_time, min_size_freq=min_size_freq)
    if plot:
        plot_segmentation(S_band, mask, sr=sr, hop_length=hop_length, fmin=fmin, fmax=fmax)
    segments = extract_segments(
        mask,
        freqs,
        sr,
        hop_length=hop_length,
        min_duration=min_duration,
        min_active_bins=min_active_bins,
        min_silence_gap=min_silence_gap
    )

    return segments

def plot_segmentation(S_db, mask, sr=22050, hop_length=256, fmin=0, fmax=3000):
    plt.figure(figsize=(12, 5))
    img = librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, fmin=fmin, fmax=fmax, y_axis="hz", x_axis="time")
    plt.contourf(mask, levels=[0.5, 1.5], colors=['red'], alpha=0.3)
    plt.colorbar(img, label="dB")
    plt.title("Segmentation Result")
    plt.show()


def process_wav_directory(
    wav_dir,
    out_dir,
    strip_prefix="a b c d e f i ",
    plot=False,
    **segment_kwargs,
):
    wav_dir = Path(wav_dir)
    out_dir = Path(out_dir)
    if not wav_dir.is_dir():
        raise NotADirectoryError(wav_dir)

    wav_files = sorted(wav_dir.glob("*.wav")) + sorted(wav_dir.glob("*.WAV"))
    # 去重（大小写重复时）
    seen = set()
    unique = []
    for p in wav_files:
        key = p.resolve()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    wav_files = sorted(unique, key=lambda p: p.name.lower())

    if not wav_files:
        raise ValueError(f"No .wav files in {wav_dir}")

    for wav_path in wav_files:
        segments = segment_vocal_units(str(wav_path), plot=plot, **segment_kwargs)
        basename = segments_csv_basename(wav_path.stem, strip_prefix)
        csv_path = out_dir / basename
        save_segments_csv(segments, csv_path)
        print(f"{wav_path.name} -> {csv_path} ({len(segments)} segments)")


if __name__ == "__main__":
    default_wav_dir = Path(
        "/home/yuxuan.li/workspace/Birdsong-project/Wav/finch/246_O"
    )
    default_strip = "a b c d e f i "
    default_out = default_wav_dir.parent / f"{default_wav_dir.name}_seg"

    parser = argparse.ArgumentParser(
        description="Segment all WAVs in a folder; save *_segments.csv to output dir."
    )
    parser.add_argument(
        "--wav_dir",
        type=Path,
        default=default_wav_dir,
        help=f"Folder containing .wav files (default: {default_wav_dir})",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Output CSV folder (default: <wav_dir.parent>/<wav_dir.name>_seg)",
    )
    parser.add_argument(
        "--strip_prefix",
        type=str,
        default=default_strip,
        help='Strip this prefix from filename stem before naming CSV (default: "a b c d e f i ")',
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show spectrogram overlay for each file (slow; mostly for debugging)",
    )
    args = parser.parse_args()

    out_dir = args.out_dir if args.out_dir is not None else (
        args.wav_dir.parent / f"{args.wav_dir.name}_seg"
    )
    process_wav_directory(
        args.wav_dir,
        out_dir,
        strip_prefix=args.strip_prefix,
        plot=args.plot,
    )