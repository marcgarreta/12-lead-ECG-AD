from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import butter, iirnotch, filtfilt


def butter_bandpass(signal: np.ndarray, lowcut: float, highcut: float, fs: int, order: int = 3) -> np.ndarray:
    """Apply a Butterworth band-pass filter to a 1D signal."""
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, signal)


def iir_notch(signal: np.ndarray, freq: float, q: float, fs: int) -> np.ndarray:
    """Apply an IIR notch filter at a given frequency to a 1D signal."""
    nyq = 0.5 * fs
    b, a = iirnotch(freq / nyq, q)
    return filtfilt(b, a, signal)


def zscore(signal: np.ndarray) -> np.ndarray:
    """Z-score normalize a 1D signal."""
    return (signal - np.mean(signal)) / np.std(signal)


def process_leads(p_signal: np.ndarray, fs: int, notch_freq: float) -> np.ndarray:
    """Filter and normalize each lead in a 2D ECG signal array."""
    processed = np.empty_like(p_signal, dtype=np.float64)
    for i in range(p_signal.shape[1]):
        lead = p_signal[:, i]
        lead = butter_bandpass(lead, 0.5, 40.0, fs)
        lead = iir_notch(lead, notch_freq, q=30, fs=fs)
        lead = zscore(lead)
        processed[:, i] = lead
    return processed


def rescale_to_int16(p_signal: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """Convert float signal to int16 and compute per-lead ADC gains."""
    int_sig = np.zeros_like(p_signal, dtype=np.int16)
    gains: list[float] = []
    for i in range(p_signal.shape[1]):
        lead = p_signal[:, i]
        max_abs = np.max(np.abs(lead))
        scale = 32767.0 / max_abs if max_abs > 0 else 1.0
        int_sig[:, i] = (lead * scale).astype(np.int16)
        gains.append(1.0 / scale)
    return int_sig, gains


def process_record(hea_path: Path, notch_freq: float) -> None:
    """Read a WFDB record, preprocess it in-place, and overwrite original files."""
    rec = wfdb.rdrecord(str(hea_path.with_suffix("")))
    fs = int(rec.fs)

    processed = process_leads(rec.p_signal.astype(np.float64), fs, notch_freq)
    int_sig, gains = rescale_to_int16(processed)

    base_comments = [c for c in rec.comments if not (
        c.startswith("Filtered:") or c.startswith("Notch:") or c.startswith("Per-lead")
    )]
    comments = base_comments + [
        "Filtered: 0.5-40 Hz 3rd-order Butterworth band-pass",
        f"Notch: {notch_freq} Hz Q=30",
        "Per-lead z-score normalized",
    ]

    wfdb.wrsamp(
        record_name=rec.record_name,
        fs=fs,
        units=rec.units,
        sig_name=rec.sig_name,
        d_signal=int_sig,
        fmt=["16"] * int_sig.shape[1],
        adc_gain=gains,
        baseline=[0] * int_sig.shape[1],
        write_dir=str(hea_path.parent),
        comments=comments,
    )


def walk_records(input_dir: Path, recursive: bool, notch_freq: float) -> None:
    """Process all WFDB `.hea` records in `input_dir` in-place."""
    pattern = "**/*.hea" if recursive else "*.hea"
    for hea_path in input_dir.glob(pattern):
        rel = hea_path.relative_to(input_dir)
        print(f"→ Processing {rel}")
        try:
            process_record(hea_path, notch_freq)
        except Exception as e:
            print(f"⚠️  Failed {rel}: {e}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]

    ap = argparse.ArgumentParser(description="In-place preprocessing for MIMIC-IV ECG WFDB records")
    ap.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Folder containing .hea/.dat files to preprocess in place",
    )
    ap.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories of `input_dir`",
    )
    ap.add_argument(
        "--notch_freq",
        type=float,
        default=60.0,
        help="Mains notch frequency (Hz)",
    )
    args = ap.parse_args()

    walk_records(args.input_dir, args.recursive, args.notch_freq)


if __name__ == "__main__":
    main()
