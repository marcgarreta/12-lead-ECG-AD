#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, iirnotch, filtfilt
from tqdm.auto import tqdm
import ast
import sys
SCRIPT_DIR = Path(__file__).resolve().parent
# Assuming script is in src/data_processing/, project root is two levels up
PROJECT_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_RAW = PROJECT_ROOT / 'data' / 'raw'
DEFAULT_PROCESSED = PROJECT_ROOT / 'data' / 'processed'

def find_nan_files(data_dir: Path, pattern: str = "*.npy") -> list[Path]:
    nan_files = []
    for f in data_dir.glob(pattern):
        arr = np.load(str(f))
        if np.isnan(arr).any():
            nan_files.append(f)
    return nan_files

def delete_files(file_list: list[Path]) -> int:
    deleted = 0
    for f in file_list:
        try:
            f.unlink()
            deleted += 1
        except:
            pass
    return deleted

def butter_bandpass(signal: np.ndarray, lowcut: float, highcut: float, fs: int, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, signal, axis=0)

def notch_filter(signal: np.ndarray, freq: float, q: float, fs: int) -> np.ndarray:
    b, a = iirnotch(freq/(fs/2), q)
    return filtfilt(b, a, signal, axis=0)

def zscore(signal: np.ndarray) -> np.ndarray:
    return (signal - np.mean(signal, axis=0)) / (np.std(signal, axis=0) + 1e-8)

def process_ptbxl(input_dir: Path, csv_path: Path, output_dir: Path):
    """
    Process PTB-XL WFDB records listed in CSV: filter, normalize, and save .npy arrays.
    The CSV used is 'ptbxl_database.csv' which includes a 'sampling_rate' column.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)

    # Load scp_statements.csv for diagnostic aggregation
    agg_path = input_dir / 'scp_statements.csv'
    agg_df = pd.read_csv(agg_path, index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    def aggregate_diagnostic(y_dic):
        # y_dic is dict mapping scp_code to value
        classes = []
        for code in y_dic.keys():
            if code in agg_df.index:
                classes.append(agg_df.loc[code, 'diagnostic_class'])
        return list(set(classes))

    # Parse scp_codes (string representation of dict) and compute superclass
    df['scp_codes'] = df['scp_codes'].apply(lambda x: ast.literal_eval(x))
    df['diagnostic_superclass'] = df['scp_codes'].apply(aggregate_diagnostic)

    # Keep only records whose diagnostic_superclass list includes "NORM"
    df = df[df['diagnostic_superclass'].apply(lambda lst: 'NORM' in lst)]
    # df = df[df['sampling_rate'] == 500]
    print(f"Filtered to {len(df)} normal PTB-XL records (500Hz subset via records500/).")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc='PTB-XL'):
        rec_name = str(row['filename_hr'])
        rec_path = input_dir / rec_name
        rec_file = Path(rec_name).name  # extract basename, e.g. "21837_hr"
        try:
            record = wfdb.rdrecord(str(rec_path))
            sig = record.p_signal.T.astype(np.float64)  # (12, T)
            fs = int(record.fs)
            for i in range(sig.shape[0]):
                lead = sig[i]
                lead = butter_bandpass(lead, 0.5, 40.0, fs)
                lead = notch_filter(lead, freq=50.0, q=30, fs=fs)
                sig[i] = zscore(lead)
            np.save(output_dir / f"{rec_file}.npy", sig.astype(np.float32))
        except Exception as e:
            print(f"⚠️  PTB-XL {rec_name} failed: {e}")

def process_mimic_direct(input_dir: Path, output_dir: Path):
    """
    Select only normal ECGs from MIMIC-IV and process them directly into .npy arrays.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Read CSV and filter for normal sinus rhythm ECGs
    df = pd.read_csv(input_dir / 'machine_measurements.csv')
    normal_df = df[(df['report_0'] == 'Sinus rhythm') & (df['report_2'] == 'Normal ECG')]
    print(f"Found {len(normal_df)} normal MIMIC records.")

    # 2) Process each normal record
    for _, row in tqdm(normal_df.iterrows(), total=len(normal_df), desc='MIMIC Direct'):
        study_id = str(row['study_id'])
        subj     = str(row['subject_id'])
        rec_base = input_dir / 'files' / f"p{subj[:4]}" / f"p{subj}" / f"s{study_id}" / study_id

        try:
            record = wfdb.rdrecord(str(rec_base))
            sig = record.p_signal.T.astype(np.float64)  # (12, T)
            fs  = int(record.fs)

            # filter and normalize per lead
            for i in range(sig.shape[0]):
                lead = sig[i]
                lead = butter_bandpass(lead, 0.5, 40.0, fs)
                lead = notch_filter(lead, freq=60.0, q=30, fs=fs)
                sig[i] = zscore(lead)

            # save directly to npy
            np.save(output_dir / f"{study_id}.npy", sig.astype(np.float32))
        except Exception as e:
            print(f"⚠️  MIMIC {study_id} failed: {e}")

def main():
    p = argparse.ArgumentParser(description='Preprocess ECG datasets modularly')
    p.add_argument('--dataset', choices=['ptbxl', 'mimic'], required=True,
                   help='Which dataset to process: ptbxl or mimic')
    p.add_argument('--input-dir', type=Path, default=DEFAULT_RAW,
                   help=f'Base raw data directory (default: {DEFAULT_RAW})')
    p.add_argument('--output-dir', type=Path, default=DEFAULT_PROCESSED,
                   help=f'Base output directory (default: {DEFAULT_PROCESSED})')
    p.add_argument('--clean-nans', action='store_true',
                   help='Remove any .npy files containing NaNs after processing')
    args = p.parse_args()

    args.input_dir = args.input_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()

    if not args.input_dir.exists():
        print(f"❌ Input directory not found: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    if args.dataset == 'ptbxl':
        print('Starting PTB-XL preprocessing...')
        process_ptbxl(
            input_dir=args.input_dir,  # root of PTB-XL download
            csv_path=args.input_dir / 'ptbxl_database.csv',
            output_dir=args.output_dir / 'ptbxl'
        )

    elif args.dataset == 'mimic':
        print('Selecting + Processing MIMIC normal ECGs directly to .npy...')
        process_mimic_direct(
            input_dir=args.input_dir,
            output_dir=args.output_dir / 'mimic' 
        )
        if args.clean_nans:
            npy_dir = args.output_dir / 'mimic' 
            print(f"Scanning {npy_dir} for NaNs...")
            nan_files = find_nan_files(npy_dir)
            if nan_files:
                count = delete_files(nan_files)
                print(f"Deleted {count}/{len(nan_files)} NaN-containing files.")
            else:
                print("No NaN-containing files found.")

if __name__ == '__main__':
    main()
