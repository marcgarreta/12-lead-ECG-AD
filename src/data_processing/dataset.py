from __future__ import annotations
import argparse
from pathlib import Path
import shutil

import pandas as pd
import wfdb
from tqdm.auto import tqdm

def select_normal_ecgs(input_csv: Path, mimic_root: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    normal_df = df[(df["report_0"] == "Sinus rhythm") & (df["report_2"] == "Normal ECG")]
    print(f"✅ Found {len(normal_df)} normal ECGs.")

    study_ids = []

    for _, row in tqdm(normal_df.iterrows(), total=len(normal_df), desc="Copying records"):
        study_id   = str(row["study_id"])
        subject_id = str(row["subject_id"])

        record_base = mimic_root / f"p{subject_id[:4]}" / f"p{subject_id}" / f"s{study_id}" / study_id
        hea_file, dat_file = record_base.with_suffix(".hea"), record_base.with_suffix(".dat")

        if not (hea_file.exists() and dat_file.exists()):
            print(f"⚠️  Missing .hea/.dat for {study_id}")
            continue

        try:
            sig, _ = wfdb.rdsamp(str(record_base))
            if sig.shape[1] != 12:
                print(f"⚠️  {study_id} is not 12-lead")
                continue

            study_ids.append(study_id)

            # copy files
            shutil.copy2(dat_file, output_dir / f"{study_id}.dat")
            lines = Path(hea_file).read_text().splitlines()
            if not any("Labels" in l for l in lines):
                lines.append("# Labels: NORM")
            (output_dir / f"{study_id}.hea").write_text("\n".join(lines))

        except Exception as e:
            print(f"⚠️  Skipping {study_id}: {e}")

    # save manifest
    pd.DataFrame({"study_id": study_ids}).to_csv(output_dir / "ids.csv", index=False)
    print(f"💾 Copied {len(study_ids)} normal ECGs to {output_dir}")

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[3]        # ecg/data/ -> ecg -> src -> REPO
    default_out = repo_root / "data" / "processed" / "mimic" / "interim"

    ap = argparse.ArgumentParser(description="Copy only normal-ECG MIMIC studies")
    ap.add_argument("--input_csv",  type=Path, required=True,
                    help="machine_measurements.csv from PhysioNet download")
    ap.add_argument("--mimic_root", type=Path, required=True,
                    help="Root folder containing the PhysioNet 'files/' tree")
    ap.add_argument("--out_dir",    type=Path, default=default_out,
                    help=f"Destination (default: {default_out})")
    args = ap.parse_args()

    select_normal_ecgs(args.input_csv, args.mimic_root, args.out_dir)
