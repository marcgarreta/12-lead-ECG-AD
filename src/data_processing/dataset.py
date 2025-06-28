import os
import pandas as pd
import wfdb
import shutil
from tqdm import tqdm

# This script now ONLY copies .dat and .hea files for normal ECGs into `output_dir`.
input_csv = "/fhome/mgarreta/mimiciv_/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/machine_measurements.csv"
mimiciv_root = "/fhome/mgarreta/mimiciv_/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/files"
output_dir = "/fhome/mgarreta/MIMICIV_PREPROCESSED"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(input_csv)
normal_df = df[(df["report_0"] == "Sinus rhythm") & (df["report_2"] == "Normal ECG")]
print(f"✅ Found {len(normal_df)} normal ECGs with sinus rhythm and normal report.")

study_ids = []

for _, row in tqdm(normal_df.iterrows(), total=len(normal_df), desc="Processing ECGs"):
    study_id = str(row["study_id"])
    subject_id = str(row["subject_id"])

    # Construct path: /files/pXXXX/pXXXXXXXX/sYYYYYYYY/YYYYYYYY
    record_base = os.path.join(
        mimiciv_root,
        f"p{subject_id[:4]}",
        f"p{subject_id}",
        f"s{study_id}",
        study_id
    )
    hea_file = record_base + ".hea"
    dat_file = record_base + ".dat"

    if not (os.path.exists(hea_file) and os.path.exists(dat_file)):
        print(f"⚠️ Skipping {study_id}: Missing .hea/.dat")
        continue

    try:
        # no preprocessing, just verify the record loads and has 12 leads
        signal, _ = wfdb.rdsamp(record_base)
        if signal.shape[1] != 12:
            print(f"⚠️ Skipping {study_id}: not 12 leads")
            continue

        study_ids.append(study_id)

        # Copy .dat into the shared folder
        shutil.copy(dat_file, os.path.join(output_dir, f"{study_id}.dat"))

        # Modify and copy .hea file
        with open(hea_file, "r") as f:
            lines = f.readlines()
        if not any("Labels" in line for line in lines):
            lines.append("# Labels: NORM\n")
        with open(os.path.join(output_dir, f"{study_id}.hea"), "w") as f:
            f.writelines(lines)

    except Exception as e:
        print(f"⚠️ Skipping {study_id}: {str(e)}")

# ==== SAVE OUTPUT ====
pd.DataFrame(study_ids, columns=["study_id"]).to_csv(os.path.join(output_dir, "ids.csv"), index=False)

print(f"\n💾 Copied {len(study_ids)} normal ECGs (.dat & .hea)")
