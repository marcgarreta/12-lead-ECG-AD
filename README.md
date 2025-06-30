12-Lead ECG Anomaly Detection

Unsupervised detection and localisation of anomalies in 12-lead electrocardiograms (ECGs) using attention-augmented variational autoencoders, plus an interactive Dash dashboard for visual triage.

⸻

Academic context

This code accompanies my Final Degree Project in Artificial Intelligence (Universitat Autònoma de Barcelona, Escola d’Enginyeria).
	•	Author: Marc Garreta Basora
	•	Supervisor: Dr. Mehmet Oguz Mulayim – @omulayim
	•	Report: see /docs/final_report.pdf for a full technical write-up.

⸻

Quick start

# clone repo
git clone https://github.com/marcgarreta/12-lead-ECG-AD
cd 12-lead-ECG-AD

# create & activate environment
conda env create -f environment.yml
conda activate ecg-anomaly-detection

# data preprocessing (choose one)

## If using PTB-XL (open)
python pre-process.py \
  --dataset ptbxl \
  --input-dir /path/to/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3

## If using MIMIC-IV ECG (restricted)
python pre-process.py \
  --dataset mimic \
  --input-dir /path/to/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0 \
  --clean-nans


⸻

Data preprocessing

All commands assume you are at the project root.
Output goes to data/processed/.

If using PTB-XL (open)

python pre-process.py \
  --dataset ptbxl \
  --input-dir /path/to/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3

If using MIMIC-IV ECG (restricted)

python pre-process.py \
  --dataset mimic \
  --input-dir /path/to/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0 \
  --clean-nans


⸻

Training

# attention-VAE (best model)
python train.py --config configs/vae_mha.yaml

# other baselines
python train.py --config configs/cae.yaml
python train.py --config configs/vae_bilstm.yaml


⸻

Running the UI

python ui/app.py          # opens http://127.0.0.1:8050

Upload a pre-processed sample (.npy, .dat/.hea), pick a model and threshold, and the dashboard will render
• 12 leads with reconstruction overlay
• attention + error heatmaps
• global anomaly score & verdict.

⸻

Key results (CPSC 2018 test set)

Model	F1	Recall	PR-AUC
VAE-BiLSTM + Multi-Head Attention	0.916	0.98	0.814


⸻

Repository layout

configs/            experiment YAMLs
data/               raw + processed + inference datasets (place where data will be stored)
ecg_ad/             library code
pre-process.py      dataset loader & cleaner
train.py            experiment runner
ui/                 Dash dashboard


⸻

