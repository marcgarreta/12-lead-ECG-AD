# 12-lead-ECG-AD

ECG Anomaly Detection Pipeline

This project implements an anomaly detection pipeline for ECG data using Python and PyTorch. The models are trained exclusively on normal (non-anomalous) recordings from the MIMIC-IV ECG dataset.

Project Structure
	•	src/ - Contains implementations of the five different models evaluated for anomaly detection.
	•	preprocessing/ - Includes scripts for preprocessing the PTB-XL and MIMIC-IV ECG datasets.

Models Evaluated

In the src/ directory, you will find five distinct model architectures, each with its own training and evaluation scripts:
	1.	Convolutional Autoencoder
	2.	Variational Autoencoder (VAE)
	3.	LSTM Autoencoder
	4.	Transformer-based Autoencoder
	5.	Generative Adversarial Network (GAN) for Anomaly Detection

Data Preprocessing

Preprocessing routines are stored under preprocessing/:
	•	PTB-XL: Signal normalization, heartbeat segmentation, and noise filtering.
	•	MIMIC-IV ECG: Raw signal extraction, resampling, and artifact removal.

Requirements
	•	Python 3.8 or higher
	•	PyTorch 1.10 or higher
	•	numpy, pandas, scipy, matplotlib

Installation and Usage
	1.	Clone this repository:

git clone <repository-url>
cd <repository-directory>


	2.	Install dependencies:

pip install -r requirements.txt


	3.	Run data preprocessing:

python preprocessing/ptbxl_preprocessing.py
python preprocessing/mimic_preprocessing.py


	4.	Train and evaluate a model (example for the Convolutional Autoencoder):

python src/conv_autoencoder.py --train



⸻

Developed as part of the Undergraduate Thesis (TFG) on ECG Anomaly Detection.
