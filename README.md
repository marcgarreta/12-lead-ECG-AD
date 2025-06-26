# 12-lead-ECG-AD

ECG Anomaly Detection Pipeline

This project implements an anomaly detection pipeline for ECG data using Python and PyTorch. The models are trained exclusively on normal (non-anomalous) recordings from the MIMIC-IV ECG dataset.

Project Structure
	•	src/ - Contains implementations of the five different models evaluated for anomaly detection.
	•	src/data_preprocessing/ - Includes scripts for preprocessing the PTB-XL and MIMIC-IV ECG datasets.

Models Evaluated

In the src/ directory, you will find five distinct model architectures, each with its own training and evaluation scripts:
	1.	Convolutional Autoencoder (CAE)
	2.	Autoencoder with Attention (AE + Attention)
	3.	Variational Autoencoder with Local Attention (VAE + Local Attention)
	4.	Variational Autoencoder with Global Attention (VAE + Global Attention)
	5.	Variational Autoencoder with GRU Layers (VAE + GRU)

Data Preprocessing

Preprocessing routines are stored under preprocessing/:
	•	PTB-XL: Signal normalization, heartbeat segmentation, and noise filtering.
	•	MIMIC-IV ECG: Raw signal extraction, resampling, and artifact removal.


Developed as part of the Undergraduate Thesis (TFG) on ECG Anomaly Detection.
