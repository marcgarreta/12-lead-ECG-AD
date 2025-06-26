import os
import sys
import pandas as pd
import numpy as np
import ast
import wfdb
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.model_selection import train_test_split

datasets = 'PTBXL'

data_folder = 'data'
output_folder = 'processed_data'

def normalize_ptbxl(a, min_a = None, max_a = None):
	if min_a is None: min_a, max_a = np.min(a, axis = 0), np.max(a, axis = 0)
	return (a - min_a) / (max_a - min_a + 0.0001), min_a, max_a

def bandpass_ptbxl(signal, lowcut=0.5, highcut=40, fs=100, order=3):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def notch_filter_ptbxl(signal, freq=50, q=30, fs=100):
    nyq = 0.5 * fs
    w0 = freq / nyq
    b, a = iirnotch(w0, q)
    return filtfilt(b, a, signal)

def preprocess_ecg_ptbxl(signal, fs=100):
    processed = []
    for lead in signal.T:
        lead = bandpass_ptbxl(lead, fs=fs)
        lead = notch_filter_ptbxl(lead, fs=fs)
        lead = (lead - np.mean(lead)) / np.std(lead)
        processed.append(lead)
    return np.array(processed).T

def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(os.path.join(path, f))[0] for f in df['filename_lr']]
    else:
        data = [wfdb.rdsamp(os.path.join(path, f))[0] for f in df['filename_hr']]
    return np.array(data)

def aggregate_diagnostic(y_dic, agg_df):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

def prepare_data(path):
    Y = pd.read_csv(os.path.join(path, 'ptbxl_database.csv'), index_col='ecg_id')
    Y['scp_codes'] = Y['scp_codes'].apply(ast.literal_eval)
    agg_df = pd.read_csv(os.path.join(path, 'scp_statements.csv'), index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]
    Y['diagnostic_superclass'] = Y.scp_codes.apply(lambda x: aggregate_diagnostic(x, agg_df))
    Y['is_normal'] = Y['diagnostic_superclass'].apply(lambda x: x == ['NORM'])
    return Y

def split_data(Y, test_size=0.15, random_state=42):
    # Select only normal samples for training
    normal_df = Y[Y['is_normal'] == True]
    abnormal_df = Y[Y['is_normal'] == False]

    # Split the normal samples into train and test
    train_df, test_normal_df = train_test_split(
        normal_df,
        test_size=test_size,
        random_state=random_state,
        stratify=normal_df['is_normal']
    )
    
    # Combine normal and abnormal samples for the test set
    test_abnormal_df = abnormal_df.sample(n=len(test_normal_df), random_state=random_state)

    # Concatenate normal and abnormal samples to form the final test set
    test_df = pd.concat([test_normal_df, test_abnormal_df]).sample(frac=1, random_state=random_state).reset_index(drop=True)

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

def load_data(dataset, dataset_type):
     if dataset_type == 'PTBXL':
        folder = os.path.join(output_folder, dataset)
        os.makedirs(folder, exist_ok=True)
        dataset_folder = os.path.join(data_folder, 'PTBXL/')

        Y = prepare_data(dataset_folder)
        train_df, test_df = split_data(Y)

        sampling_rate = 100 

        def process_and_save(df, name):
            signals = load_raw_data(df, sampling_rate, dataset_folder)
            processed = np.array([preprocess_ecg_ptbxl(sig, fs=sampling_rate) for sig in signals])
            np.save(os.path.join(folder, f'{name}.npy'), processed)

        process_and_save(train_df, 'train')
        process_and_save(test_df, 'test')

        # Create labels: 1 for abnormal, 0 for normal
        test_labels = test_df['is_normal'].apply(lambda x: 0 if x else 1).values
        np.save(os.path.join(folder, 'labels.npy'), test_labels)

if __name__ == '__main__':
    commands = sys.argv[1:]
    if len(commands) > 0:
        for d in commands:
            if d.upper() == 'PTBXL':
                load_data(d, 'PTBXL')
            else:
                print(f"Unknown dataset: {d}")
    else:
        print("Usage: python preprocess.py <datasets>")
        print("where <datasets> is a space-separated list of datasets (e.g., PTBXL)")
