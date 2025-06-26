import os
import wfdb
import torch
from torch.utils.data import Dataset

class ECGDataset(Dataset):
    def __init__(self, records, num_leads=12, sample_length=5000):
        self.records = records
        self.num_leads = num_leads
        self.sample_length = sample_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        folder_path, record_id = self.records[idx]
        record_path = os.path.join(folder_path, record_id)
        record = wfdb.rdrecord(record_path)
        signal = torch.tensor(record.p_signal[:, :self.num_leads], dtype=torch.float32)

        if signal.shape[0] >= self.sample_length:
            return signal[:self.sample_length]
        else:
            pad_len = self.sample_length - signal.shape[0]
            padding = torch.zeros(pad_len, self.num_leads)
            return torch.cat([signal, padding], dim=0)

class ECGWindowDataset(Dataset):
    def __init__(self, base_dataset, window_size=500, stride=250):
        self.base_dataset = base_dataset
        self.window_size = window_size
        self.stride = stride
        self.windows = []

        for idx in range(len(base_dataset)):
            signal = base_dataset[idx]
            length = signal.shape[0]
            for start in range(0, length - window_size + 1, stride):
                self.windows.append((idx, start))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        ecg_idx, start = self.windows[idx]
        signal = self.base_dataset[ecg_idx]
        window = signal[start:start + self.window_size]
        return window.T  # [12, window_size]

