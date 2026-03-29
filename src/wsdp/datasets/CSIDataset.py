import torch
import numpy as np

from torch.utils.data import Dataset

class CSIDataset(Dataset):
    def __init__(self, data_list, labels, use_phase=False):
        if use_phase and np.iscomplexobj(data_list):
            # Phase+Amplitude dual-channel: stack |H| and angle(H) along
            # last axis.  PA-CSI (Sensors 2025) shows this outperforms
            # amplitude-only for most sensing tasks.
            amplitude = np.abs(data_list)
            phase = np.angle(data_list)
            data_list = np.concatenate([amplitude, phase], axis=-1)
        else:
            data_list = np.abs(data_list)
        self.data_list = torch.from_numpy(data_list).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data_list[idx], self.labels[idx]