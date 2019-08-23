import torch
from torch.utils.data import Dataset

class Dataset(Dataset):
    def __init__(self, data):
        self.data = data
        # every pulses should be 1024 sample points
        self.padded_len = 1024

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def collate_fn(self, datas):
        batch = {}

        batch["id"] = [data["id"] for data in datas]

        batch['label'] = torch.LongTensor([data["label"] for data in datas])

        batch['ecg'] = torch.tensor(
            [pad_to_len(data['ecg'], self.padded_len, 0)
             for data in datas]
        )

        return batch


def pad_to_len(pulse, padded_len, padding):
    padded_pulse = []
    for i in range(padded_len):
        if i < len(pulse):
            padded_pulse.append(pulse[i])
        else:
            padded_pulse.append(padding)

    return padded_pulse
