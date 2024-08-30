import torch
from torch.utils.data import Dataset

class VideoMIDIDataset(Dataset):
    def __init__(self, data_pairs):
        self.data_pairs = data_pairs

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        midi_data, video_data = self.data_pairs[idx]
        return torch.tensor(video_data, dtype=torch.float32).permute(3, 0, 1, 2), torch.tensor(midi_data, dtype=torch.float32)
