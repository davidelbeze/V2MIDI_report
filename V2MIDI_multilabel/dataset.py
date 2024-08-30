import torch
from torch.utils.data import Dataset

class VideoMIDIDataset(Dataset):
    def __init__(self, data_pairs):
        self.data_pairs = data_pairs

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        try:
            midi_data, video_data = self.data_pairs[idx]
            video_tensor = torch.tensor(video_data, dtype=torch.float32).permute(3, 0, 1, 2)
            midi_tensor = torch.tensor(midi_data, dtype=torch.long)
            return video_tensor, midi_tensor
        except Exception as e:
            print(f"Skipping index {idx} due to error: {e}")
            return None  # Returning None to indicate a problem

