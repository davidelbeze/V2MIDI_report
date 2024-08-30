import os
import glob
import cv2
import numpy as np
import mido
import random
from midi_preprocessing import process_midi

class DataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.folders = glob.glob(os.path.join(data_dir, "batch_*"))

    def _load_midi(self, midi_path):
        return process_midi(midi_path)
    
    def _load_video(self, video_path, start_frame):
        cap = cv2.VideoCapture(video_path)
        frames = []
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if count >= start_frame and count < start_frame + 16:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (112, 112))
                frames.append(frame)
            count += 1
            if count == start_frame + 16:
                break
        cap.release()
        return np.array(frames) / 255.0

    def get_data_pairs(self, num_pairs):
        selected_folders = random.sample(self.folders, num_pairs)
        for folder in selected_folders:
            midi_path = glob.glob(os.path.join(folder, "*.mid"))[0]
            video_path = glob.glob(os.path.join(folder, "*.mp4"))[0]
            start_frame = random.randint(0, 368)
            midi_data = self._load_midi(midi_path)[start_frame:start_frame+16]
            video_data = self._load_video(video_path, start_frame)
            yield (midi_data, video_data)

# Example Usage
if __name__ == "__main__":
    loader = DataLoader("/workspace/v2midi_test/midi_videos_train")
    data_pairs = list(loader.get_data_pairs(10))  # Load 10 data pairs
    print(f"Loaded {len(data_pairs)} data pairs.")
    if data_pairs:
        print(f"First data pair shapes: MIDI {data_pairs[0][0].shape}, Video {data_pairs[0][1].shape}")
