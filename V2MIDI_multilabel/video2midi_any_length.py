import torch
import numpy as np
import os
import cv2
from model import V2MIDIModel
from data_loader import DataLoader as CustomDataLoader
from torchinfo import summary
from midi_regeneration_constant_velocity import create_midi_from_representation

def load_model(model_path, device):
    model = V2MIDIModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Switch to evaluation mode
    return model

def count_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count

def run_inference(model, device, video_path, data_loader):

    num_frames = count_frames(video_path)
    print(f"The video at {video_path} has {num_frames} frames.")
    predictions = []

    tail = num_frames % 16
    iterable_frames = num_frames - tail

    for start_frame in range(0, iterable_frames, 16):
        print(f"Debug: Start frame is {start_frame}.")
        video_data_segment = data_loader._load_video(video_path, start_frame)
        if video_data_segment.shape[0] < 16:
            padding = np.zeros((16 - video_data_segment.shape[0], 112, 112, 3))
            video_data_segment = np.vstack((video_data_segment, padding))

        video_data_tensor = torch.tensor(video_data_segment, dtype=torch.float32).permute(3, 0, 1, 2).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(video_data_tensor)
            predicted_classes = torch.argmax(output, dim=2).cpu().numpy()[0]
            predictions.extend(predicted_classes)

    return np.array(predictions)

def create_reconstructed_midi(predictions, output_path):
    create_midi_from_representation(predictions, output_path)
    print(f"MIDI file saved at {output_path}")

def main(model_path, video_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)
    summary(model, input_size=(1, 3, 16, 112, 112))
    data_loader = CustomDataLoader(os.path.dirname(video_path))

    predictions = run_inference(model, device, video_path, data_loader)
    output_midi_path = os.path.splitext(video_path)[0] + '_reconstructed_whole.mid'
    create_reconstructed_midi(predictions, output_midi_path)

if __name__ == "__main__":
    model_path = "/workspace/v2midi_debug/saved_models_multiclass_2608_2/model_epoch_500.pth"
    video_path = "/workspace/v2midi_debug/dolce_batch/dolce.mp4"
    main(model_path, video_path)
