import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import numpy as np
import json
import os
from model import V2MIDIModel
from data_loader import DataLoader as CustomDataLoader
from midi_preprocessing_multiclass import print_midi_representation
from torchinfo import summary
from dataset import VideoMIDIDataset
from midi_regeneration_constant_velocity import create_midi_from_representation

def load_model(model_path, device):
    model = V2MIDIModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def eval_model(model_path, fixed_video_path, save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = load_model(model_path, device)
    summary(model, input_size=(1, 3, 16, 112, 112))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    data_loader = CustomDataLoader("/lustre/fsn1/projects/rech/fkc/uhx75if/v2midi/img2img-images_clean/midi_videos_validation_one_sample")

    full_video_predictions = []  # List to store all segment predictions

    for start_frame in range(0, 368, 16):
        video_data_segment = data_loader._load_video(fixed_video_path, start_frame)
        video_data_tensor = torch.tensor(video_data_segment, dtype=torch.float32).permute(3, 0, 1, 2).unsqueeze(0).to(device)

        with torch.no_grad():
            model.eval()
            output = model(video_data_tensor)
            predicted_classes = torch.argmax(output, dim=2).cpu().numpy()
            print("\nPredicted subpart MIDI Representation:")
            print_midi_representation(predicted_classes[0])  # Assuming the output format

            full_video_predictions.append(predicted_classes[0])

    # Concatenate all predictions to get a full sequence
    full_video_predictions = np.concatenate(full_video_predictions)
    output_midi_path = os.path.join(save_path, 'full_video_midi_output.mid')
    create_midi_from_representation(full_video_predictions, output_midi_path)
    print(f"MIDI file created at {output_midi_path}")

if __name__ == "__main__":
    model_path = "/lustre/fsn1/projects/rech/fkc/uhx75if/v2midi/saved_models_multiclass_2608_2/model_epoch_200.pth" # Change to load other model
    fixed_video_path = '/lustre/fsn1/projects/rech/fkc/uhx75if/v2midi/img2img-images_clean/midi_videos_validation/batch_6551/video_6551.mp4' # Change to load other video
    save_path = "/lustre/fsn1/projects/rech/fkc/uhx75if/v2midi/output_midi_files" # Change output path for the reconstructed MIDI file
    eval_model(model_path, fixed_video_path, save_path)