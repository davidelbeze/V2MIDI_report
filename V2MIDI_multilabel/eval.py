import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import numpy as np
import json
import os
from model import V2MIDIModel
from data_loader import DataLoader as CustomDataLoader
from midi_preprocessing_multiclass import print_midi_representation, process_midi
from torchinfo import summary
from midi_regeneration_constant_velocity import create_midi_from_representation

def load_model(model_path, device):
    model = V2MIDIModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    # model.eval()
    return model

def accuracy_score(truth, predictions):
    overall_accuracy = np.mean(truth == predictions)
    class_accuracies = {}
    for cls in np.unique(truth):
        class_indices = truth == cls
        class_accuracies[cls] = np.mean(predictions[class_indices] == truth[class_indices]) if np.any(class_indices) else 0.0
    return overall_accuracy, class_accuracies

def eval_batch(model, device, video_path, midi_path, data_loader, batch_folder):
    ground_truth_midi = process_midi(midi_path)
    print("\nFull Ground Truth MIDI Representation:")
    print_midi_representation(ground_truth_midi)
    full_video_predictions = []

    for start_frame in range(0, 384, 16):
        video_data_segment = data_loader._load_video(video_path, start_frame)
        if video_data_segment.shape[0] < 16:
            padding = np.zeros((16 - video_data_segment.shape[0], 112, 112, 3))
            video_data_segment = np.vstack((video_data_segment, padding))
        
        video_data_tensor = torch.tensor(video_data_segment, dtype=torch.float32).permute(3, 0, 1, 2).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(video_data_tensor)
            predicted_classes = torch.argmax(output, dim=2).cpu().numpy()[0]
            full_video_predictions.extend(predicted_classes)

    full_video_predictions = np.array(full_video_predictions)
    print("\nFull video predictions:")
    print(full_video_predictions)
    print(full_video_predictions.shape)
    output_midi_path = os.path.join(batch_folder, 'reconstructed_midi.mid')
    create_midi_from_representation(full_video_predictions, output_midi_path)
    print(f"MIDI file saved at {output_midi_path}")

    overall_accuracy, class_accuracies = accuracy_score(ground_truth_midi, full_video_predictions)
    return overall_accuracy, class_accuracies, ground_truth_midi, full_video_predictions

def eval_model(model_path, batches_folder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)
    summary(model, input_size=(1, 3, 16, 112, 112))
    data_loader = CustomDataLoader(batches_folder)

    for batch_folder in os.listdir(batches_folder):
        batch_index = batch_folder.split('_')[1]
        video_path = os.path.join(batches_folder, batch_folder, f'video_{batch_index}.mp4')
        midi_path = os.path.join(batches_folder, batch_folder, f'midi_{batch_index}.mid')

        print(f"Evaluating {batch_folder}...")
        overall_accuracy, class_accuracies, truth, predictions = eval_batch(model, device, video_path, midi_path, data_loader, os.path.join(batches_folder, batch_folder))
        print(f"Overall Accuracy: {overall_accuracy:.2f}")
        print("Per Class Accuracy:", class_accuracies)
        print(f"Ground Truth for {batch_folder}:")
        print_midi_representation(truth)
        print(f"Predictions for {batch_folder}:")
        print_midi_representation(predictions)

if __name__ == "__main__":
    model_path = "/workspace/v2midi_debug/saved_models_multiclass_2608_2/model_epoch_500.pth"
    batches_folder = "/workspace/v2midi_debug/midi_videos_validation_few"
    eval_model(model_path, batches_folder)
