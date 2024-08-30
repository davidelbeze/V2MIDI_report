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

def validate_model(model, validation_data_loader, device, validation_epoch_size):
    model.eval()
    total_loss = 0
    
    criterion = torch.nn.NLLLoss()

    validation_data_pairs = list(validation_data_loader.get_data_pairs(validation_epoch_size))
    validation_dataset = VideoMIDIDataset(validation_data_pairs)
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=True)

    print_val_sample = True

    with torch.no_grad():
        for batch in validation_loader:
            if batch is None:
                continue
            video_data, midi_data = batch
            video_data, midi_data = video_data.to(device), midi_data.to(device)
            outputs = model(video_data)

            if print_val_sample:
                validation_sample_output = torch.argmax(outputs[0], dim=1).cpu().numpy()
                validation_actual_midi = midi_data[0].cpu().numpy()
                print("\nValidation Random Sample - Actual MIDI Representation:")
                print_midi_representation(validation_actual_midi)
                print("\nValidation Random Sample - Predicted MIDI Representation:")
                print_midi_representation(validation_sample_output)

                print_val_sample = False

            loss = criterion(outputs.view(-1, 6), midi_data.view(-1))
            total_loss += loss.item()

    return total_loss / len(validation_loader)



def train_model(epoch_size, validation_data_loader, validation_epoch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    data_loader = CustomDataLoader("/lustre/fsn1/projects/rech/fkc/uhx75if/v2midi/img2img-images_clean/midi_videos_train")
    model = V2MIDIModel().to(device)
    summary(model, input_size=(1, 3, 16, 112, 112))

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.NLLLoss()
    stats = {"loss": [], "validation_loss": []}
    save_path = "/lustre/fsn1/projects/rech/fkc/uhx75if/v2midi/saved_models_multiclass_2608_2_3008_3"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Fixed validation sample setup
    fixed_video_path = '/lustre/fsn1/projects/rech/fkc/uhx75if/v2midi/img2img-images_clean/midi_videos_validation/batch_6551/video_6551.mp4'
    fixed_midi_path = '/lustre/fsn1/projects/rech/fkc/uhx75if/v2midi/img2img-images_clean/midi_videos_validation/batch_6551/midi_6551.mid'
    fixed_video_data = data_loader._load_video(fixed_video_path, 0)
    fixed_midi_data = data_loader._load_midi(fixed_midi_path)[:16]
    fixed_video_data = torch.tensor(fixed_video_data, dtype=torch.float32).permute(3, 0, 1, 2).unsqueeze(0).to(device)
    fixed_midi_data = torch.tensor(fixed_midi_data, dtype=torch.long).unsqueeze(0).to(device)

    for epoch in range(100):
        data_pairs = list(data_loader.get_data_pairs(epoch_size))
        dataset = VideoMIDIDataset(data_pairs)
        loader = DataLoader(dataset, batch_size=1, shuffle=True)

        epoch_losses = []

        print_train_sample = True

        for batch in loader:
            if batch is None:  # Check if the batch is None (i.e., faulty video)
                continue
            video_data, midi_data = batch
            video_data, midi_data = video_data.to(device), midi_data.to(device)
            optimizer.zero_grad()
            outputs = model(video_data)
            loss = criterion(outputs.view(-1, 6), midi_data.view(-1))
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

            if print_train_sample:
                with torch.no_grad():
                    sample_output = torch.argmax(outputs[0], dim=1).cpu().numpy()
                    actual_midi = midi_data[0].cpu().numpy()
                    print("\nTrain Random Sample - Actual MIDI Representation:")
                    print_midi_representation(actual_midi)
                    print("\nTrain Random Sample - Predicted MIDI Representation:")
                    print_midi_representation(sample_output)
                    print_train_sample = False


        val_loss = validate_model(model, validation_data_loader, device, validation_epoch_size)

        stats["loss"].append(np.mean(epoch_losses))
        stats["validation_loss"].append(val_loss)

        print(f"Epoch {epoch}, Training Loss: {np.mean(epoch_losses)}, Validation Loss: {val_loss}")

    torch.save(model.state_dict(), os.path.join(save_path, "model_final.pth"))
    with open(os.path.join(save_path, 'training_stats_final.json'), 'w') as f:
        json.dump(stats, f)

if __name__ == "__main__":
    our_validation_data_loader = CustomDataLoader("/lustre/fsn1/projects/rech/fkc/uhx75if/v2midi/img2img-images_clean/midi_videos_validation")
    train_model(epoch_size=1000, validation_data_loader=our_validation_data_loader, validation_epoch_size=10)
