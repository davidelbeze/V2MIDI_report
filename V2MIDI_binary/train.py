import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import numpy as np
import json
import os
import time
from model import V2MIDIModel
from data_loader import DataLoader as CustomDataLoader
from midi_preprocessing import print_midi_representation
from torchinfo import summary
from dataset import VideoMIDIDataset

def validate_model(model, validation_loader, device):
    model.eval()
    total_loss, total_accuracy = 0, 0
    total_positives, correct_positives = 0, 0
    total_negatives, correct_negatives = 0, 0
    criterion = torch.nn.BCELoss()
    with torch.no_grad():
        for video_data, midi_data in validation_loader:
            video_data, midi_data = video_data.to(device), midi_data.to(device)
            outputs = model(video_data)
            loss = criterion(outputs, midi_data)
            total_loss += loss.item()
            predictions = outputs.round()
            total_accuracy += (predictions == midi_data).float().mean().item()

            # Per-class accuracy calculation
            correct_positives += ((predictions == 1) & (midi_data == 1)).sum().item()
            total_positives += (midi_data == 1).sum().item()
            correct_negatives += ((predictions == 0) & (midi_data == 0)).sum().item()
            total_negatives += (midi_data == 0).sum().item()

    model.train()
    val_accuracy_pos = correct_positives / total_positives if total_positives > 0 else 0
    val_accuracy_neg = correct_negatives / total_negatives if total_negatives > 0 else 0
    return total_loss / len(validation_loader), total_accuracy / len(validation_loader), val_accuracy_pos, val_accuracy_neg

def train_model(epoch_size, validation_data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    data_loader = CustomDataLoader("/lustre/fsn1/projects/rech/fkc/uhx75if/v2midi/img2img-images_clean/midi_videos_train")
    model = V2MIDIModel().to(device)
    summary(model, input_size=(1, 3, 16, 112, 112))

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    stats = {"loss": [], "validation_loss": [], "validation_accuracy": [], "validation_accuracy_pos": [], "validation_accuracy_neg": [], "epoch_duration": [], "learning_rate": [], "gradient_norms": [], "memory_usage": [], "data_loading_time": []}
    save_path = "/lustre/fsn1/projects/rech/fkc/uhx75if/v2midi/saved_models_v2midi_binary_2608"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for epoch in range(1000):
        start_time = time.time()
        data_loading_start = time.time()
        data_pairs = list(data_loader.get_data_pairs(epoch_size))
        data_loading_time = time.time() - data_loading_start
        dataset = VideoMIDIDataset(data_pairs)
        loader = DataLoader(dataset, batch_size=1, shuffle=True)

        epoch_losses = []
        total_grad_norm = 0
        for video_data, midi_data in loader:
            video_data, midi_data = video_data.to(device), midi_data.to(device)
            optimizer.zero_grad()
            outputs = model(video_data)
            loss = torch.nn.BCELoss()(outputs, midi_data)
            loss.backward()
            total_grad_norm += sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
            optimizer.step()
            epoch_losses.append(loss.item())

        epoch_duration = time.time() - start_time
        val_loss, val_accuracy, val_accuracy_pos, val_accuracy_neg = validate_model(model, validation_data_loader, device)
        memory_usage = torch.cuda.memory_allocated(device)
        gradient_norms = total_grad_norm / len(loader)

        mean_loss = np.mean(epoch_losses)
        stats["loss"].append(mean_loss)
        stats["validation_loss"].append(val_loss)
        stats["validation_accuracy"].append(val_accuracy)
        stats["validation_accuracy_pos"].append(val_accuracy_pos)
        stats["validation_accuracy_neg"].append(val_accuracy_neg)
        stats["epoch_duration"].append(epoch_duration)
        stats["learning_rate"].append(optimizer.param_groups[0]['lr'])
        stats["gradient_norms"].append(gradient_norms)
        stats["memory_usage"].append(memory_usage)
        stats["data_loading_time"].append(data_loading_time)
        
        print(f"\nEpoch {epoch}, Loss: {mean_loss}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}, Pos Accuracy: {val_accuracy_pos}, Neg Accuracy: {val_accuracy_neg}, Grad Norm: {gradient_norms}, Epoch Time: {epoch_duration}, Memory Usage: {memory_usage}, Data Loading Time: {data_loading_time}")
        with torch.no_grad():
            sample_output = outputs[0].cpu().numpy().round()
            actual_midi = midi_data[0].cpu().numpy()
            print("\nActual MIDI Representation:")
            print_midi_representation(actual_midi)
            print("\nPredicted MIDI Representation:")
            print_midi_representation(sample_output)

        if epoch % 100 == 0:  # Save every 100 epochs
            torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{epoch}.pth"))

    torch.save(model.state_dict(), os.path.join(save_path, "model_final.pth"))
    with open(os.path.join(save_path, 'training_stats.json'), 'w') as f:
        json.dump(stats, f)

if __name__ == "__main__":
    validation_data_loader = CustomDataLoader("/lustre/fsn1/projects/rech/fkc/uhx75if/v2midi/img2img-images_clean/midi_videos_validation")
    validation_dataset = VideoMIDIDataset(list(validation_data_loader.get_data_pairs(10)))  # Assuming validation set size
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=True)
    train_model(epoch_size=100, validation_data_loader=validation_loader)
