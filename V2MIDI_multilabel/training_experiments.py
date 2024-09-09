import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import json
import os
import time
from model import V2MIDIModel
from data_loader import DataLoader as CustomDataLoader
from dataset import VideoMIDIDataset
from sklearn.metrics import f1_score, accuracy_score
from torchinfo import summary

# Function to compute metrics for per-class accuracy and F1 score
def compute_metrics(predictions, targets, num_classes):
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    for i in range(num_classes):
        class_correct[i] += (predictions == i).sum().item()
        class_total[i] += (targets == i).sum().item()
    overall_accuracy = accuracy_score(targets, predictions)
    f1 = f1_score(targets, predictions, average='weighted')
    per_class_accuracy = [class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(num_classes)]
    return overall_accuracy, per_class_accuracy, f1

# Validation loop
def validate_model(model, validation_loader, device, criterion):
    model.eval()  # Switch to evaluation mode
    total_loss = 0
    total_batches = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():  # Disable gradient calculations during validation
        for batch in validation_loader:
            if batch is None:
                continue
            video_data, midi_data = batch
            video_data, midi_data = video_data.to(device), midi_data.to(device)

            outputs = model(video_data)
            loss = criterion(outputs.view(-1, 6), midi_data.view(-1))
            total_loss += loss.item()
            total_batches += 1

            _, predicted = torch.max(outputs, 2)
            all_predictions.extend(predicted.view(-1).cpu().numpy())
            all_targets.extend(midi_data.view(-1).cpu().numpy())

    overall_accuracy, per_class_accuracy, f1 = compute_metrics(np.array(all_predictions), np.array(all_targets), 6)
    return total_loss / total_batches if total_batches > 0 else 0, overall_accuracy, per_class_accuracy, f1

# Training loop
def train_model(params):
    epoch_size = params['epoch_size']
    batch_size = params['batch_size']
    lr = params['lr']
    weight_decay = params.get('weight_decay', 0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}, Batch Size: {batch_size}, LR: {lr}, Weight Decay: {weight_decay}")

    data_loader = CustomDataLoader("/lustre/fsn1/projects/rech/fkc/uhx75if/v2midi/img2img-images_clean/midi_videos_train")
    model = V2MIDIModel().to(device)
    summary(model, input_size=(1, 3, 16, 112, 112))

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.NLLLoss()

    # Prepare to save statistics
    stats = {"epoch_times": [], "loss": [], "validation_loss": [], "training_accuracy": [], "validation_accuracy": [],
             "training_per_class_accuracy": [], "validation_per_class_accuracy": [], "training_f1": [], "validation_f1": []}
    save_path = f"/lustre/fsn1/projects/rech/fkc/uhx75if/v2midi/saved_models_run_lr_{lr}_batch_{batch_size}_wd_{weight_decay}"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Validation dataset
    validation_data_loader = CustomDataLoader("/lustre/fsn1/projects/rech/fkc/uhx75if/v2midi/img2img-images_clean/midi_videos_validation")
    validation_dataset = VideoMIDIDataset(list(validation_data_loader.get_data_pairs(128)))
    validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=True)

    for epoch in range(100):  # Adjust epoch count
        start_time = time.time()

        # Load training data
        data_pairs = list(data_loader.get_data_pairs(epoch_size))
        dataset = VideoMIDIDataset(data_pairs)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model.train()
        epoch_losses = []
        all_predictions = []
        all_targets = []

        for batch in loader:
            if batch is None:
                continue
            video_data, midi_data = batch
            video_data, midi_data = video_data.to(device), midi_data.to(device)

            optimizer.zero_grad()
            outputs = model(video_data)
            loss = criterion(outputs.view(-1, 6), midi_data.view(-1))
            loss.backward()

            # Calculate gradient norm
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

            optimizer.step()

            epoch_losses.append(loss.item())
            _, predicted = torch.max(outputs, 2)
            all_predictions.extend(predicted.view(-1).cpu().numpy())
            all_targets.extend(midi_data.view(-1).cpu().numpy())

        # Calculate training metrics
        training_overall_accuracy, training_per_class_accuracy, training_f1 = compute_metrics(np.array(all_predictions), np.array(all_targets), 6)

        # Validation loop
        val_loss, validation_overall_accuracy, validation_per_class_accuracy, validation_f1 = validate_model(model, validation_loader, device, criterion)

        epoch_time = time.time() - start_time
        stats["epoch_times"].append(epoch_time)
        stats["loss"].append(np.mean(epoch_losses))
        stats["validation_loss"].append(val_loss)
        stats["training_accuracy"].append(training_overall_accuracy)
        stats["validation_accuracy"].append(validation_overall_accuracy)
        stats["training_per_class_accuracy"].append(training_per_class_accuracy)
        stats["validation_per_class_accuracy"].append(validation_per_class_accuracy)
        stats["training_f1"].append(training_f1)
        stats["validation_f1"].append(validation_f1)

        print(f"Epoch {epoch}, Time: {epoch_time:.2f}s, Loss: {np.mean(epoch_losses):.4f}, Val Loss: {val_loss:.4f}, Train Acc: {training_overall_accuracy:.4f}, Val Acc: {validation_overall_accuracy:.4f}")

        if epoch % 50 == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{epoch}.pth"))

    torch.save(model.state_dict(), os.path.join(save_path, "model_final.pth"))
    with open(os.path.join(save_path, 'training_stats.json'), 'w') as f:
        json.dump(stats, f)

# Function to launch multiple runs with different parameters
def launch_experiments():
    param_sets = [
        {"epoch_size": 1024, "batch_size": 16, "lr": 0.001, "weight_decay": 0.0},
        {"epoch_size": 1024, "batch_size": 32, "lr": 0.001, "weight_decay": 0.01},
        {"epoch_size": 2048, "batch_size": 16, "lr": 0.0005, "weight_decay": 0.0},
        {"epoch_size": 2048, "batch_size": 32, "lr": 0.0005, "weight_decay": 0.01}
    ]

    for params in param_sets:
        print(f"Starting training with params: {params}")
        train_model(params)

if __name__ == "__main__":
    launch_experiments()
