# V2MIDI Binary Classification Model

This folder contains the implementation of the binary classification model for the V2MIDI project, which aims to synchronize visual art and music generation using AI.

## Contents

- `data_loader.py`: Custom data loader for processing MIDI and video files.
- `dataset.py`: PyTorch dataset class for handling video-MIDI pairs.
- `midi_preprocessing.py`: MIDI file processing and representation.
- `model.py`: Implementation of the V2MIDIModel using R3D-18 architecture.
- `train.py`: Training script for the model.
- `v2midi.slurm`: SLURM job script for running on a cluster.

## Key Features

- Uses a pre-trained R3D-18 model for video feature extraction.
- Binary classification: predicts note presence/absence for each frame.
- Handles MIDI files and synchronized video data.
- Includes validation and performance tracking.

## Performance

After 931 epochs:
- Training Loss: 0.1079
- Validation Loss: 0.3539
- Validation Accuracy: 90%
- Positive Class Accuracy: 73.33%
- Negative Class Accuracy: 93.85%

This binary model served as a proof of concept, demonstrating the viability of our approach to audio-visual synchronization. Its success provided the confidence to proceed with the development of a more sophisticated multilabel V2MIDI model.

## Note

Adjust paths and parameters in the scripts as needed for your environment.

## Dataset

This model is trained on the MIDIvideo dataset, which is hosted on Hugging Face due to its large size:

🔗 [V2MIDI Dataset on Hugging Face](https://huggingface.co/datasets/obvious-research/V2MIDI)

Before running the training scripts, ensure you have downloaded the necessary portions of the dataset. Refer to the dataset's README on Hugging Face for download and usage instructions.

## Usage

To train the model:

```bash
sbatch v2midi.slurm
