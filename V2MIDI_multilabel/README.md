# V2MIDI Multilabel Classification

This folder contains the code for the multilabel classification approach of the V2MIDI project, which aims to synchronize visual art and music generation using AI.

## Contents

- `v2midi_train.slurm`: SLURM script for training on a GPU cluster.
- `train.py`: Main training script for the V2MIDI model.
- `model.py`: Definition of the V2MIDIModel using a pre-trained R3D-18 architecture.
- `dataset.py`: Custom dataset class for handling video-MIDI pairs.
- `data_loader.py`: Custom data loader for processing and yielding video-MIDI pairs.
- `midi_preprocessing_multiclass.py`: MIDI processing utilities for multiclass representation.
- `video2midi.py`: Script for running inference on a single video and reconstructing its whole MIDI prediction.
- `eval.py`: Script for running inference on a validation set (or its subset), reconstructing predicted MIDI files and getting accuracy metrics.
- `midi_regeneration_constant_velocity.py`: Utility for regenerating MIDI files from model outputs.

## Key Features

- Multilabel classification for 6 drum categories (including "no note").
- Uses a modified R3D-18 model pre-trained on Kinetics400.
- Processes 16-frame video segments and corresponding MIDI data.
- Includes validation during training and fixed sample evaluation.

## Usage

1. Adjust paths in the scripts to match your data locations.
2. Use `v2midi_train.slurm` to submit the training job on a SLURM-managed cluster.
3. For inference, use `video2midi.py` after modifying the paths to your model and input video.

## Dataset

This model is trained on the MIDIvideo dataset, which is hosted on Hugging Face due to its large size:

🔗 [V2MIDI Dataset on Hugging Face](https://huggingface.co/datasets/obvious-research/V2MIDI)

Before running the training scripts, ensure you have downloaded the necessary portions of the dataset. Refer to the dataset's README on Hugging Face for download and usage instructions.

## Note

This is an active research project. We are continuously working on improving the model and will be updating this repository with new results, statistics, and improvements. Stay tuned for updates!
