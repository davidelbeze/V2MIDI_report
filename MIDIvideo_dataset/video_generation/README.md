# Video Generation for V2MIDI

This folder contains the code for large-scale video generation in the V2MIDI project. It turns MIDI files and their corresponding Parseq/Deforum configurations into synchronized videos.

## Contents

- `v2midi_dataset.slurm`: SLURM job script for running the video generation on a supercomputer
- `main_video_generation.py`: Main Python script for video generation
- `new_deforum_settings.txt`: Default settings for Deforum video generation

## How It Works

1. **Job Scheduling** (`v2midi_dataset.slurm`):
   - Sets up the environment and resources for running on a supercomputer
   - Launches the main Python script with necessary arguments

2. **Video Generation** (`main_video_generation.py`):
   - Distributes work across multiple GPUs
   - Processes Parseq configurations and generates videos using Stable Diffusion and Deforum
   - Handles job queuing, error recovery, and output organization

3. **Deforum Settings** (`new_deforum_settings.txt`):
   - Provides default parameters for video generation
   - Includes settings for resolution, animation mode, and various visual effects

## Key Features

- Parallel processing using multiple GPUs
- Robust error handling and job recovery
- Integration with Stable Diffusion and Deforum for AI-generated videos
- Customizable video settings

## Usage

1. Ensure you have access to a multi-GPU system or supercomputer
2. Update paths and settings in `v2midi_dataset.slurm` and `main_video_generation.py`
3. Submit the SLURM job

## Customization

- Modify GPU allocation and job array settings in `v2midi_dataset.slurm`
- Adjust video generation parameters in `new_deforum_settings.txt`
- Customize error handling and job distribution in `main_video_generation.py`

## Output

The script generates:

- MP4 video files synchronized with input MIDI
- Organized output structure matching the input dataset

## Note

This code is designed for large-scale processing and requires significant computational resources. It's part of the V2MIDI project, creating AI-generated videos synchronized with MIDI music.

## Requirements

- SLURM-enabled cluster or supercomputer
- PyTorch with CUDA support
- Stable Diffusion and Deforum installations
- Access to a dataset of MIDI and Parseq configuration pairs
