# MIDIvideo_dataset

This folder contains the core components and examples of the MIDIvideo dataset creation process for the V2MIDI project. The MIDIvideo dataset pairs MIDI files with AI-generated synchronized videos, creating a unique resource for audio-visual AI research.

## Folder Structure

- `MIDI2ParseqDeforum/`: Code for converting MIDI files to Parseq/Deforum configurations
- `video_generation/`: Scripts for large-scale video generation using the configurations
- `examples/`: Sample videos and corresponding MIDI audio exports in various resolutions and styles

## Key Components

1. **MIDI2ParseqDeforum**
   - Processes MIDI files and creates corresponding Parseq/Deforum configurations
   - Maps MIDI events to visual parameters
   - Generates randomized visual prompts

2. **video_generation**
   - Handles large-scale video generation using Stable Diffusion and Deforum
   - Utilizes multi-GPU processing for efficiency
   - Manages job scheduling and error handling

3. **examples**
   - Showcases various video outputs with different resolutions and generation models
   - Includes MP3 exports of corresponding MIDI files for easy playback

## Usage

1. Start with the MIDI2ParseqDeforum process to create configurations
2. Use the video_generation scripts to produce synchronized videos
3. Refer to the examples folder for sample outputs and inspiration

## Note on Dataset Specifications

While the examples folder contains videos of various resolutions and styles, the final MIDIvideo dataset used for V2MIDI model training is standardized at 256x256 resolution.

## Getting Started

For detailed instructions on each component, please refer to the README files within the MIDI2ParseqDeforum and video_generation folders.

## Requirements

- Python 3.7+
- PyTorch with CUDA support
- Stable Diffusion and Deforum installations
- Access to multi-GPU system or supercomputer for large-scale generation
