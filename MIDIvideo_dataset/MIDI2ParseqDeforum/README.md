# MIDI2ParseqDeforum

## Overview

This folder contains the core scripts for transforming MIDI files into Parseq and Deforum configurations, a crucial step in the V2MIDI project. These configurations are used to generate synchronized videos for each MIDI file, bridging the gap between audio and visual content creation.

## Contents

- `MIDI_to_parseq.py`: Converts MIDI files to Parseq configurations
- `parseq_to_rendered.py`: Transforms Parseq configurations into Deforum-compatible formats
- `dataset_creation.py`: Orchestrates the entire process of creating the dataset

## How It Works

1. **MIDI Processing** (`MIDI_to_parseq.py`):
   - Reads MIDI files and extracts frame events
   - Maps MIDI notes to visual parameters (e.g., rotation, strength, translation)
   - Generates Parseq configurations with randomized visual effects

2. **Parseq to Deforum Conversion** (`parseq_to_rendered.py`):
   - Converts Parseq configurations to Deforum-compatible format
   - Calculates keyframe values, deltas, and percentages
   - Integrates visual prompts for AI-generated content

3. **Dataset Creation** (`dataset_creation.py`):
   - Processes all MIDI files in a specified folder
   - Generates Parseq and Deforum configurations for each MIDI file
   - Organizes outputs into a structured dataset

## Key Features

- Focuses on house music drum patterns
- Maps 5 key drum instruments to specific visual effects:
  - Hi-Hat Open (MIDI Note 46) → rotation_3d_x
  - Pedal Hi-Hat (MIDI Note 44) → rotation_3d_y
  - Kick Drum (MIDI Note 36) → strength
  - Snare Drum (MIDI Note 38) → translation_z
  - Closed Hi-Hat (MIDI Note 42) → rotation_3d_z
- Incorporates randomized visual prompts for diverse AI-generated content
- Handles standardized 16-second MIDI sequences
- Creates configurations for 384 frames (16 seconds at 24 fps)

## Usage

1. Place your MIDI files in a designated folder (e.g., 'House_MIDI_16s_2')
2. Set up a directory for visual prompts (e.g., 'video2midi_prompts')
3. Update paths in `dataset_creation.py`
4. Run the dataset creation script


## Customization

You can customize various aspects of the configuration generation:

- In `MIDI_to_parseq.py`:
  - Adjust MIDI note to visual parameter mappings
  - Add new handled visual effects or carefully modify ranges
  - Change the duration of effects (e.g., kick duration)
 
- In `dataset_creation.py`:
  - Adjust the fps (default is 24)
  - Modify the randomization ranges for certain parameters


## Output

The script creates a structured dataset with the following for each MIDI file:

- Original MIDI file
- Parseq configuration JSON file
- Deforum-rendered configuration JSON file

The output is organized in this structure:

midi_parseq_dataset/
    midi_parseq_rendered_1/
        midi_1.mid
        midi_1_parseq_config.json
        midi_1_parseq_rendered.json
    midi_parseq_rendered_2/
        midi_2.mid
        midi_2_parseq_config.json
        midi_2_parseq_rendered.json
    ...

This output is ready for the video generation phase of the V2MIDI project.

## Note
This code is part of the larger V2MIDI project, which aims to create synchronized audio-visual content using AI. The configurations generated here serve as input for the subsequent video generation process. The visual prompts and AI-generated content aspects rely on external tools and resources not included in this repository.
