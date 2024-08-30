import os
import random
import json
import shutil
from mido import MidiFile
from MIDI_to_parseq import midi_to_frame_events, midi_to_parseq_config
from parseq_to_rendered import save_converted_config

def generate_parseq_configs_for_midi(midi_file_path, prompts_directory, output_dir, fps=24):
    # Extract MIDI file name without extension
    midi_file_name = os.path.basename(midi_file_path).split('.')[0]

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate frame events and total frames
    frame_events, total_frames = midi_to_frame_events(midi_file_path, fps)
    
    # Generate 1 Parseq configuration for the MIDI file
    parseq_config = midi_to_parseq_config(
        frame_events,
        total_frames,
        fps,
        rotation_3d_x_note=46,  # Hi-Hat Open for rotation_3d_x
        rotation_3d_y_note=44,  # Pedal Hi-Hat for rotation_3d_y
        strength_note=36,       # Kick for strength
        translation_z_note=38,  # Snare for translation_z
        rotation_3d_z_note=42,  # Closed Hi-Hat for rotation_3d_z
        strength_default=random.uniform(0.8, 0.9),
        strength_kick=random.uniform(0.25, 0.5),
        kick_duration_seconds=random.uniform(0.05, 0.15),
        prompts_directory=prompts_directory
    )
    
    # Save Parseq config to JSON
    parseq_config_path = os.path.join(output_dir, f"{midi_file_name}_parseq_config.json")
    with open(parseq_config_path, 'w') as f:
        json.dump(parseq_config, f, indent=4)
    
    # Render the Parseq config for Deforum
    rendered_config_path = os.path.join(output_dir, f"{midi_file_name}_parseq_rendered.json")
    save_converted_config(parseq_config_path, rendered_config_path)
    
    # Copy the MIDI file to the output directory
    shutil.copy(midi_file_path, output_dir)

def process_all_midi_files(midi_folder, prompts_directory, output_base_dir):
    # Ensure the output base directory exists
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)
    
    # Process each MIDI file in the midi_folder
    for i, midi_file in enumerate(os.listdir(midi_folder), 1):
        if midi_file.endswith('.mid') or midi_file.endswith('.midi'):
            midi_file_path = os.path.join(midi_folder, midi_file)
            output_dir = os.path.join(output_base_dir, f"midi_parseq_rendered_{i}")
            generate_parseq_configs_for_midi(midi_file_path, prompts_directory, output_dir)

# Paths
base_path = '/Users/dieny/Desktop/1706_newTests'
midi_folder = os.path.join(base_path, 'House_MIDI_16s_2')
prompts_directory = os.path.join(base_path, 'video2midi_prompts')
output_base_dir = os.path.join(base_path, 'midi_parseq_dataset')

# Process all MIDI files
process_all_midi_files(midi_folder, prompts_directory, output_base_dir)

print("All MIDI files processed and Parseq configurations generated successfully.")
