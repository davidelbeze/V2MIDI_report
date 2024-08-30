import json
import mido
import os
import random

def midi_to_frame_events(midi_file_path, fps):
    midi = mido.MidiFile(midi_file_path)
    
    # Define total frames for 16 seconds
    total_frames = 16 * fps
    
    # Initialize a list to hold the events for each frame
    frame_events = [[] for _ in range(total_frames)]
    
    # Calculate the number of ticks per frame based on the initial tempo
    initial_tempo = 500000  # Default tempo is 120 BPM
    ticks_per_frame = (initial_tempo / 1_000_000) / fps * midi.ticks_per_beat
    
    # Track the cumulative time in ticks
    cumulative_ticks = 0
    
    for track in midi.tracks:
        for msg in track:
            if not msg.is_meta:
                cumulative_ticks += msg.time
                # Convert cumulative ticks to seconds
                current_time_seconds = mido.tick2second(cumulative_ticks, midi.ticks_per_beat, initial_tempo)
                # Determine the corresponding frame
                frame_index = int(current_time_seconds * fps)
                if frame_index < total_frames and msg.type in ['note_on', 'note_off']:
                    frame_events[frame_index].append(msg)
                elif frame_index >= total_frames:
                    break
    
    return frame_events, total_frames

def get_random_prompt(prompts_directory):
    theme_directories = [os.path.join(prompts_directory, d) for d in os.listdir(prompts_directory) if os.path.isdir(os.path.join(prompts_directory, d))]
    selected_theme = random.choice(theme_directories)
    prompt_files = [os.path.join(selected_theme, f) for f in os.listdir(selected_theme) if f.startswith('prompt_')]
    selected_prompt_file = random.choice(prompt_files)
    with open(selected_prompt_file, 'r') as file:
        prompt = file.read().strip()
    return prompt

def map_velocity_to_range(velocity, min_value, max_value):
    if not (1 <= velocity <= 127):
        raise ValueError("Velocity must be between 1 and 127")

    # Normalize the velocity to a 0 to 1 range
    normalized_velocity = (velocity - 1) / 126.0
    
    # Map the normalized velocity to the specified range
    mapped_value = min_value + normalized_velocity * (max_value - min_value)
    
    return mapped_value

def set_rotation_frames(rotation_values, frame_events, total_frames, rotation_note):
    current_rotation = 0.0

    # Process the rotation_note events and update rotation_values
    for frame_index, events in enumerate(frame_events):
        for event in events:
            if event.type == 'note_on' and event.note == rotation_note and event.channel == 9:
                increment = map_velocity_to_range(event.velocity, 0, 1)
                for i in range(5):
                    if frame_index + i < total_frames:
                        current_rotation += increment
                        rotation_values[frame_index + i] = current_rotation

    # Fill the rotation_values array to ensure a continuous increasing function
    for i in range(1, total_frames):
        if rotation_values[i] == 0.0:
            rotation_values[i] = rotation_values[i - 1]

def set_rotation_z_frames(rotation_z_values, frame_events, total_frames, rotation_z_note):
    # Process the rotation_z_note events and update rotation_z_values
    for frame_index, events in enumerate(frame_events):
        for event in events:
            if event.type == 'note_on' and event.note == rotation_z_note and event.channel == 9:
                for i in range(10):
                    if frame_index + i < total_frames:
                        rotation_z_values[frame_index + i] = map_velocity_to_range(event.velocity, 0, 10)

    # Ensure rotation_z_values return to 0 after the 10 frames
    for frame_index, events in enumerate(frame_events):
        for event in events:
            if event.type == 'note_on' and event.note == rotation_z_note and event.channel == 9:
                reset_frame = frame_index + 10
                if reset_frame < total_frames:
                    rotation_z_values[reset_frame] = 0

def set_strength_frames(strength_values, frame_events, total_frames, fps, strength_note, strength_kick, strength_default, kick_duration_seconds):
    for frame_index, events in enumerate(frame_events):
        for event in events:
            if event.type == 'note_on' and event.note == strength_note and event.channel == 9:
                strength_values[frame_index] = strength_kick
                reset_frame = frame_index + int(kick_duration_seconds * fps)
                if reset_frame < total_frames:
                    strength_values[reset_frame] = strength_default

def set_translation_z_frames(translation_z_values, frame_events, total_frames, translation_z_note):
    current_translation_z = 1.0

    # Process the translation_z_note events and update translation_z_values
    for frame_index, events in enumerate(frame_events):
        for event in events:
            if event.type == 'note_on' and event.note == translation_z_note and event.channel == 9:
                increment = (event.velocity) / 5
                for i in range(5):
                    if frame_index + i < total_frames:
                        current_translation_z += increment
                        translation_z_values[frame_index + i] = current_translation_z

    # Fill the translation_z_values array to ensure a continuous increasing function
    for i in range(1, total_frames):
        if translation_z_values[i] == 1.0:
            translation_z_values[i] = translation_z_values[i - 1]

def midi_to_parseq_config(
    frame_events, total_frames, fps,
    rotation_3d_x_note=None, rotation_3d_y_note=None, strength_note=None,
    translation_z_note=None, rotation_3d_z_note=None,
    strength_default=0.8, strength_kick=0.3, kick_duration_seconds=0.1,
    prompts_directory='video2midi_prompts'):

    parseq_config = {
        "meta": {
            "docName": "MIDI to Parseq"
        },
        "prompts": {
            "positive": get_random_prompt(prompts_directory),
            "negative": "watermark, logo, text, signature, copyright, writing, letters, low quality, artefacts, cropped, bad art, poorly drawn, lowres, simple, pixelated, grain, noise, blurry, cartoon, computer game, video game, painting, drawing, sketch, disfigured, deformed, mutant, suit, formal, nsfw, nude."
        },
        "options": {
            "input_fps": "",
            "bpm": 120,
            "output_fps": 24,
            "cc_window_width": 0,
            "cc_window_slide_rate": 1,
            "cc_use_input": False
        },
        "displayFields": [],
        "keyframes": []
    }

    keyframes = parseq_config["keyframes"]

    # Initialize value arrays
    rotation_3d_x_values = [0.0] * total_frames
    rotation_3d_y_values = [0.0] * total_frames
    rotation_3d_z_values = [0.0] * total_frames
    translation_z_values = [1.0] * total_frames
    strength_values = [strength_default] * total_frames

    # Set frames for each parameter
    if rotation_3d_x_note:
        set_rotation_frames(rotation_3d_x_values, frame_events, total_frames, rotation_3d_x_note)
        parseq_config["displayFields"].append("rotation_3d_x")
    if rotation_3d_y_note:
        set_rotation_frames(rotation_3d_y_values, frame_events, total_frames, rotation_3d_y_note)
        parseq_config["displayFields"].append("rotation_3d_y")
    if rotation_3d_z_note:
        set_rotation_z_frames(rotation_3d_z_values, frame_events, total_frames, rotation_3d_z_note)
        parseq_config["displayFields"].append("rotation_3d_z")
    if strength_note:
        set_strength_frames(strength_values, frame_events, total_frames, fps, strength_note, strength_kick, strength_default, kick_duration_seconds)
        parseq_config["displayFields"].append("strength")
    if translation_z_note:
        set_translation_z_frames(translation_z_values, frame_events, total_frames, translation_z_note)
        parseq_config["displayFields"].append("translation_z")

    # Create keyframes with all values grouped
    for i in range(total_frames):
        keyframe = {"frame": i}
        if rotation_3d_x_note:
            keyframe["rotation_3d_x"] = rotation_3d_x_values[i]
        if rotation_3d_y_note:
            keyframe["rotation_3d_y"] = rotation_3d_y_values[i]
        if rotation_3d_z_note:
            keyframe["rotation_3d_z"] = rotation_3d_z_values[i]
        if translation_z_note:
            keyframe["translation_z"] = translation_z_values[i]
        if strength_note:
            keyframe["strength"] = strength_values[i]
        keyframes.append(keyframe)

    parseq_config["total_frames"] = total_frames
    
    return parseq_config

