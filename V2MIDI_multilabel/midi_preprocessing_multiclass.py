import mido
import numpy as np

# Define the relevant note categories
RELEVANT_NOTES = [0, 36, 38, 42, 44, 46]

note_to_index = {0: 0, 36: 1, 38: 2, 42: 3, 44: 4, 46: 5}

def process_midi(midi_path, output_shape=(384,), fps=24):
    midi = mido.MidiFile(midi_path)
    
    # Initialize the multiclass representation array
    midi_representation = np.zeros(output_shape, dtype=int)
    
    # Define total frames for the given duration (16 seconds)
    total_frames = output_shape[0]
    
    # Calculate the number of ticks per frame based on the initial tempo
    initial_tempo = 500000  # Default tempo is 120 BPM
    ticks_per_frame = (initial_tempo / 1_000_000) / fps * midi.ticks_per_beat
    
    # Track the cumulative time in ticks
    cumulative_ticks = 0
    
    for track in midi.tracks:
        for msg in track:
            if not msg.is_meta and msg.channel == 9:
                cumulative_ticks += msg.time
                # Convert cumulative ticks to seconds
                current_time_seconds = mido.tick2second(cumulative_ticks, midi.ticks_per_beat, initial_tempo)
                # Determine the corresponding frame
                frame_index = int(current_time_seconds * fps)
                if frame_index < total_frames and msg.type == 'note_on' and msg.note in RELEVANT_NOTES and msg.velocity > 0:
                    midi_representation[frame_index] = note_to_index[msg.note]

    return midi_representation

def print_midi_representation(midi_representation):
    num_frames = midi_representation.shape[0]
    
    for frame in range(num_frames):
        if midi_representation[frame] == 0:
            print(f"Frame {frame}: No Note Played")
        else:
            print(f"Frame {frame}: Note {midi_representation[frame]} Played")

if __name__ == "__main__":
    midi_path = "midi_1.mid"
    midi_rep = process_midi(midi_path)
    print(midi_rep.shape)
    print_midi_representation(midi_rep)
