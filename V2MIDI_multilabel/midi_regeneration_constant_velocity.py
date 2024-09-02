import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage
import numpy as np
from midi_preprocessing_multiclass import process_midi

# Constants
RELEVANT_NOTES = [0, 36, 38, 42, 44, 46]  # Note categories

index_to_note = {0: 0, 1: 36, 2: 38, 3: 42, 4: 44, 5: 46}

BPM = 120  # Standard BPM
TICKS_PER_BEAT = 480  # Standard resolution
SECONDS_PER_MINUTE = 60
FPS = 24
FRAME_DURATION_TICKS = TICKS_PER_BEAT * (BPM / FPS / SECONDS_PER_MINUTE)  # Frame duration in MIDI ticks

def create_midi_from_representation(midi_representation, output_path):
    # Create a new MIDI file with a single track
    midi = MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    track = MidiTrack()
    midi.tracks.append(track)

    # Set tempo (500000 microseconds per beat = 120 BPM)
    track.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(BPM)))

    # Initialize time accumulator
    current_time = 0

    # Process the representation array
    for frame in midi_representation:
        velocity = 50
        note_category = frame
        note = int(note_category)

        if note == 0:
            current_time += int(FRAME_DURATION_TICKS)

        if note != 0:
            # Add note on at the current time
            track.append(Message('note_on', channel=9, note=index_to_note[note], velocity=int(velocity), time=current_time))
            # Note lasts exactly one frame
            track.append(Message('note_off', channel=9, note=index_to_note[note], velocity=int(velocity), time=int(FRAME_DURATION_TICKS)))
            current_time = 0

    # Save the MIDI file
    midi.save(output_path)


if __name__ == "__main__":

    midi_path = "midi_1.mid"
    output_path = "reconstructed.mid"

    midi_representation = process_midi(midi_path)
    
    create_midi_from_representation(midi_representation, output_path)