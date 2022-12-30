import pretty_midi
from src.util import data_handling
import argparse
import sys
import tensorflow as tf
import numpy as np


if __name__ == '__main__':
    # Command-line argument parsing for data path
    parser = argparse.ArgumentParser(description='Transcribe Audio file to MIDI file')
    parser.add_argument('model_path', action='store',
                        help='Path to the model')
    parser.add_argument('input', action='store',
                        help='Path to the input .mid file')
    parser.add_argument('output', action='store',
                        help='Path to the output .mid file')
    parser.add_argument('--roll', action='store_true',
                        help='create a piano roll png for the output generation')
    parameters = vars(parser.parse_args(sys.argv[1:]))

    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # Load model
    model = tf.keras.models.load_model(parameters['model_path'])

    # Parse input midi file -> PrettyMidi
    try:
        melody = pretty_midi.PrettyMIDI(parameters['input'])
    except:
        print('Unable to parse input file, exiting...')
        sys.exit(1)

    generated_sequence = model(melody)

    pm = data_handling.add_accompaniment_track(pm=melody,
                                               accomp_notes=generated_sequence,
                                               out_file='generated.mid',
                                               initial_tempo=120,
                                               melody_instrument=0,
                                               accomp_instrument=32,
                                               concat_sequential=True)

    if parameters['roll']:
        data_handling.plot_piano_roll(generated_sequence)
