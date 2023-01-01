import pretty_midi
from src.util import data_handling as dh
import argparse
import sys
import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    # Command-line argument parsing for data path
    parser = argparse.ArgumentParser(description='Generate accompaniment for given MIDI melody')
    parser.add_argument('--model_path', action='store',
                        help='Path to the model')
    parser.add_argument('--input', action='store',
                        help='Path to the input .mid file')
    parser.add_argument('--output', action='store',
                        help='Path to the output .mid file')
    parser.add_argument('--roll', action='store_true',
                        help='create a piano roll png for the output generation')
    parameters = vars(parser.parse_args(sys.argv[1:]))

    # Load model
    model = tf.keras.models.load_model(parameters['model_path'])

    sequence_seconds = 15
    start_at_seconds = 0
    skip_empty_intro = True

    # Parse input midi file using PrettyMidi and collect list of input sequence windows
    try:
        melody_pm = pretty_midi.PrettyMIDI(parameters['input'])
        melody_seq = dh.midi_to_activation_sequence(melody_pm, sequence_seconds, instrument_track='MELODY',
                                               vocab_size=128, sample_frequency=60,
                                               offset=start_at_seconds, skip_leading_space=skip_empty_intro)
    except Exception as e:
        print(f'Unable to parse input file ({e}), exiting...')
        sys.exit(1)

    melody_seq = np.expand_dims(melody_seq, axis=0)
    generated_seq = model.predict(melody_seq)

    pm = dh.add_accompaniment_track(melody_pm, generated_seq,
                                    parameters['output'],
                                    velocity=100,
                                    instrument_num=33,
                                    concat_sequential=True)

    if parameters['roll']:
        dh.plot_piano_roll(generated_seq)
