import pretty_midi
from src.util import data_handling as dh
import argparse
import sys
import tensorflow as tf


if __name__ == '__main__':
    # Command-line argument parsing for data path
    parser = argparse.ArgumentParser(description='Generate accompaniment for given MIDI melody')
    parser.add_argument('model_path', action='store',
                        help='Path to the model')
    parser.add_argument('input', action='store',
                        help='Path to the input .mid file')
    parser.add_argument('output', action='store',
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
        melody = pretty_midi.PrettyMIDI(parameters['input'])
        melody_seq = dh.import_midi_sequence(parameters['input'], sequence_seconds, track='MELODY',
                                             offset=start_at_seconds, skip_leading_space=skip_empty_intro)
    except:
        print('Unable to parse input file, exiting...')
        sys.exit(1)

    generated_seq = []
    for input_window in melody_seq:
        generated_seq.append(model.predict(input_window))

    accompaniment_track = dh.build_accompaniment_track(generated_seq)

    pm = dh.add_accompaniment_track(pm=melody,
                                    accomp_notes=generated_seq,
                                    out_file='generated.mid',
                                    initial_tempo=120,
                                    melody_instrument=0,
                                    accomp_instrument=32,
                                    concat_sequential=True)

    if parameters['roll']:
        dh.plot_piano_roll(generated_seq)
