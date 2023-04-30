from src.util import data_handling as dh
import argparse
import sys
import tensorflow as tf

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate accompaniment for given MIDI melody')
    parser.add_argument('--model_path', action='store', help='Path to the model')
    parser.add_argument('--input', action='store', help='Path to the input .mid file')
    parser.add_argument('--output', action='store', help='Path to the output .mid file')
    parser.add_argument('--roll', action='store_true', help='create a piano roll png for the output generation')
    parameters = vars(parser.parse_args(sys.argv[1:]))

    # Load model
    model = tf.keras.models.load_model(parameters['model_path'])

    # Configure sequence
    sequence_seconds = 15
    start_at_seconds = 0  # sequence start offset from beginning of file
    skip_empty_intro = True  # sequence starts at first note after offset if True, else at offset
    num_pitches = 128  # 0 to 127, representing the notes from C-1 to G9
    samples_per_sec = 1  # data resolution
    only_keep_melody_track = True  # build new MIDI file using only the melody track from the input file

    # Parse input midi file using PrettyMidi and collect list of input sequence windows
    try:
        melody_seq, melody_pm = dh.import_midi_input_sequence(parameters['input'], sequence_seconds,
                                                              instrument_tracks=('MELODY',), vocab_size=num_pitches,
                                                              sample_frequency=samples_per_sec, offset=start_at_seconds,
                                                              skip_leading_space=skip_empty_intro,
                                                              isolate_track=only_keep_melody_track)
    except Exception as e:
        print(f'Unable to parse input file ({e}), exiting...')
        sys.exit(1)

    generated_seq = model.predict(melody_seq)

    # acoustic bass, see https://fmslogo.sourceforge.io/manual/midi-instrument.html for instrument choices
    accompaniment_instrument = 33
    activation_threshold = 0.1
    pm = dh.add_accompaniment_track(melody_pm, generated_seq,
                                    parameters['output'],
                                    velocity=100,
                                    instrument_num=accompaniment_instrument,
                                    concat_sequential=True,
                                    sample_frequency=samples_per_sec,
                                    activation_threshold=activation_threshold)

    if parameters['roll']:
        dh.plot_piano_roll(pm, tracks=('MELODY', 'generation'))
