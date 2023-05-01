from util import data_handling as dh
import argparse
import sys
import tensorflow as tf

class AccompanimentGenerator:
    def __init__(self, model_path, midi_file, out_file, roll=False,
                 sequence_seconds=15, start_secs=0, skip_empty_intro=True,
                 vocab_size=128, samples_per_sec=1, only_keep_melody_track=True):
        self.model_path = model_path
        self.midi_file = midi_file
        self.out_file  = out_file
        self.roll = roll

        # Configure sequence
        self.sequence_seconds = 15
        self.start_secs = start_secs  # sequence start offset from beginning of file
        self.skip_empty_intro = skip_empty_intro  # sequence starts at first note after offset if True, else at offset
        self.vocab_size = vocab_size  # 0 to 127, representing the notes from C-1 to G9
        self.samples_per_sec = samples_per_sec  # data resolution
        self.only_keep_melody_track = only_keep_melody_track  # build new MIDI file using only the melody track from the input file

        # Load model
        self.model = tf.keras.models.load_model(model_path)

    def generate(self):

        # Parse input midi file using PrettyMidi and collect list of input sequence windows
        try:
            melody_seq, melody_pm = dh.import_midi_input_sequence(self.midi_file, self.sequence_seconds,
                                                                instrument_tracks=('MELODY',), vocab_size=self.num_pitches,
                                                                sample_frequency=self.samples_per_sec, offset=self.start_at_seconds,
                                                                skip_leading_space=self.skip_empty_intro,
                                                                isolate_track=self.only_keep_melody_track)
        except Exception as e:
            print(f'Unable to parse input file ({e}), exiting...')
            sys.exit(1)

        generated_seq = self.model.predict(melody_seq)

        # acoustic bass, see https://fmslogo.sourceforge.io/manual/midi-instrument.html for instrument choices
        accompaniment_instrument = 33
        activation_threshold = 0.1
        pm = dh.add_accompaniment_track(melody_pm, generated_seq,
                                        self.out_file,
                                        velocity=100,
                                        instrument_num=accompaniment_instrument,
                                        concat_sequential=True,
                                        sample_frequency=self.samples_per_sec,
                                        activation_threshold=activation_threshold)

        if self.roll:
            dh.plot_piano_roll(pm, tracks=('MELODY', 'generation'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate accompaniment for given MIDI melody')
    parser.add_argument('--model_path', action='store', help='Path to the model')
    parser.add_argument('--input', action='store', help='Path to the input .mid file')
    parser.add_argument('--output', action='store', help='Path to the output .mid file')
    parser.add_argument('--roll', action='store_true', help='create a piano roll png for the output generation')
    parameters = vars(parser.parse_args(sys.argv[1:]))

    generator = AccompanimentGenerator(parameters['model_path'], parameters['input'], parameters['output'], parameters['roll'])
    generator.generate()