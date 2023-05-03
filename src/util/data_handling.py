import pretty_midi
import collections
import pandas as pd
import numpy as np
from tensorflow import data as tfd
from tensorflow.keras.losses import Loss
# from tensorflow.keras.losses import LossFunctionWrapper
import tensorflow as tf
from numpy import array
import glob
from matplotlib import pyplot as plt

# tf.config.run_functions_eagerly(True)


class SequenceLoss(Loss):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    def call(self, y_true, y_pred):
        # log_y_pred = tf.math.log(y_pred)
        # elements = -tf.math.multiply_no_nan(x=log_y_pred, y=y_true)
        # return tf.reduce_mean(tf.reduce_sum(elements,axis=1))

        # converted predicted to boolean using threshold
        # y_pred = tf.where(y_pred > self.threshold, 1, 0)
        # (false positives + false negatives) = predicted XOR actual
        # errors = tf.math.logical_xor(y_pred.numpy(), y_true.numpy())
        # sum((false positives + false negatives) / vocab) / samples
        # num_errors = len(tf.where())
        # = no. 1s in XOR / (vocab * samples)
        # ---
        # r1 = tf.where(y_true > 0.5)
        # r2 = tf.where(y_pred > 0.5)
        # total = y_true.shape[1] + y_true.shape[2]
        # r4 = r3.numpy().flatten()
        # return len(r4) / total
        # y_pred_bool = tf.where(y_pred > 0.5, 1.0, 0.0)
        # a = tf.math.greater_equal(y_pred, tf.constant(0.5))
        a = tf.cast(y_pred, tf.bool)
        a1 = y_pred.numpy()
        b = tf.where(y_true > 0.5)
        y_pred_bool = tf.where(y_pred > 0.5, True, False)
        y_true_bool = tf.where(y_pred > 0.5, True, False)
        xor_bool = tf.math.logical_xor(y_pred_bool, y_true_bool)
        xor = tf.where(xor_bool is True, 1.0, 0.0)
        unique = tf.reduce_sum(xor)
        total_active = tf.reduce_sum(y_pred_bool) + tf.reduce_sum(y_true)
        # correct = tf.where(y_pred_bool == y_true)
        # incorrect = tf.where(y_pred_bool != y_true, 1.0, 0.0)
        # loss = tf.convert_to_tensor(1 - (correct/(correct + incorrect)))
        loss = tf.math.subtract(1, tf.math.divide(tf.subtract(total_active, tf.math.multiply(unique, 2)), unique))
        # loss = tf.convert_to_tensor(1 - ((total_active - (2 * unique)) / unique))
        return loss
        # return tf.losses.binary_crossentropy(y_true=y_true, y_pred=y_pred)


def midi_to_notes(pm: pretty_midi.PrettyMIDI, instrument_index=0) -> pd.DataFrame:
    instrument = pm.instruments[instrument_index]
    notes = collections.defaultdict(list)

    # Sort the notes by start time
    sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
    prev_start = sorted_notes[0].start

    for note in sorted_notes:
        start = note.start
        end = note.end
        notes['pitch'].append(note.pitch)
        notes['start'].append(start)
        notes['end'].append(end)
        notes['step'].append(start - prev_start)
        notes['duration'].append(end - start)
        prev_start = start

    return pd.DataFrame({name: array(value) for name, value in notes.items()})


def create_sequences_for_replication(
        dataset: tfd.Dataset,
        seq_length: int,
        vocab_size=128,
) -> tfd.Dataset:
    """Returns TF Dataset of sequences and their identity as labels."""
    windows = dataset.window(seq_length, shift=1, stride=1,
                             drop_remainder=True)

    # `flat_map` flattens the" dataset of datasets" into a dataset of tensors
    flatten = lambda x: x.batch(seq_length, drop_remainder=True)
    sequences = windows.flat_map(flatten)

    # Normalize note pitch from 0-128 to 0.0-1.0 range
    def scale_pitch(x):
        x = x / [vocab_size, 1.0, 1.0]
        return x

    # Split the labels
    def replicate_labels(sequences):
        scaled = scale_pitch(sequences)
        return scaled, scaled

    return sequences.map(replicate_labels, num_parallel_calls=tfd.AUTOTUNE)


def create_sequences_for_accompaniment(
        melody_dataset: np.array,
        accomp_dataset: np.array,
        seq_duration: float,
        vocab_size=128,
        sample_frequency=60
) -> tfd.Dataset:
    """Returns TF Dataset of melody sequences and their accompaniment sequences."""

    pitch_col, start_col, end_col = 0, 1, 2

    def get_activation_tensor(note_events, sample_time, ds_idx):
        activation = np.zeros(vocab_size)
        for note_idx in range(ds_idx, len(note_events)):  # TODO check bounds
            note = note_events[note_idx]
            if note[start_col] <= sample_time:
                if note[end_col] > sample_time:
                    activation[int(note[pitch_col])] = 1.0
                else:
                    ds_idx += 1  # increment dataset pointer
            else:
                break
        return activation, ds_idx

    mel_windows = []
    acc_windows = []

    mel_idx = 0
    acc_idx = 0
    file_end_time = max(melody_dataset[-1][end_col], accomp_dataset[-1][end_col])
    file_samples = int(np.ceil(file_end_time * sample_frequency))
    sequence_samples = seq_duration * sample_frequency
    if file_samples < sequence_samples:
        return None

    mel_window = np.zeros((vocab_size, sequence_samples))
    acc_window = np.zeros((vocab_size, sequence_samples))
    for sample_num in range(file_samples):
        sample_time = sample_num / sample_frequency
        active_mel_pitches, mel_idx = get_activation_tensor(melody_dataset, sample_time, mel_idx)
        active_acc_pitches, acc_idx = get_activation_tensor(accomp_dataset, sample_time, acc_idx)

        if sample_num < sequence_samples:  # build entire first window
            mel_window[:, sample_num] = active_mel_pitches
            acc_window[:, sample_num] = active_acc_pitches
            if sample_num != sequence_samples - 1:
                continue  # skip adding incomplete windows to dataset
        else:  # shift windows by one sample
            mel_window = mel_window[:, 1:]
            acc_window = acc_window[:, 1:]
            active_mel_pitches = np.expand_dims(active_mel_pitches, axis=1)
            active_acc_pitches = np.expand_dims(active_acc_pitches, axis=1)
            mel_window = np.append(mel_window, active_mel_pitches, axis=1)
            acc_window = np.append(acc_window, active_acc_pitches, axis=1)

        if mel_window.any() and acc_window.any():  # don't add data if either window is empty
            mel_windows.append(mel_window)
            acc_windows.append(acc_window)

    dataset = tfd.Dataset.from_tensor_slices((mel_windows, acc_windows))
    return dataset


def parse_pop_song_accompaniment(filename):
    # create dataset with (MELODY sequence):(PIANO sequence)
    song = pretty_midi.PrettyMIDI(filename)
    melody_notes = None
    accomp_notes = None
    key_order = ['pitch', 'start', 'end']
    for instrument_idx in range(len(song.instruments)):
        if song.instruments[instrument_idx].name == 'MELODY':
            melody_notes = midi_to_notes(song, instrument_index=instrument_idx)
            melody_notes = np.stack([melody_notes[key] for key in key_order], axis=1)
            # melody_ds = tfd.Dataset.from_tensor_slices(melody_notes)
        elif song.instruments[instrument_idx].name == 'PIANO':
            accomp_notes = midi_to_notes(song, instrument_index=instrument_idx)
            accomp_notes = np.stack([accomp_notes[key] for key in key_order], axis=1)
            # accomp_ds = tfd.Dataset.from_tensor_slices(accomp_notes)
    return melody_notes, accomp_notes


def get_pop_data(path, sequence_duration, vocab_size=128, max_files=10, sampling_frequency=60):
    # TODO: use os to make cross-platform. Currently needs a '/' at end of path
    midi_files = glob.glob(str(f'{path}**/*.mid*'))
    dataset = None
    num_processed = 0
    for song in midi_files:
        if num_processed < max_files:
            melody_notes, accomp_notes = parse_pop_song_accompaniment(song)
            if (melody_notes is None) or (accomp_notes is None):
                continue

            song_ds = create_sequences_for_accompaniment(melody_notes, accomp_notes, sequence_duration,
                                                         vocab_size=vocab_size, sample_frequency=sampling_frequency)
            if song_ds is not None:
                dataset = song_ds if dataset is None else dataset.concatenate(song_ds)
            num_processed += 1
        else:
            break
    return dataset


def plot_piano_roll(song: pretty_midi.PrettyMIDI, tracks=('MELODY', 'generation'),
                    axes=True, save_file=None):
    plt.figure(figsize=(20, 4))
    colors = ['b', 'r', 'g', 'p']
    plt_idx = 0
    for instrument_idx in range(len(song.instruments)):
        if song.instruments[instrument_idx].name in tracks:
            notes = song.instruments[instrument_idx].notes
            pitches, starts, stops = [], [], []
            for note in notes:
                pitches.append(note.pitch)
                starts.append(note.start)
                stops.append(note.end)
            plot_pitch = np.stack([pitches, pitches], axis=0)
            plot_start_stop = np.stack([starts, stops], axis=0)
            plt.plot(plot_start_stop, plot_pitch, color=colors[plt_idx], marker="|",
                     label=song.instruments[instrument_idx].name)
            plt_idx += 1
    if axes:
        plt.xlabel('Time [s]')
        plt.ylabel('Pitch')
        _ = plt.title('Melody and Generated Accompaniment')
    else:
        plt.axis('off')
    if save_file:
        plt.savefig(save_file, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()


def build_accompaniment_track(sequence: np.ndarray, instrument_num=33,
                              sample_frequency=60, velocity=100,
                              concat_sequential=True, activation_threshold=0.8):
    instrument = pretty_midi.Instrument(program=instrument_num, is_drum=False,
                                        name='generation')

    notes = [(-1, -1) for _ in range(128)]  # TODO: pass vocab size
    active_notes = np.argwhere(sequence[0, :, :] >= activation_threshold)
    sequence_end = sequence.shape[-1] / sample_frequency
    for pitch, sample_idx in active_notes:
        if notes[pitch] == (-1, -1):
            notes[pitch] = (sample_idx, sample_idx)
        else:
            if notes[pitch][1] == sample_idx - 1:
                notes[pitch] = (notes[pitch][0], sample_idx)
            else:
                note = pretty_midi.Note(velocity=velocity, pitch=pitch,
                                        start=(notes[pitch][0] / sample_frequency),
                                        end=(notes[pitch][1] / sample_frequency) + 1)
                instrument.notes.append(note)
                notes[pitch] = (sample_idx, sample_idx)

    # create note for any non-repeating pitches
    for pitch_idx in range(len(notes)):
        if notes[pitch_idx] != (-1, -1):
            note = pretty_midi.Note(velocity=velocity, pitch=pitch_idx,
                                    start=(notes[pitch_idx][0] / sample_frequency),
                                    end=(notes[pitch_idx][1] / sample_frequency) + 1)
            instrument.notes.append(note)

    return instrument


def add_accompaniment_track(pm: pretty_midi.PrettyMIDI, accomp_seq, out_file: str,
                            velocity: int = 100,  # note loudness
                            instrument_num=33, concat_sequential=True,
                            sample_frequency=60, activation_threshold=0.8
                            ) -> pretty_midi.PrettyMIDI:
    acc_instrument = build_accompaniment_track(accomp_seq, instrument_num, velocity=velocity,
                                               concat_sequential=concat_sequential,
                                               sample_frequency=sample_frequency,
                                               activation_threshold=activation_threshold)

    pm.instruments.append(acc_instrument)
    pm.write(out_file)
    return pm


def sort_notes(song: pretty_midi.PrettyMIDI):
    for instrument_idx in range(len(song.instruments)):
        song.instruments[instrument_idx].notes = \
            sorted(song.instruments[instrument_idx].notes, key=lambda note: note.start)
    return song


def get_activation_sequence(song: pretty_midi.PrettyMIDI, num_samples: int,
                            instrument_idxs, vocab_size=128, sample_frequency=60,
                            offset=0, starting_note_idxs=None, skip_leading_space=True,
                            isolate_track=True, add_batch_dimension=False):
    seq_duration = num_samples / float(sample_frequency)
    activation_seqs = {idx: np.zeros((vocab_size, num_samples)) for idx in instrument_idxs}

    if starting_note_idxs is None:
        starting_note_idxs = {x: 0 for x in instrument_idxs}

    last_note_reached = False

    for instrument_idx in instrument_idxs:
        # narrow note events down to sequence length from offset
        started = False
        seq_start = offset
        end_time = offset + seq_duration
        starting_note_idx = 0
        ending_note_idx = 0
        for note_idx in range(starting_note_idxs[instrument_idx], len(song.instruments[instrument_idx].notes)):
            note = song.instruments[instrument_idx].notes[note_idx]
            if not started:
                if note.end >= offset:
                    started = True
                    starting_note_idx = note_idx
                    ending_note_idx = note_idx
                    if skip_leading_space:
                        seq_start = max(offset, note.start)
                        end_time = seq_start + seq_duration
                    if note_idx == len(song.instruments[instrument_idx].notes) - 1:
                        last_note_reached = True
                else:
                    continue
            else:
                if note.start > end_time:
                    break  # outside of sequence scope
                else:
                    ending_note_idx = note_idx
                    if note_idx == len(song.instruments[instrument_idx].notes) - 1:
                        last_note_reached = True

            # NOTE: integer truncation of note start/end times shifts notes to previous sample
            # with 1/sample_frequency resolution
            start_col = max(int((note.start - seq_start) * sample_frequency), 0)
            end_col = min(int((note.end - seq_start) * sample_frequency), num_samples - 1)
            activation_seqs[instrument_idx][int(note.pitch), start_col:end_col] = 1.0  # activate corresponding note

        if isolate_track:  # TODO: work with multiple instrument tracks
            # only keep indicated track
            song.instruments = [song.instruments[instrument_idx]]

            # only keep notes within sequence scope
            song.instruments[0].notes = song.instruments[0].notes[starting_note_idx: ending_note_idx]

            # shift notes to beginning of song and clip notes that extend outside sequence
            for note in song.instruments[0].notes:
                note.start = max(0, note.start - seq_start)
                note.end = min(seq_duration, note.end - seq_start)

    if add_batch_dimension:
        for i_idx in instrument_idxs:
            activation_seqs[i_idx] = np.expand_dims(activation_seqs[i_idx], axis=0)  # add batch dimension
    return activation_seqs, song, last_note_reached


def get_instrument_idxs(song: pretty_midi.PrettyMIDI, instrument_names=('MELODY', 'PIANO')):
    instrument_idxs_dict = {}
    for instrument_idx in range(len(song.instruments)):
        if song.instruments[instrument_idx].name in instrument_names:
            instrument_idxs_dict[song.instruments[instrument_idx].name] = instrument_idx
    return tuple(instrument_idxs_dict[name] for name in instrument_names)


def generate_training_sequences(filepath, instrument_tracks=('MELODY', 'PIANO'), vocab_size=128,
                                num_samples=960, sample_frequencies=(4, 8, 16, 32, 64),
                                binary_activations=False, add_batch_dimension=False):
    song = pretty_midi.PrettyMIDI(filepath)
    song = sort_notes(song)
    instrument_idxs = get_instrument_idxs(song, instrument_names=instrument_tracks)
    note_idxs = {inst_idx: 0 for inst_idx in instrument_idxs}
    training_pairs = []
    # end = max([song.instruments[i_idx].notes[-1].end] for i_idx in instrument_idxs)
    # TODO: update -1 above to find note with latest end

    reduction_matrix = None
    packed_size = 0
    if binary_activations:
        reduction_matrix, packed_size = get_activation_reduction_matrix(data_size=32, vocab_size=vocab_size)

    for frequency in sample_frequencies:
        # sequence_duration = num_samples / float(frequency)
        # iterate over notes in melody
        valid = True
        last_note_reached = False
        while (not last_note_reached) and valid:
            activation_seqs, _, last_note_reached = get_activation_sequence(song, num_samples=num_samples,
                                                                            instrument_idxs=instrument_idxs,
                                                                            vocab_size=vocab_size,
                                                                            sample_frequency=frequency,
                                                                            offset=0, skip_leading_space=True,
                                                                            starting_note_idxs=note_idxs,
                                                                            isolate_track=False,
                                                                            add_batch_dimension=add_batch_dimension)
            for seq_key in activation_seqs.keys():
                if not np.any(activation_seqs[seq_key]):
                    valid = False
                    break

            if valid:
                mel_activation = activation_seqs[instrument_idxs[0]]
                acc_activation = activation_seqs[instrument_idxs[1]]

                if binary_activations:
                    mel_activation = reduce_activation(mel_activation, reduction_matrix=reduction_matrix,
                                                       data_size=32, packed_size=packed_size)
                    acc_activation = reduce_activation(acc_activation, reduction_matrix=reduction_matrix,
                                                       data_size=32, packed_size=packed_size)

                if len(mel_activation.shape) == 4:
                    training_pairs.append((mel_activation[0], acc_activation[0]))
                else:
                    training_pairs.append((mel_activation, acc_activation))
                for i_idx in instrument_idxs:
                    note_idxs[i_idx] += 1

    return training_pairs


def import_midi_input_sequence(filepath, num_samples: int, instrument_tracks=('MELODY',),
                               vocab_size=128, sample_frequency=60, offset=0,
                               skip_leading_space=True, isolate_track=True,
                               add_batch_dimension=True):
    song = pretty_midi.PrettyMIDI(filepath)
    song = sort_notes(song)
    instrument_idxs = get_instrument_idxs(song, instrument_names=instrument_tracks)

    activation_seqs, song, _ = get_activation_sequence(song, num_samples=num_samples,
                                                       instrument_idxs=instrument_idxs,
                                                       vocab_size=vocab_size, sample_frequency=sample_frequency,
                                                       offset=offset, skip_leading_space=skip_leading_space,
                                                       isolate_track=isolate_track,
                                                       add_batch_dimension=add_batch_dimension)
    activation_seq = activation_seqs[instrument_idxs[0]]
    return activation_seq, song


def get_activation_reduction_matrix(data_size=32, vocab_size=128):
    # TODO: check order of powers(ascend/descend) -- endianness doesn't matter as long as consistent
    packed_size = vocab_size // data_size  # TODO: check non-zero remainder behavior
    powers = [2 ** idx for idx in range(data_size)]
    # powers = powers * packed_size
    powers = np.array(powers)
    return powers, packed_size


def reduce_activation(activation, reduction_matrix, data_size, packed_size):
    return np.expand_dims(np.array([reduction_matrix @ (activation[0, idx * data_size:(idx + 1) * data_size, :]) for idx in range(packed_size)]), axis=0)
