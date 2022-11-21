import pretty_midi
import collections
import pandas as pd
import numpy as np
from tensorflow import data as tfd
from numpy import array
# from os import listdir
import glob

key_order = ['pitch', 'step', 'duration']


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
                    activation[int(note[pitch_col])] = 1.0  # TODO: vocab 0-based index?
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

    mel_window = np.zeros((vocab_size, sequence_samples))
    acc_window = np.zeros((vocab_size, sequence_samples))
    for sample_num in range(file_samples):
        sample_time = sample_num / sample_frequency
        active_mel_pitches, mel_idx = get_activation_tensor(melody_dataset, sample_time, mel_idx)
        active_acc_pitches, acc_idx = get_activation_tensor(accomp_dataset, sample_time, acc_idx)

        if sample_num < sequence_samples:  # build entire first window
            mel_window[:, sample_num] = active_mel_pitches
            acc_window[:, sample_num] = active_acc_pitches
        else:  # shift windows by one sample
            mel_window = mel_window[:, 1:]
            acc_window = acc_window[:, 1:]
            mel_window = np.append(mel_window, active_mel_pitches)
            acc_window = np.append(acc_window, active_acc_pitches)

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


def get_pop_data(path, sequence_duration, vocab_size=128):
    # TODO: use os to make cross-platform. Currently needs a '/' at end of path
    midi_files = glob.glob(str(f'{path}**/*.mid*'))
    dataset = None
    for song in midi_files:
        melody_notes, accomp_notes = parse_pop_song_accompaniment(song)
        if (melody_notes is None) or (accomp_notes is None):
            continue

        song_ds = create_sequences_for_accompaniment(melody_notes, accomp_notes,
                                                     sequence_duration, vocab_size=vocab_size)

        dataset = song_ds if dataset is None else dataset.concatenate(song_ds)
    return dataset
