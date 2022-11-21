import pretty_midi
import collections
import pandas as pd
import numpy as np
from tensorflow import data as tfd
from numpy import array
# from os import listdir
import glob

key_order = ['pitch', 'step', 'duration']


def midi_to_notes(midi_file: str, instrument_index=0) -> pd.DataFrame:
    pm = pretty_midi.PrettyMIDI(midi_file)
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

    # Normalize note pitch
    def scale_pitch(x):
        x = x / [vocab_size, 1.0, 1.0]
        return x

    # Split the labels
    def split_labels(sequences):
        scaled = scale_pitch(sequences)
        return scaled, scaled

    return sequences.map(split_labels, num_parallel_calls=tfd.AUTOTUNE)


def create_sequences_for_accompaniment(
        melody_dataset: tfd.Dataset,
        accomp_dataset: tfd.Dataset,
        seq_length: int,
        vocab_size=128,
        num_samples=1800
) -> tfd.Dataset:
    """Returns TF Dataset of sequences and their identity as labels."""

    def get_activation_tensor(dataset, sample_time, ds_idx):
        activation = np.zeros(vocab_size)
        for note_idx in range(ds_idx, len(dataset)):  # TODO check bounds
            note = dataset[note_idx]
            if note['start'] <= sample_time:
                if note['end'] > sample_time:
                    activation[int(note['pitch'])] = 1.0  # TODO: vocab 0-based index?
                else:
                    ds_idx += 1  # increment dataset pointer
            else:
                break
        return activation, ds_idx

    mel_idx = 0
    acc_idx = 0
    sample_period = 30.0/num_samples
    for sample_num in range(num_samples):
        sample_time = sample_num * sample_period
        active_mel_pitches, mel_idx = get_activation_tensor(melody_dataset, sample_time, mel_idx)
        active_acc_pitches, acc_idx = get_activation_tensor(accomp_dataset, sample_time, acc_idx)

    # add window, window = window[1:], window.extend(empty_column, axis=1), window[-1] |= active_pitches

    windows = dataset.window(seq_length, shift=1, stride=1,
                             drop_remainder=True)

    # `flat_map` flattens the" dataset of datasets" into a dataset of tensors
    flatten = lambda x: x.batch(seq_length, drop_remainder=True)
    sequences = windows.flat_map(flatten)

    # Normalize note pitch
    def scale_pitch(x):
        x = x / [vocab_size, 1.0, 1.0]
        return x

    # Split the labels
    def split_labels(sequences):
        scaled = scale_pitch(sequences)
        return scaled, scaled

    return sequences.map(split_labels, num_parallel_calls=tfd.AUTOTUNE)


def parse_pop_song_accompaniment(filename):
    # create dataset with (MELODY sequence):(PIANO sequence)
    song = pretty_midi.PrettyMIDI(filename)
    melody_ds = None
    accomp_ds = None
    key_order = ['pitch', 'start', 'end']
    for instrument_idx in range(len(song.instruments)):
        if song.instruments[instrument_idx].name is 'MELODY':
            melody_notes = midi_to_notes(instrument_index=instrument_idx)
            melody_notes = np.stack([melody_notes[key] for key in key_order], axis=1)
            melody_ds = tfd.Dataset.from_tensor_slices(melody_notes)
        elif song.instruments[instrument_idx].name is 'PIANO':
            accomp_notes = midi_to_notes(instrument_index=instrument_idx)
            accomp_notes = np.stack([accomp_notes[key] for key in key_order], axis=1)
            accomp_ds = tfd.Dataset.from_tensor_slices(accomp_notes)
    return melody_ds, accomp_ds


def get_pop_data(path):
    # TODO: use os to make cross-platform. Currently needs a '/' at end of path
    midi_files = glob.glob(str(f'{path}**/*.mid*'))
    file_datasets = []
    for song in midi_files:
        melody_ds, accomp_ds = parse_pop_song_accompaniment(song)
        if (melody_ds is None) or (accomp_ds is None):
            continue
        create_sequences_for_accompaniment(melody_ds, accomp_ds)

