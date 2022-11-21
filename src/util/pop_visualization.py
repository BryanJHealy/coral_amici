"""
Adapted from tensorflow documentation found here:
https://github.com/tensorflow/docs/blob/master/site/en/tutorials/audio/music_generation.ipynb
"""

import pretty_midi
import glob
from matplotlib import pyplot as plt
from typing import Optional
import tensorflow as tf
import numpy as np
import pandas as pd
import collections
import seaborn as sns

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)


def midi_to_notes(midi_file: str) -> pd.DataFrame:
    pm = pretty_midi.PrettyMIDI(midi_file)
    instrument = pm.instruments[0]
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

    return pd.DataFrame({name: np.array(value) for name, value in notes.items()})


def plot_piano_roll(notes: pd.DataFrame, count: Optional[int] = None):
    if count:
        title = f'First {count} notes'
    else:
        title = f'Whole track'
        count = len(notes['pitch'])
    plt.figure(figsize=(20, 4))
    plot_pitch = np.stack([notes['pitch'], notes['pitch']], axis=0)
    plot_start_stop = np.stack([notes['start'], notes['end']], axis=0)
    plt.plot(
        plot_start_stop[:, :count], plot_pitch[:, :count], color="b", marker=".")
    plt.xlabel('Time [s]')
    plt.ylabel('Pitch')
    _ = plt.title(title)
    plt.show()


def plot_distributions(notes: pd.DataFrame, drop_percentile=2.5):
    plt.figure(figsize=[15, 5])
    plt.subplot(1, 3, 1)
    sns.histplot(notes, x="pitch", bins=20)

    plt.subplot(1, 3, 2)
    max_step = np.percentile(notes['step'], 100 - drop_percentile)
    sns.histplot(notes, x="step", bins=np.linspace(0, max_step, 21))

    plt.subplot(1, 3, 3)
    max_duration = np.percentile(notes['duration'], 100 - drop_percentile)
    sns.histplot(notes, x="duration", bins=np.linspace(0, max_duration, 21))


def notes_to_midi(notes: pd.DataFrame, out_file: str, instrument_name: str,
                  velocity: int = 100,  # note loudness
                  ) -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program(
            instrument_name))

    prev_start = 0
    for i, note in notes.iterrows():
        start = float(prev_start + note['step'])
        end = float(start + note['duration'])
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=int(note['pitch']),
            start=start,
            end=end,
        )
        instrument.notes.append(note)
        prev_start = start

    pm.instruments.append(instrument)
    pm.write(out_file)
    return pm


def create_sequences(
        dataset: tf.data.Dataset,
        seq_length: int,
        vocab_size=128,
) -> tf.data.Dataset:
    """Returns TF Dataset of sequence and label examples."""
    seq_length = seq_length + 1

    # Take 1 extra for the labels
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
        inputs = sequences[:-1]
        labels_dense = sequences[-1]
        labels = {key: labels_dense[i] for i, key in enumerate(key_order)}

        return scale_pitch(inputs), labels

    return sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)


if __name__ == '__main__':
    # TODO: use relative path
    filenames = glob.glob(str('/media/steamgames/coral/coral_amici/src/data/POP909/**/*.mid*'))
    print(f'#files: {len(filenames)}')

    # sample from dataset
    sample_file = filenames[1]
    print(sample_file)

    # convert sample to pretty_midi
    pm_sample = pretty_midi.PrettyMIDI(sample_file)
    # display_audio(pm_sample) # TODO: fix fluidsynth

    # get instrument info
    print('Number of instruments:', len(pm_sample.instruments))
    for idx in pm_sample.instruments:
        print(idx)
    instrument = pm_sample.instruments[0]
    instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
    print('Instrument name:', instrument_name)

    # Extract notes
    for i, note in enumerate(instrument.notes[:10]):
        note_name = pretty_midi.note_number_to_name(note.pitch)
        duration = note.end - note.start
        print(f'{i}: pitch={note.pitch}, note_name={note_name},'
              f' duration={duration:.4f}')

    raw_notes = midi_to_notes(sample_file)
    print(raw_notes.head())

    plot_piano_roll(raw_notes)
    plot_distributions(raw_notes)

    # -----------
    num_files = 2
    all_notes = []  # list of dataframes per file.instrument: (pitch, start, end, step, duration)
    for f in filenames[:num_files]:
        notes = midi_to_notes(f)
        all_notes.append(notes)

    all_notes = pd.concat(all_notes)

    n_notes = len(all_notes)
    print('Number of notes parsed:', n_notes)

    key_order = ['pitch', 'step', 'duration']
    train_notes = np.stack([all_notes[key] for key in key_order], axis=1)

    notes_ds = tf.data.Dataset.from_tensor_slices(train_notes)
    print(notes_ds.element_spec)

    seq_length = 25
    vocab_size = 128
    seq_ds = create_sequences(notes_ds, seq_length, vocab_size)
    print(seq_ds.element_spec)

    for seq, target in seq_ds.take(1):
        print('sequence shape:', seq.shape)
        print('sequence elements (first 10):', seq[0: 10])
        print()
        print('target:', target)

    batch_size = 64
    buffer_size = n_notes - seq_length  # the number of items in the dataset
    train_ds = (seq_ds
                .shuffle(buffer_size)
                .batch(batch_size, drop_remainder=True)
                .cache()
                .prefetch(tf.data.experimental.AUTOTUNE))

    print(train_ds.element_spec)

    input_shape = (seq_length, 3)
    print(f'input shape: {input_shape}')