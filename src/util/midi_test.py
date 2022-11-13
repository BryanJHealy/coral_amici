"""
Adapted from tensorflow documentation found here:
https://github.com/tensorflow/docs/blob/master/site/en/tutorials/audio/music_generation.ipynb
"""

# import fluidsynth
import pretty_midi
import glob
from IPython import display
from matplotlib import pyplot as plt
from typing import Dict, List, Optional, Sequence, Tuple
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np
import pandas as pd
import collections
import seaborn as sns

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

SAMPLING_RATE = 16000


def display_audio(pm: pretty_midi.PrettyMIDI, seconds=30):
    waveform = pm.fluidsynth(fs=SAMPLING_RATE)
    waveform_short = waveform[:seconds * SAMPLING_RATE]
    return display.Audio(waveform_short, rate=SAMPLING_RATE)


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


def get_note_names(raw_notes: pd.DataFrame):
    return np.vectorize(pretty_midi.note_number_to_name(raw_notes['pitch']))


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


def mse_with_positive_pressure(y_true: tf.Tensor, y_pred: tf.Tensor):
    mse = (y_true - y_pred) ** 2
    positive_pressure = 10 * tf.maximum(-y_pred, 0.0)
    return tf.reduce_mean(mse + positive_pressure)


def predict_next_note(
        notes: np.ndarray,
        keras_model: tf.keras.Model,
        temperature: float = 1.0):
    """Generates a note IDs using a trained sequence model."""

    assert temperature > 0

    # Add batch dimension
    inputs = tf.expand_dims(notes, 0)

    predictions = model.predict(inputs)
    pitch_logits = predictions['pitch']
    step = predictions['step']
    duration = predictions['duration']

    pitch_logits /= temperature
    pitch = tf.random.categorical(pitch_logits, num_samples=1)
    pitch = tf.squeeze(pitch, axis=-1)
    duration = tf.squeeze(duration, axis=-1)
    step = tf.squeeze(step, axis=-1)

    # `step` and `duration` values should be non-negative
    step = tf.maximum(0, step)
    duration = tf.maximum(0, duration)

    return int(pitch), float(step), float(duration)


if __name__ == '__main__':
    # TODO: use relative path
    filenames = glob.glob(str('/media/steamgames/coral/src/data/maestro/**/*.mid*'))
    print(f'#files: {len(filenames)}')

    # sample from dataset
    sample_file = filenames[1]
    print(sample_file)

    # convert sample to pretty_midi
    pm_sample = pretty_midi.PrettyMIDI(sample_file)
    # display_audio(pm_sample) # TODO: fix fluidsynth

    # get instrument info
    print('Number of instruments:', len(pm_sample.instruments))
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

    # plot_piano_roll(raw_notes)
    # plot_distributions(raw_notes)

    # -----------
    num_files = 1  # 5
    all_notes = []
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
    learning_rate = 0.005

    inputs = tf.keras.Input(input_shape)
    x = tf.keras.layers.LSTM(128)(inputs)

    outputs = {
        'pitch': tf.keras.layers.Dense(128, name='pitch')(x),
        'step': tf.keras.layers.Dense(1, name='step')(x),
        'duration': tf.keras.layers.Dense(1, name='duration')(x),
    }

    model = tf.keras.Model(inputs, outputs)

    loss = {
        'pitch': tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True),
        'step': mse_with_positive_pressure,
        'duration': mse_with_positive_pressure,
    }

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(loss=loss, optimizer=optimizer)

    model.summary()

    # losses = model.evaluate(train_ds, return_dict=True)
    # print(losses)

    model.compile(
        loss=loss,
        loss_weights={
            'pitch': 0.05,
            'step': 1.0,
            'duration': 1.0,
        },
        optimizer=optimizer,
    )

    # model.evaluate(train_ds, return_dict=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath='./training_checkpoints/ckpt_{epoch}',
            save_weights_only=True),
        tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=5,
            verbose=1,
            restore_best_weights=True),
    ]

    epochs = 1

    history = model.fit(
        train_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    model.save('lstm_basic')


    j = 0
    for d in notes_ds:
        print(d)
        print(d[0])
        print(d[1])
        print(d[2])
        j+=1
        if j > 0:
            break

    def representative_dataset():
        # for input_value in tf.data.Dataset.from_tensor_slices(notes_ds).batch(1).take(100):
        #     yield [input_value]
        # for data in notes_ds.batch(1).take(100):
        #     yield tf.dtypes.cast(data, tf.float32)
        # for data in notes_ds:
        #     yield {
        #         'pitch': tf.dtypes.cast(data[0], tf.float32),
        #         'step': tf.dtypes.cast(data[1], tf.float32),
        #         'duration': tf.dtypes.cast(data[2], tf.float32)
        #     }
        # for seq, target in seq_ds.take(1):
        #     print(seq)
        #     yield {
        #         'input_1': tf.dtypes.cast(seq, tf.float32)
        #         # 'pitch': tf.dtypes.cast(data[0], tf.float32),
        #         # 'step': tf.dtypes.cast(data[1], tf.float32),
        #         # 'duration': tf.dtypes.cast(data[2], tf.float32)
        #     }
        # for data in seq_ds.batch(1, drop_remainder=True):
        #     yield tf.dtypes.cast(data, tf.float32)
        for _ in range(100):
            # [[76, 0, 1.673],
            #  [72, 0.25, 1.673]]
            data = np.random.rand(1, 25, 3)
            for dseq in data:
                for dnote in dseq:
                    dnote[0] *= 70
                    dnote[0] = round(dnote[0] + 30)
                    dnote[1] *= 0.5
                    dnote[2] *= 0.5
            yield [data.astype(np.float32)]


    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter._experimental_default_to_single_batch_in_tensor_list_ops = True
    converter.inference_input_type = tf.int8  # or tf.uint8
    converter.inference_output_type = tf.int8  # or tf.uint8
    tflite_model = converter.convert()

    with open('lstm_basic.tflite', 'wb') as f:
        f.write(tflite_model)

    plt.plot(history.epoch, history.history['loss'], label='total loss')
    plt.show()

    # Inference
    temperature = 2.0
    num_predictions = 120

    sample_notes = np.stack([raw_notes[key] for key in key_order], axis=1)

    # The initial sequence of notes; pitch is normalized similar to training
    # sequences
    input_notes = (
            sample_notes[:seq_length] / np.array([vocab_size, 1, 1]))

    generated_notes = []
    prev_start = 0
    for _ in range(num_predictions):
        pitch, step, duration = predict_next_note(input_notes, model, temperature)
        start = prev_start + step
        end = start + duration
        input_note = (pitch, step, duration)
        generated_notes.append((*input_note, start, end))
        input_notes = np.delete(input_notes, 0, axis=0)
        input_notes = np.append(input_notes, np.expand_dims(input_note, 0), axis=0)
        prev_start = start

    generated_notes = pd.DataFrame(
        generated_notes, columns=(*key_order, 'start', 'end'))

    print(generated_notes.head(10))

    out_file = 'output.mid'
    out_pm = notes_to_midi(
        generated_notes, out_file=out_file, instrument_name=instrument_name)
    # display_audio(out_pm)

    plot_piano_roll(generated_notes)

    plot_distributions(generated_notes)
