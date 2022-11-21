"""
Adapted from tensorflow documentation found here:
https://github.com/tensorflow/docs/blob/master/site/en/tutorials/audio/music_generation.ipynb
"""

import glob
import tensorflow as tf
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import RepeatVector
import numpy as np
import pandas as pd
from src.util.data_handling import midi_to_notes
from src.util.data_handling import create_sequences_for_replication

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)


def mse_with_positive_pressure_step(y_true: tf.Tensor, y_pred: tf.Tensor):
    # print('y_true: ', y_true)
    # print('y_pred: ', y_pred)
    truth = tf.expand_dims(y_true[:,:,1], 2)
    mse = (truth - y_pred) ** 2
    positive_pressure = 10 * tf.maximum(-truth, 0.0)
    return tf.reduce_mean(mse + positive_pressure)


def mse_with_positive_pressure_duration(y_true: tf.Tensor, y_pred: tf.Tensor):
    truth = tf.expand_dims(y_true[:,:,1], 2)
    # print('y_true: {}, truth: {}'.format(y_true, truth))
    # print('y_pred: ', y_pred)
    mse = (truth - y_pred) ** 2
    positive_pressure = 10 * tf.maximum(-truth, 0.0)
    return tf.reduce_mean(mse + positive_pressure)


def scce_wrapper(y_true: tf.Tensor, y_pred: tf.Tensor):
    scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    return scce(tf.expand_dims(y_true[:,:,0],2), y_pred)


if __name__ == '__main__':
    # TODO: use relative path
    filenames = glob.glob(str('/media/steamgames/coral/coral_amici/src/data/maestro/**/*.mid*'))
    # filenames = glob.glob(str('/media/steamgames/coral/coral_amici/src/data/lakh/clean_midi/**/*.mid*'))
    # print(f'#files: {len(filenames)}')

    # -----------
    num_files = 100
    all_notes = []
    for f in filenames[:num_files]:
        notes = midi_to_notes(f)
        all_notes.append(notes)

    all_notes = pd.concat(all_notes)

    n_notes = len(all_notes)
    # print('Number of notes parsed:', n_notes)

    key_order = ['pitch', 'step', 'duration']
    train_notes = np.stack([all_notes[key] for key in key_order], axis=1)

    notes_ds = tf.data.Dataset.from_tensor_slices(train_notes)
    # print(notes_ds.element_spec)

    seq_length = 25
    vocab_size = 128
    seq_ds = create_sequences_for_replication(notes_ds, seq_length, vocab_size)
    # print(seq_ds.element_spec)

    # for seq, target in seq_ds.take(1):
    #     print('sequence shape:', seq.shape)
    #     print('sequence elements (first 10):', seq[0: 10])
    #     print()
    #     print('target:', target)

    batch_size = 64
    buffer_size = n_notes - seq_length  # the number of items in the dataset
    train_ds = (seq_ds
                .shuffle(buffer_size)
                .batch(batch_size, drop_remainder=True)
                .cache()
                .prefetch(tf.data.experimental.AUTOTUNE))

    # print(train_ds.element_spec)

    input_shape = (seq_length, 3)
    # print(f'input shape: {input_shape}')
    learning_rate = 0.005

    inputs = tf.keras.Input(input_shape)
    # TODO: try different activation functions ('relu')
    encoder = LSTM(128)(inputs)
    decode1 = RepeatVector(seq_length)(encoder)
    decode2 = LSTM(128, return_sequences=True)(decode1)

    outputs = {
        'pitch': TimeDistributed(Dense(128, name='pitch'))(decode2),
        'step': TimeDistributed(Dense(1, name='step'))(decode2),
        'duration': TimeDistributed(Dense(1, name='duration'))(decode2),
    }

    model = tf.keras.Model(inputs, outputs)

    loss = {
        'pitch': scce_wrapper,
        'step': mse_with_positive_pressure_step,
        'duration': mse_with_positive_pressure_duration,
    }

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(loss=loss, optimizer=optimizer)

    model.summary()

    # losses = model.evaluate(train_ds, return_dict=True)
    # print(losses)

    # model.compile(
    #     loss=loss,
    #     loss_weights={
    #         'pitch': 0.05,
    #         'step': 1.0,
    #         'duration': 1.0,
    #     },
    #     optimizer=optimizer,
    # )

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

    epochs = 50

    history = model.fit(
        train_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    model.save('lstm_autoencoder')