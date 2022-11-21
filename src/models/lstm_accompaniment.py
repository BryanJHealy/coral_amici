"""
Simple LSTM encoder-decoder network
"""
import tensorflow as tf
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import RepeatVector


class LstmAccompaniment:
    def __init__(self, sequence_duration=15, sampling_frequency=60,
                 vocab_size=128, learning_rate=0.005):
        samples_per_sequence = sequence_duration * sampling_frequency
        input_shape = (samples_per_sequence, vocab_size)
        # print(f'input shape: {input_shape}')

        # TODO: try different activation functions ('relu')
        inputs = tf.keras.Input(input_shape)
        encoder = LSTM(128)(inputs)
        decode1 = RepeatVector(samples_per_sequence)(encoder)
        decode2 = LSTM(128, return_sequences=True)(decode1)
        outputs = TimeDistributed(Dense(vocab_size, name='active_pitches'))(decode2)

        self.model = tf.keras.Model(inputs, outputs)

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(loss=tf.keras.losses.BinaryCrossentropy, optimizer=optimizer)
        self.model.summary()

    def train(self, train_ds: tf.data.Dataset,
              epochs: int,
              ckpt_path='./training_checkpoints/ckpt_{epoch}'):

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=ckpt_path,
                save_weights_only=True),
            tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=5,
                verbose=1,
                restore_best_weights=True),
        ]

        history = self.model.fit(
            train_ds,
            epochs=epochs,
            callbacks=callbacks,
        )

    def evaluate(self, eval_ds: tf.data.Dataset):
        losses = self.model.evaluate(eval_ds, return_dict=True)
        print(losses)

    def save(self, dir_name='lstm_accompaniment'):
        self.model.save(dir_name)

    def load(self, dir_name='lstm_accompaniment'):
        self.model = tf.keras.load_model(dir_name)
