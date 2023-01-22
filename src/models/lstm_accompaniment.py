"""
Simple LSTM encoder-decoder network
"""
import tensorflow as tf
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import RepeatVector


class LstmAccompaniment:
    def __init__(self, samples_per_sequence=240, vocab_size=128, learning_rate=0.005, compression_factor=2):
        input_shape = (vocab_size, samples_per_sequence)
        inner_size = int(samples_per_sequence/compression_factor)

        inputs = tf.keras.layers.Input(input_shape)
        encoder = LSTM(inner_size)(inputs)
        decode1 = RepeatVector(vocab_size)(encoder)
        decode2 = LSTM(inner_size, return_sequences=True)(decode1)
        outputs = TimeDistributed(Dense(samples_per_sequence, name='active_pitches'))(decode2)
        self.model = tf.keras.Model(inputs, outputs)

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(loss=tf.keras.losses.binary_crossentropy, optimizer=optimizer)
        self.model.summary()

    def train(self, train_ds: tf.data.Dataset,
              epochs: int,
              ckpt_path='./training_checkpoints/ckpt_{epoch}',
              batch_size=64):

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
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
        )
        pass

    def evaluate(self, eval_ds: tf.data.Dataset):
        losses = self.model.evaluate(eval_ds, return_dict=True)
        print(losses)

    def save(self, dir_name='lstm_accompaniment'):
        self.model.save(dir_name)

    def load(self, dir_name='lstm_accompaniment'):
        self.model = tf.keras.load_model(dir_name)
