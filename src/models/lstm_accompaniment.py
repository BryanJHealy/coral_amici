"""
Simple LSTM encoder-decoder network
"""
import tensorflow as tf
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras import Model
from src.util.data_handling import SequenceLoss


class CustomModel1(Model):
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


class LstmAccompaniment:
    def __init__(self, samples_per_sequence=240, vocab_size=128, learning_rate=0.005, compression_factor=2,
                 binary_activations=False, data_size=32, *args, **kwargs):
        super().__init__(*args, **kwargs)
        sample_depth = vocab_size//data_size if binary_activations else vocab_size
        input_shape = (sample_depth, samples_per_sequence)
        inner_size = int(samples_per_sequence/compression_factor)

        inputs = tf.keras.layers.Input(input_shape)
        encoder = LSTM(inner_size)(inputs)
        decode1 = RepeatVector(sample_depth)(encoder)
        decode2 = LSTM(inner_size, return_sequences=True)(decode1)
        outputs = TimeDistributed(Dense(samples_per_sequence, name='active_pitches'))(decode2)
        # self.model = tf.keras.Model(inputs, outputs)
        self.model = CustomModel1(inputs, outputs)

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(loss=tf.keras.losses.mean_squared_error, optimizer=optimizer)
        # self.model.compile(loss=tf.keras.losses.binary_crossentropy, optimizer=optimizer)
        # self.model.compile(loss=SequenceLoss(threshold=0.5), optimizer=optimizer)
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
