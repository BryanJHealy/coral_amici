import numpy as np
import tensorflow as tf
import note_seq
import pretty_midi
import fluidsynth


class BasicModel():
    def __init__(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layersLSTM(),
            # tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(units=1, input_shape=[1])
        ])

        x = np.array([[1.], [2.]])
        y = np.array([[2.], [4.]])
        self.model.compile(optimizer='sgd', loss='mean_squared_error')
        self.model.fit(x, y, epochs=1)
        self.model.summary()

    def train(self):
        pass

    def evaluate(self):
        pass

    def save_checkpoint(self):
        pass

    def convert(self):
        x = np.array([[1.], [2.]])
        y = np.array([[2.], [4.]])

        self.model.compile(optimizer='sgd', loss='mean_squared_error')
        self.model.fit(x, y, epochs=1)
        self.model.summary()

        def representative_dataset():
            for _ in range(100):
                data = np.random.rand(1, 2, 1)
                yield [data.astype(np.float32)]

        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8  # or tf.uint8
        converter.inference_output_type = tf.int8  # or tf.uint8
        tflite_model = converter.convert()

        with open('vae.tflite', 'wb') as f:
            f.write(tflite_model)


if __name__ == '__main__':
    model = BasicModel()
