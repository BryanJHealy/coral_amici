import numpy as np
import tensorflow as tf


class BasicModel():
    def __init__(self):
        x = np.array([[1.], [2.]])
        y = np.array([[2.], [4.]])
        model = tf.keras.models.Sequential([
            # tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(units=1, input_shape=[1])
        ])

        model.compile(optimizer='sgd', loss='mean_squared_error')
        model.fit(x, y, epochs=1)
        model.summary()

        def representative_dataset():
            for _ in range(100):
                data = np.random.rand(1, 2, 1)
                yield [data.astype(np.float32)]

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8  # or tf.uint8
        converter.inference_output_type = tf.int8  # or tf.uint8
        self.tflite_model = converter.convert()

        with open('basic.tflite', 'wb') as f:
            f.write(self.tflite_model)

    def run(self):
        self.interpreter = tf.lite.Interpreter(model_content=self.tflite_model,
                                               experimental_delegates=[
                                                   tf.lite.Interpreter.load_delegate('libedgetpu.so.1')])
        self.interpreter.allocate_tensors()  # Needed before execution!
        output = self.interpreter.get_output_details()[0]  # Model has single output.
        input = self.interpreter.get_input_details()[0]  # Model has single input.
        input_data = tf.constant(1., shape=[1, 1])
        self.interpreter.set_tensor(input['index'], input_data)
        self.interpreter.invoke()
        self.interpreter.get_tensor(output['index']).shape


if __name__ == '__main__':
    model = BasicModel()