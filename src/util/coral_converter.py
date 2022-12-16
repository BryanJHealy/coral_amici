import numpy as np
import tensorflow as tf
import argparse, sys

if __name__ == '__main__':
    # Command-line argument parsing for data path
    parser = argparse.ArgumentParser(description='Convert Tensorflow model to Coral TFLite')
    parser.add_argument('model_path', action='store',
                        help='Path to the trained model folder')
    parameters = vars(parser.parse_args(sys.argv[1:]))

    path = parameters['model_path']

    # Load model
    model = tf.keras.models.load_model(path)

    # Convert
    # TODO: add get_representative_dataset function to data_handling
    def representative_dataset():
        for _ in range(100):
            data = np.random.rand(1, model.input_spec.shape[1], model.input_spec.shape[2])
            # for dseq in data:
            #     for dnote in dseq:
            #         dnote[0] *= 70
            #         dnote[0] = round(dnote[0] + 30)
            #         dnote[1] *= 0.5
            #         dnote[2] *= 0.5
            yield [data.astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.experimental_new_converter = True
    converter.inference_input_type = tf.int8  # or tf.uint8
    converter.inference_output_type = tf.int8  # or tf.uint8
    tflite_model = converter.convert()

    with open('converted.tflite', 'wb') as f:
        f.write(tflite_model)
