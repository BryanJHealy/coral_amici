import argparse
import sys
import tensorflow as tf
import numpy as np
from src.util.data_handling import get_pop_data
from src.models.lstm_accompaniment import LstmAccompaniment


if __name__ == '__main__':
    # Command-line argument parsing for data path
    parser = argparse.ArgumentParser(description='Transcribe Audio file to MIDI file')
    parser.add_argument('data_root_path', action='store',
                        help='Path to the POP909 root folder, end with \"/\"')
    parameters = vars(parser.parse_args(sys.argv[1:]))

    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # TODO: use path libraries
    # TODO: add verbose flags
    sequence_duration = 15  # seconds per sequence
    print('Building dataset...')
    max_files = 10
    dataset = get_pop_data(parameters['data_root_path'], sequence_duration, max_files=max_files)

    batch_size = 64
    buffer_size = dataset.cardinality().numpy()  # the number of items in the dataset
    train_ds = (dataset
                .shuffle(buffer_size)
                .batch(batch_size, drop_remainder=True)
                .cache()
                .prefetch(tf.data.experimental.AUTOTUNE))

    # print(train_ds.element_spec)

    print('Building model...')
    model = LstmAccompaniment(sequence_duration=15, sampling_frequency=60, learning_rate=0.005)
    print('Training model...')
    model.train(train_ds=train_ds, epochs=50)
    print('Saving model...')
    model.save()
    print('Evaluating model...')
    model.evaluate(train_ds)  # TODO: separate an eval set
    print('Done.')
