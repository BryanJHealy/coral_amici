# import argparse
import tensorflow as tf
import numpy as np
from src.models.lstm_accompaniment import LstmAccompaniment
import tensorflow_datasets as tfds


if __name__ == '__main__':
    # Command-line argument parsing
    # parser = argparse.ArgumentParser(description='Build quantized_pop if needed, then use it to train LSTM '
    #                                              'autoencoder to generate accompaniments for a given melody')

    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)

    samples_per_sequence = 15  # see src/data/QuantizedPop.BUILDER_CONFIGS for configurations
    batch_size = 64

    print('Loading dataset...')
    train_ds = tfds.load(f'quantized_pop/s{samples_per_sequence}', split='train[:1%]',
                         shuffle_files=True, as_supervised=True)

    print('Building model...')
    model = LstmAccompaniment(samples_per_sequence=samples_per_sequence, learning_rate=0.005)

    print('Training model...')
    model.train(train_ds=train_ds, batch_size=batch_size, epochs=1)

    print('Saving model...')
    model.save()

    print('Evaluating model...')
    eval_ds = tfds.load(f'quantized_pop/s{samples_per_sequence}', split='train[1:2%]',
                        shuffle_files=False, as_supervised=True)  # TODO: check if shuffling will occur between loads
    model.evaluate(train_ds)
    print('Done.')
