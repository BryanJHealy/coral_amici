# import argparse
import tensorflow as tf
import numpy as np
from src.models.lstm_vae import LstmVAE
import tensorflow_datasets as tfds


if __name__ == '__main__':
    # Command-line argument parsing
    # parser = argparse.ArgumentParser(description='Build quantized_pop if needed, then use it to train LSTM '
    #                                              'autoencoder to generate accompaniments for a given melody')

    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)

    samples_per_sequence = 15  # see src/data/QuantizedPop.BUILDER_CONFIGS for configurations
    batch_size = 32

    print('Building model...')
    model = LstmVAE(samples_per_sequence=samples_per_sequence, learning_rate=0.005, compression_factor=0.5,
                    batch_size=batch_size, eps_std=1.0)

    print('Loading accompaniment-recreation dataset...')
    train_ds = tfds.load(f'quantized_pop/s{samples_per_sequence}r', split='train[:10%]',
                         shuffle_files=True, as_supervised=True)

    print('Training model...')
    model.train(train_ds=train_ds, batch_size=batch_size, epochs=1)

    print('Loading accompaniment dataset...')
    train_ds = tfds.load(f'quantized_pop/s{samples_per_sequence}', split='train[:2%]',
                         shuffle_files=True, as_supervised=True)

    print('Training model...')
    model.train(train_ds=train_ds, batch_size=batch_size, epochs=1)

    print('Saving model...')
    model.save()

    print('Evaluating model...')
    eval_ds = tfds.load(f'quantized_pop/s{samples_per_sequence}', split='train[1:2%]',
                        shuffle_files=False, as_supervised=True)  # TODO: check if shuffling will occur between loads
    model.evaluate(eval_ds)
    print('Done.')
