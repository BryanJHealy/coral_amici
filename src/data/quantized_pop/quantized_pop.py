"""quantized_pop dataset."""

import tensorflow_datasets as tfds
import os
# import sys
# sys.path.insert(0, '/media/steamgames/coral/coral_amici/src')
import util.data_handling as dh
from tensorflow import float64


VERSION = tfds.core.Version("0.1.0")


class PopConfig(tfds.core.BuilderConfig):
    """BuilderConfig for Pop909."""

    def __init__(self, *, num_samples=None, sample_frequencies=None, vocab_size=128, **kwargs):
        """BuilderConfig for Pop909.
    Args:
      num_samples: int. The total number of samples per sequence.
      sample_frequencies: [int]. List of frequencies to sample data. e.g. 60 samples/sec
      **kwargs: keyword arguments forwarded to super.
    """
        super(PopConfig, self).__init__(version=VERSION, **kwargs)
        self.num_samples = num_samples
        self.sample_frequencies = sample_frequencies
        self.vocab_size = vocab_size


class QuantizedPop(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for quantized_pop dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }
    BUILDER_CONFIGS = [
        PopConfig(
            name="s960",
            description="960 samples per sequence",
            num_samples=960,
            sample_frequencies=[4,8,16,32,64],
            vocab_size=128
        ),
        PopConfig(
            name="s640",
            description="640 samples per sequence",
            num_samples=640,
            sample_frequencies=[4,8,16,32,64],
            vocab_size=128
        ),
        PopConfig(
            name="s3840",
            description="3840 samples per sequence",
            num_samples=3840,
            sample_frequencies=[4,8,16,32,64],
            vocab_size=128
        ),
    ]

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(quantized_pop): Specifies the tfds.core.DatasetInfo object
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                'melody': tfds.features.Tensor(shape=(1, 128, 960), dtype=float64),
                'accompaniment': tfds.features.Tensor(shape=(1, 128, 960), dtype=float64),
            }),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=('melody', 'accompaniment'),  # Set to `None` to disable
            homepage='https://github.com/BryanJHealy/coral_amici',
            disable_shuffling=False
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        path = dl_manager.download_and_extract('https://github.com/music-x-lab/POP909-Dataset/raw/master/POP909.zip')

        num_files = 909
        files = list(range(1, num_files + 1))
        # np.random.shuffle(files)
        # split = int(0.9 * num_files)
        # train_set = files[:split]
        train_set = files
        # test_set = files[split:]
        return {
            'train': self._generate_examples(path, train_set),
            # 'test': self._generate_examples(path, test_set),
        }

    def _generate_examples(self, path, files):
        """Yields examples."""
        for song_num in files:
            sample_frequencies = self._builder_config.sample_frequencies
            num_samples = self._builder_config.num_samples
            vocab_size = self._builder_config.vocab_size
            fpath = os.path.join(path, 'POP909', f'{song_num:03d}', f'{song_num:03d}.mid')
            sequences = dh.generate_training_sequences(filepath=fpath, instrument_tracks=('MELODY', 'PIANO'),
                                                       num_samples=num_samples, sample_frequencies=sample_frequencies,
                                                       vocab_size=vocab_size, add_batch_dimension=True)
            for seq_idx in range(len(sequences)):
                key = f'{song_num}_{seq_idx}'
                yield key, {
                    'melody': sequences[seq_idx][0],
                    'accompaniment': sequences[seq_idx][1],
                }
