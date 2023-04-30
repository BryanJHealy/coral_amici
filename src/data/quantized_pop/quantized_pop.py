"""quantized_pop dataset."""

import tensorflow_datasets as tfds
from os.path import join
from src.util.data_handling import generate_training_sequences
from numpy import float64

VERSION = tfds.core.Version("0.1.0")


class PopConfig(tfds.core.BuilderConfig):
    """BuilderConfig for Pop909."""

    def __init__(self, *, num_samples=None, sample_frequencies=None, vocab_size=128, binary_activations=False,
                 data_size=32, accompaniment=True, **kwargs):
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
        self.binary_activations = binary_activations
        self.data_size = data_size
        self.accompaniment = accompaniment


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
            sample_frequencies=[4, 8, 16, 32, 64],
            vocab_size=128,
            binary_activations=False
        ),
        PopConfig(
            name="s960b",
            description="960 samples per sequence, binary activations",
            num_samples=960,
            sample_frequencies=[16],
            vocab_size=128,
            binary_activations=True
        ),
        PopConfig(
            name="s640",
            description="640 samples per sequence",
            num_samples=640,
            sample_frequencies=[4, 8, 16, 32, 64],
            vocab_size=128,
            binary_activations=False
        ),
        PopConfig(
            name="s240",
            description="240 samples per sequence",
            num_samples=240,
            sample_frequencies=[4, 8, 16, 32, 64],
            vocab_size=128,
            binary_activations=False
        ),
        PopConfig(
            name="s60",
            description="60 samples per sequence",
            num_samples=60,
            sample_frequencies=[4, 8],
            vocab_size=128,
            binary_activations=False
        ),
        PopConfig(
            name="s15",
            description="15 samples per sequence",
            num_samples=15,
            sample_frequencies=[1],
            vocab_size=128,
            binary_activations=False
        ),
        PopConfig(
            name="s15r",
            description="15 samples per sequence, reproduction",
            num_samples=15,
            sample_frequencies=[1],
            vocab_size=128,
            binary_activations=False,
            accompaniment=False
        ),
        PopConfig(
            name="s15b",
            description="15 samples per sequence, binary activations",
            num_samples=15,
            sample_frequencies=[1],
            vocab_size=128,
            binary_activations=True
        ),
        PopConfig(
            name="s3840",
            description="3840 samples per sequence",
            num_samples=3840,
            sample_frequencies=[4, 8, 16, 32, 64],
            vocab_size=128,
            binary_activations=False
        ),
    ]

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        num_samples = self._builder_config.num_samples
        sample_depth = self._builder_config.vocab_size
        if self._builder_config.binary_activations:
            sample_depth = sample_depth//self._builder_config.data_size
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'melody': tfds.features.Tensor(shape=(1, sample_depth, num_samples), dtype=float64),
                'accompaniment': tfds.features.Tensor(shape=(1, sample_depth, num_samples), dtype=float64),
            }),
            # (input, target) tuple used if `as_supervised=True` in `builder.as_dataset`
            supervised_keys=('melody', 'accompaniment'),
            homepage='https://github.com/BryanJHealy/coral_amici',
            disable_shuffling=False
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        path = dl_manager.download_and_extract('https://github.com/music-x-lab/POP909-Dataset/raw/master/POP909.zip')
        return {
            'train': self._generate_examples(path, list(range(1, 910))),
        }

    def _generate_examples(self, path, files):
        """Yields examples."""
        for song_num in files:
            filepath = join(path, 'POP909', f'{song_num:03d}', f'{song_num:03d}.mid')
            tracks = ('MELODY', 'PIANO') if self._builder_config.accompaniment else ('PIANO', 'PIANO')
            sequences = generate_training_sequences(filepath=filepath, instrument_tracks=tracks,
                                                    num_samples=self._builder_config.num_samples,
                                                    sample_frequencies=self._builder_config.sample_frequencies,
                                                    vocab_size=self._builder_config.vocab_size,
                                                    binary_activations=self._builder_config.binary_activations,
                                                    add_batch_dimension=True)
            for seq_idx in range(len(sequences)):
                key = f'{song_num}_{seq_idx}'
                yield key, {
                    'melody': sequences[seq_idx][0],
                    'accompaniment': sequences[seq_idx][1],
                }
