"""quantized_pop dataset."""

import tensorflow_datasets as tfds


class QuantizedPop(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for quantized_pop dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(quantized_pop): Specifies the tfds.core.DatasetInfo object
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                'melody': tfds.features.Tensor(shape=(128, 900, None)),
                'accompaniment': tfds.features.Tensor(shape=(128, 900, None)),
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
        files = list(range(1, num_files+1))
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
        # TODO(quantized_pop): Yields (key, example) tuples from the dataset
        for f in path.glob('*.jpeg'):
            yield 'key', {
                'image': f,
                'label': 'yes',
            }
