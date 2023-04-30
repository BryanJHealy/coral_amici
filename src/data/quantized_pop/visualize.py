import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


if __name__ == '__main__':
    ds = tfds.load('quantized_pop/s960b', split='train', shuffle_files=True)
    examples = ds.take(10)
    for example in examples:
        a = example['accompaniment']
        m = example['melody']

        fig = plt.figure()
        fig.add_subplot(2, 1, 1)
        plt.imshow(m[0])
        plt.axis('off')
        plt.title('Melody')

        fig.add_subplot(2, 1, 2)
        plt.imshow(a[0])
        plt.axis('off')
        plt.title('Accompaniment')
        plt.show()

        continue