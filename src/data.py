import tensorflow as tf
import numpy as np
import os
import math
import csv
from tqdm import tqdm
import helpers
from PIL import Image
import argparse
import sklearn


def load_image(filename, target_size, color_mode, preprocess_function=None):
    # Read file into PIL Image instance
    img = Image.open(filename).convert('RGB')

    # Run image through preprocessing pipeline
    if preprocess_function:
        img = preprocess_function(img, target_size)

    # Convert image to NumPy array
    return np.asarray(img, dtype='float32')


def preprocess_image(img, target_size):
    def _crop_center(img):
        width, height = img.size
        if width == height:
            return img

        length = min(width, height)

        left = (width - length) // 2
        upper = (height - length) // 2
        right = left + length
        lower = upper + length

        box = (left, upper, right, lower)
        return img.crop(box)

    def _resize(img):
        return img.resize(target_size, Image.NEAREST)

    return _resize(img)


def preprocess_dataset(images_path, descriptions_filename, preprocessed_dataset_filename, img_height, img_width):
    with open(descriptions_filename, 'r', newline='') as f:
        x = []
        y = []

        # Dynamically build x and y by iterating through each line in the ground-truth CSV
        for row in tqdm(csv.DictReader(f)):
            # Construct image filename from the given images directory and the ISIC image ID in the CSV
            image_filename = os.path.join(images_path, row['image_id']+'.jpg')

            # Load and preprocess the image in the aforementioned filename
            x.append(load_image(image_filename, (img_height, img_width), 'rgb', preprocess_image))

            # Label of the first classification task (melanoma vs nevus and seborrheic keratosis)
            y.append(int(float(row['melanoma'])))

        # Construct NumPy ndarrays out of the lists
        x = np.array(x, dtype='float32')
        y = np.array(y, dtype='float32')
        assert x.shape[0] == y.shape[0]

        # Store final, preprocessed, compressed dataset
        np.savez_compressed(preprocessed_dataset_filename, x=x, y=y)
        return x, y


def load_dataset(preprocessed_dataset_filename):
    dataset = np.load(preprocessed_dataset_filename)
    x = dataset['x']
    y = dataset['y']
    class_weights = sklearn.utils.class_weight.compute_class_weight('balanced', np.unique(y), y)

    return x, y, class_weights


class BinaryLabelImageSequence(tf.keras.utils.Sequence):
    def __init__(self, x, y, batch_size, augment, preprocess_input, seed=None):
        self.x = x
        self.y = y

        self.batch_size = batch_size
        self.augment = augment
        self.seed = seed

        self.imgaug = tf.keras.preprocessing.image.ImageDataGenerator(
            # Standardization
            preprocessing_function=preprocess_input,
            rescale=None,
            samplewise_center=False,
            samplewise_std_normalization=False,
            featurewise_center=False,
            featurewise_std_normalization=False,
            zca_whitening=False,

            # Allowed transformations
            horizontal_flip=True,
            vertical_flip=True,
            width_shift_range=0.1,
            height_shift_range=0.1,
            rotation_range=360,
            zoom_range=0.2,
        )

    def __len__(self):
        return int(math.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        x_batch = self.x[idx*self.batch_size : (1+idx)*self.batch_size]
        y_batch = self.y[idx*self.batch_size : (1+idx)*self.batch_size]

        # Standardize and augment batch
        for i in range(len(x_batch)):
            x_batch[i] = self.imgaug.standardize(x_batch[i])
            if self.augment:
                params = self.imgaug.get_random_transform(x_batch[i].shape, self.seed)
                x_batch[i] = self.imgaug.apply_transform(x_batch[i], params)

        return x_batch, y_batch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str)
    parser.add_argument('--descriptions', type=str)
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()

    if args.images and args.descriptions:
        x, y = preprocess_dataset(args.images, args.descriptions, args.dataset, 224, 224)
    else:
        x, y = load_dataset(args.dataset)

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15, 20))
    rows = 4
    columns = 6
    for cell in range(1, columns*rows+1):
        i = np.random.randint(0, x.shape[0])
        plt.subplot(rows, columns, cell)
        plt.title('Melanoma' if y[i] == 1 else 'Non melanoma')
        plt.imshow(x[i]/255)
        plt.axis('off')
    plt.show()
