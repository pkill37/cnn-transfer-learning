import tensorflow as tf
import numpy as np
import os
import math
import re
import json
import sklearn.model_selection


def list_images(directory):
    return [os.path.join(root, f) for root, _, files in os.walk(directory) for f in files if re.match(r'ISIC_\d+\.jpeg', f)]


def list_descriptions(directory):
    return [os.path.join(root, f) for root, _, files in os.walk(directory) for f in files if re.match(r'ISIC_\d+', f)]


def load_img(filename, target_size, color_mode):
    obj = tf.keras.preprocessing.image.load_img(filename, target_size=target_size, color_mode=color_mode)
    arr = tf.keras.preprocessing.image.img_to_array(obj, data_format='channels_last', dtype='float32')
    return arr


def load_label(filename):
    with open(filename) as f:
        data = json.load(f)
    return 1 if data['meta']['clinical']['benign_malignant'] == 'benign' else 0


def load_data(images_filenames, descriptions_filenames, img_height, img_width, begin=None, end=None):
    assert len(images_filenames) == len(descriptions_filenames)

    # Load everything
    if not begin and not end:
        begin = 0
        end = len(images_filenames)
    # Load from the interval [begin, end]
    else:
        begin = max(begin, 0)
        end = min(end, len(images_filenames))
    assert begin < end

    # Load RGB image into h*w*3 matrix
    x = np.array([load_img(image, (img_height, img_width), 'rgb') for image in images_filenames[begin:end]], dtype='float32')
    assert x.shape == (end-begin, img_height, img_width, 3)

    # Load binary labels into column vector
    y = np.array([load_label(description) for description in descriptions_filenames[begin:end]], dtype='int')
    assert y.shape == (end-begin,)

    # Make sure classes are balanced
    unique, counts = np.unique(y, return_counts=True)
    assert counts[0] == counts[1] and unique[0] == 0

    return x, y


class BinaryLabelImageSequence(tf.keras.utils.Sequence):
    def __init__(self, x, y, img_height, img_width, batch_size, augment, seed=None):
        self.x = x
        self.y = y

        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.augment = augment
        self.seed = seed

        self.imgaug = tf.keras.preprocessing.image.ImageDataGenerator(
            # Standardization
            preprocessing_function=tf.keras.applications.vgg16.preprocess_input,
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
            brightness_range=(0,1),
            shear_range=0.5,
            zoom_range=0.2,
            channel_shift_range=0.005,
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


def split_data(x, y, split):
    assert split[0]+split[1]+split[2] == 1

    # First split everything into train and test
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=split[2], stratify=y)

    # Then split train into train and validation
    x_train, x_validation, y_train, y_validation = sklearn.model_selection.train_test_split(x_train, y_train, test_size=split[1], stratify=y_train)

    print('train', x_train.shape, y_train.shape)
    print('validation', x_validation.shape, y_validation.shape)
    print('test', x_test.shape, y_test.shape)

    return x_train, y_train, x_validation, y_validation, x_test, y_test


def generators(images_path, descriptions_path, img_height, img_width, split, batch_size, augmentation):
    x, y = load_data(
        images_filenames=list_images(images_path),
        descriptions_filenames=list_descriptions(descriptions_path),
        img_height=img_height,
        img_width=img_width,
    )

    x_train, y_train, x_validation, y_validation, x_test, y_test = split_data(x, y, split)

    train_generator = BinaryLabelImageSequence(x=x_train, y=y_train, img_height=img_height, img_width=img_width, batch_size=batch_size, augment=augmentation)
    validation_generator = BinaryLabelImageSequence(x=x_validation, y=y_validation, img_height=img_height, img_width=img_width, batch_size=batch_size, augment=False)
    test_generator = BinaryLabelImageSequence(x=x_test, y=y_test, img_height=img_height, img_width=img_width, batch_size=batch_size, augment=False)

    return train_generator, validation_generator, test_generator
