import tensorflow as tf
import numpy as np
import os
import math
import csv
from tqdm import tqdm
import helpers
from PIL import Image
import argparse
import matplotlib.pyplot as plt
import itertools
import models


AUGMENTATIONS = [
    lambda x: x.transpose(Image.FLIP_LEFT_RIGHT),
    lambda x: x.transpose(Image.FLIP_TOP_BOTTOM),
    lambda x: x.transpose(Image.ROTATE_90),
    lambda x: x.transpose(Image.ROTATE_180),
    lambda x: x.transpose(Image.ROTATE_270),
]

AUGMENTATIONS = [c for j in range(len(AUGMENTATIONS)+1) for c in itertools.combinations(AUGMENTATIONS, j)]


def load_image(filename, target_size):
    def _crop(img):
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

    def _resize(img, target_size):
        return img.resize(target_size, Image.NEAREST)

    img = Image.open(filename).convert('RGB')
    img = _crop(img)
    img = _resize(img, target_size)
    return img


def augment(img, i):
    for f in AUGMENTATIONS[i]:
        img = f(img)
    return img


def process(images_path, descriptions_filename, target_size, augmentation_factor):
    with open(descriptions_filename, 'r', newline='') as f:
        x = []
        y = []

        # Dynamically build x and y by iterating through each line in the ground-truth CSV
        for i, row in enumerate(tqdm(csv.DictReader(f))):
            # Construct image filename from the given images directory and the ISIC image ID in the CSV
            image_filename = os.path.join(images_path, row['image_id']+'.jpg')

            # Read and decode the image and label in the aforementioned filename
            img = load_image(image_filename, target_size)
            label = int(float(row['melanoma']))

            # Repeat until we reach the desired augmentation factor
            for j in range(augmentation_factor):
                # Apply j-th augmentation
                tmp = augment(img, j)
                # Convert to array
                tmp = np.asarray(tmp, dtype='float32')
                x.append(tmp)
                y.append(label)

        # Construct NumPy ndarrays out of the lists
        x = np.array(x, dtype='float32')
        y = np.array(y, dtype='float32')
        assert x.shape[0] == y.shape[0]

        return x, y


def load(preprocessed_dataset_filename):
    dataset = np.load(preprocessed_dataset_filename)
    x = dataset['x']
    y = dataset['y']
    return x, y


def plot(array, ncols):
    nrows = np.math.ceil(len(array)/float(ncols))
    cell_w = array.shape[2]
    cell_h = array.shape[1]
    channels = array.shape[3]
    result = np.zeros((cell_h*nrows, cell_w*ncols, channels), dtype=array.dtype)
    for i in range(0, nrows):
        for j in range(0, ncols):
            result[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w, :] = array[i*ncols+j]
    plt.figure()
    plt.axis('off')
    plt.imshow(result)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str)
    parser.add_argument('--descriptions', type=str)
    parser.add_argument('--pretrained-model', type=str, choices=['vgg16', 'inceptionv3'])
    parser.add_argument('--augmentation-factor', type=int)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    _, _, target_size = getattr(models, args.pretrained_model)(extract_until=1, freeze_until=0, l1=None, l2=None)

    x, y = process(args.images, args.descriptions, target_size, args.augmentation_factor)
    np.savez_compressed(args.output, x=x, y=y)

    plot(x/255, ncols=args.augmentation_factor)
