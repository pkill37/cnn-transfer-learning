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

AUGMENTATIONS = [c for j in range(1, len(AUGMENTATIONS)+1) for c in itertools.combinations(AUGMENTATIONS, j)]


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

    def _correct(img):
        """
        Normalize PIL image

        Normalizes luminance to (mean,std)=(0,1), and applies a [1%, 99%] contrast stretch
        """
        img_y, img_b, img_r = img.convert('YCbCr').split()

        img_y_np = np.asarray(img_y).astype(float)

        img_y_np /= 255
        img_y_np -= img_y_np.mean()
        img_y_np /= img_y_np.std()
        scale = np.max([np.abs(np.percentile(img_y_np, 1.0)),
                        np.abs(np.percentile(img_y_np, 99.0))])
        img_y_np = img_y_np / scale
        img_y_np = np.clip(img_y_np, -1.0, 1.0)
        img_y_np = (img_y_np + 1.0) / 2.0

        img_y_np = (img_y_np * 255 + 0.5).astype(np.uint8)

        img_y = Image.fromarray(img_y_np)

        img_ybr = Image.merge('YCbCr', (img_y, img_b, img_r))
        return img_ybr.convert('RGB')

    img = Image.open(filename).convert('RGB')
    img = _crop(img)
    img = _resize(img, target_size)
    img = _correct(img)
    return img


def augment(img, i):
    for f in AUGMENTATIONS[i]:
        img = f(img)
    return img


def process(images_path, descriptions_filename, target_img_size, target_m):
    with open(descriptions_filename, 'r', newline='') as f:
        x = []
        y = []

        # Dynamically build x and y by iterating through each line in the ground-truth CSV
        for i, row in enumerate(tqdm(csv.DictReader(f))):
            # Construct image filename from the given images directory and the ISIC image ID in the CSV
            image_filename = os.path.join(images_path, row['image_id']+'.jpg')

            # Read and decode the image and label in the aforementioned filename
            img = load_image(image_filename, target_img_size)
            label = int(float(row['melanoma']))
            x.append(img)
            y.append(label)

        assert len(x) == len(y)

        # Augment minority until classes are balanced and augmentation goal is reached
        if target_m:
            assert target_m > len(y)

            # Group data into classes for class balancing
            positive = [(xv, yv) for (xv,yv) in zip(x,y) if yv == 1]
            negative = [(xv, yv) for (xv,yv) in zip(x,y) if yv == 0]
            minority = helpers.smallest(positive, negative)
            majority = helpers.biggest(positive, negative)
            assert minority and majority

            for S in [minority, majority]:
                augmentation_factor = (target_m//2) // len(S)
                assert len(AUGMENTATIONS) >= augmentation_factor
                S += [(augment(xv, j), yv) for (xv,yv) in S for j in range(augmentation_factor)]

            # Construct NumPy ndarrays out of the lists
            total = minority + majority
            x = np.array([np.asarray(xv, dtype='float32') for (xv,yv) in total], dtype='float32')
            y = np.array([yv for (xv,yv) in total], dtype='float32')
        else:
            x = np.array([np.asarray(xv, dtype='float32') for xv in x], dtype='float32')
            y = np.array(y, dtype='float32')

        assert x.shape[0] == y.shape[0]
        return x, y


def load(preprocessed_dataset_filename):
    dataset = np.load(preprocessed_dataset_filename)
    x = dataset['x']
    y = dataset['y']
    return x, y


def plot(x, y):
    fig = plt.figure(figsize=(15, 20))
    rows = 4
    columns = 9
    for cell in range(1, columns*rows+1):
        plt.subplot(rows, columns, cell)
        try:
            plt.title('Melanoma' if y[cell] == 1 else 'Non melanoma')
            plt.imshow(x[cell])
        except:
            pass
        plt.axis('off')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str)
    parser.add_argument('--descriptions', type=str)
    parser.add_argument('--pretrained-model', type=str, choices=['vgg16', 'inceptionv3'])
    parser.add_argument('--total-samples', type=int)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    target_img_size = models.IMG_SHAPE[args.pretrained_model]

    x, y = process(args.images, args.descriptions, target_img_size, args.total_samples)
    np.savez_compressed(args.output, x=x, y=y)
