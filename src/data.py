import argparse
import itertools
import os
import math
import csv

import sklearn.model_selection
import tensorflow as tf
import numpy as np
import PIL

import helpers


AUGMENTATIONS = [
    lambda x: x.transpose(PIL.Image.FLIP_LEFT_RIGHT),
    lambda x: x.transpose(PIL.Image.FLIP_TOP_BOTTOM),
    lambda x: x.transpose(PIL.Image.ROTATE_90),
    lambda x: x.transpose(PIL.Image.ROTATE_180),
    lambda x: x.transpose(PIL.Image.ROTATE_270),
]

AUGMENTATIONS = [c for j in range(1, len(AUGMENTATIONS)+1) for c in itertools.combinations(AUGMENTATIONS, j)]


def standardize(x, mean, std):
    x[..., 0] -= mean[0]
    x[..., 1] -= mean[1]
    x[..., 2] -= mean[2]

    x[..., 0] /= std[0]
    x[..., 1] /= std[1]
    x[..., 2] /= std[2]

    return x


def load_image(filename, target_size):
    assert target_size[0] == target_size[1]

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
        return img.resize(target_size, PIL.Image.NEAREST)

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

        img_y = PIL.Image.fromarray(img_y_np)

        img_ybr = PIL.Image.merge('YCbCr', (img_y, img_b, img_r))
        return img_ybr.convert('RGB')

    img = PIL.Image.open(filename).convert('RGB')
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
        for i, row in enumerate(csv.DictReader(f)):
            print(i)

            # Construct image filename from the given images directory and the ISIC image ID in the CSV
            image_filename = os.path.join(images_path, row['image']+'.jpg')

            # Read and decode the image and label in the aforementioned filename
            img = load_image(image_filename, target_img_size)
            label = int(float(row['MEL'])) # 1 is melanoma, 0 is non-melanoma
            x.append(img)
            y.append(label)

        # Ensure size consistency between lists
        assert len(x) == len(y)

        # Augment minority until classes are balanced and augmentation goal is reached
        if target_m:
            assert target_m > len(y)
            print(f'Augmenting {len(y)} samples to approximately {target_m} class-balanced samples...')

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

        # Ensure size consistency between x matrix and y vector
        assert x.shape[0] == y.shape[0]

        # Ensure the preprocessed dataset's classes were reasonably balanced
        _, counts = np.unique(y, return_counts=True)
        assert abs(counts[0] - counts[1]) < 100

        return x, y


def load(preprocessed_dataset_filename):
    dataset = np.load(preprocessed_dataset_filename)
    return dataset['x'], dataset['y']


def save(x, y, output):
    np.savez_compressed(os.path.join(output, os.path.basename(output)), x=x, y=y)
    for i, (x_i, y_i) in enumerate(zip(x, y)):
        img = PIL.Image.fromarray(x_i.astype(np.uint8))
        img.save(os.path.join(output, f'{i}_{y_i}.jpg'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=helpers.is_dir, required=True)
    parser.add_argument('--descriptions', type=helpers.is_file, required=True)
    parser.add_argument('--target-size', type=int, required=True)
    parser.add_argument('--target-samples', type=int, required=True)
    parser.add_argument('--output', type=helpers.is_dir, required=True)
    args = parser.parse_args()

    x, y = process(args.images, args.descriptions, (args.target_size, args.target_size), args.target_samples)
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, train_size=0.85, shuffle=True, stratify=y)
    del x
    del y
    save(x_train, y_train, os.path.join(args.output, 'train'))
    save(x_test, y_test, os.path.join(args.output, 'test'))
