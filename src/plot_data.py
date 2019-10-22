import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

import helpers
import data


def plot(x, y, target_file):
    fig = plt.figure(figsize=(15, 20))
    rows = 4
    columns = 6
    for cell in range(1, columns*rows+1):
        i = np.random.randint(0, x.shape[0])
        plt.subplot(rows, columns, cell)
        plt.title('Melanoma' if y[i] == 1 else 'Non melanoma')
        plt.imshow(x[i]/255)
        plt.axis('off')

    plt.savefig(os.path.join(target_dir, f'extract{extract}_lambda{l2}.png'))
    plt.close()


def main(train, test, target_dir):
    x_train, y_train = data.load(train)
    x_test, y_test = data.load(test)

    helpers.create_or_recreate_dir(target_dir)
    plot(x_train, y_train, os.path.join(target_dir, 'train.png'))
    plot(x_test, y_test, os.path.join(target_dir, 'test.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=helpers.is_file, required=True)
    parser.add_argument('--test', type=helpers.is_file, required=True)
    parser.add_argument('--target', type=helpers.is_dir, required=True)
    args = parser.parse_args()

    main(args.train, args.test, args.target)
