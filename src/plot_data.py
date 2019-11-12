import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

import helpers
import data


def plot_data(x, y, target):
    fig = plt.figure()
    rows = 3
    columns = 6
    for cell in range(1, columns*rows+1):
        i = np.random.randint(0, x.shape[0])
        plt.subplot(rows, columns, cell)
        plt.title('Melanoma' if y[i] == 1 else 'Non melanoma', fontsize=6)
        plt.imshow(x[i]/255)
        plt.axis('off')

    plt.savefig(target)
    plt.close()

def plot_barplot(x, y, target_file):
    fig = plt.figure()
    for cell in range(1, columns*rows+1):
        i = np.random.randint(0, x.shape[0])
        plt.subplot(rows, columns, cell)
        plt.title('Melanoma' if y[i] == 1 else 'Non melanoma')
        plt.imshow(x[i]/255)
        plt.axis('off')

    plt.savefig(os.path.join(target_dir, f'extract{extract}_lambda{l2}.png'))
    plt.close()


def main(train, test, target):
    x_train, y_train = data.load(train)
    x_test, y_test = data.load(test)

    helpers.create_or_recreate_dir(target)
    plot(x_train, y_train, os.path.join(target, 'train.png'))
    plot(x_test, y_test, os.path.join(target, 'test.png'))
    plot_barplot(x_train + x_test, y_train + y_test, os.path.join(target_dir, 'barplot.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=helpers.is_file, required=True)
    parser.add_argument('--test', type=helpers.is_file, required=True)
    parser.add_argument('--target', type=helpers.is_dir, required=True)
    args = parser.parse_args()

    main(args.train, args.test, args.target)
