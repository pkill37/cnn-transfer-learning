import os
import re
import argparse
import csv
import json

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics

import helpers
import data


def plot_train_loss(stat, target_file):
    train_loss = stat['train_loss']
    epochs = range(len(train_loss))

    plt.figure()
    plt.plot(epochs, train_loss, 'b', label='Train cost')
    plt.ylabel('Cost')
    plt.xlabel('Epoch')
    #plt.ylim(0, 1)
    plt.title(f'Train cost')
    plt.legend()

    plt.savefig(target_file)


def main(experiments, plots):
    stats = os.path.join(experiments, 'stats.json')
    with open(stats) as f:
        stats = json.loads(f.read())

    # Pick top 10 sorted by F1
    stats = sorted(stats, key=lambda x: x['classification_report']['weighted avg']['f1-score'])
    stats = stats[:10]

    for i, stat in enumerate(stats):
        target_path = os.path.join(plots, stat['id'])
        helpers.create_or_recreate_dir(target_path)
        target_file = os.path.join(target_path, f'{i}_cost.png')
        plot_train_loss(stat, target_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments', type=helpers.is_dir, required=True)
    parser.add_argument('--plots', type=helpers.is_dir, required=True)
    args = parser.parse_args()

    main(args.experiments, args.plots)
