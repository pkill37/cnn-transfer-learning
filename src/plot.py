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


def truncate_floats(stats):
    def truncate(f):
        return float("{:.4f}".format(f))

    for stat in stats:
        for h in stat['hyperparameters']:
            if isinstance(stat['hyperparameters'][h], float):
                stat['hyperparameters'][h] = truncate(stat['hyperparameters'][h])
        stat['train_acc'][-1] = truncate(stat['train_acc'][-1])
        stat['val_acc'][-1] = truncate(stat['val_acc'][-1])
        stat['test_acc'] = truncate(stat['test_acc'])
    return stats


def tabulate(stats, target_file):
    assert len(stats) > 0

    """
    \begin{center}
    \begin{tabular}{ |c|c|c| }
     \hline
     cell1 & cell2 & cell3 \\
     cell4 & cell5 & cell6 \\
     cell7 & cell8 & cell9 \\
     \hline
    \end{tabular}
    \end{center}
    """

    # Truncate floats to 2 decimal places for presentation purposes
    stats = truncate_floats(stats)

    # Build header
    latex = '\\begin{table}[ht]\n'
    latex += '\\centering\n'
    latex += '\\begin{tabular}{ '
    latex += '|c'*(len(stats[0]['hyperparameters']) + 3) + '|'
    latex += ' }\n'
    latex += '\\hline\n'
    for h in stats[0]['hyperparameters']:
        latex += f"{h} & "
    latex += "train acc & val acc & test acc \\\\\n"
    latex += '\\hline\n'

    # Build body
    for stat in stats:
        for h in stat['hyperparameters']:
            latex += f"{stat['hyperparameters'][h]} & "
        latex += f"{stat['train_acc'][-1]} & {stat['val_acc'][-1]} & {stat['test_acc']} \\\\\n"
    latex += '\\hline\n'
    latex += '\\end{tabular}\n'
    latex += '\\caption{Foobar}\n'
    latex += '\\label{table:foobar}\n'
    latex += '\\end{table}\n'

    print(latex)

    with open(target_file, 'w') as f:
        f.write(latex)


def main(experiments, plots):
    stats = os.path.join(experiments, 'stats.json')
    with open(stats) as f:
        stats = json.loads(f.read())

    # Sort by F1
    stats = sorted(stats, key=lambda x: x['classification_report']['weighted avg']['f1-score'], reverse=True)
    stats = stats[:20]

    # Tabulate results
    tabulate(stats, os.path.join(plots, 'table.tex'))

    # Plot graphs
    """
    for i, stat in enumerate(stats):
        target_path = os.path.join(plots, stat['id'])
        helpers.create_or_recreate_dir(target_path)
        target_file = os.path.join(target_path, f'{i}_cost.png')
        plot_train_loss(stat, target_file)
    """


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments', type=helpers.is_dir, required=True)
    parser.add_argument('--plots', type=helpers.is_dir, required=True)
    args = parser.parse_args()

    main(args.experiments, args.plots)
