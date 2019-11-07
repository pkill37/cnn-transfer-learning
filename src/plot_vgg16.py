import os
import re
import argparse
import csv
import json

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import helpers


def truncate(f):
    return float("{:.2e}".format(f))

def truncate_floats(stats):
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
    stats_truncated = truncate_floats(stats)

    # Build header
    latex = '\\begin{table}[ht]\n'
    latex += '\\centering\n'
    latex += '\\begin{tabular}{ '
    latex += '|c|c|c|c|c|c|'
    latex += ' }\n'
    latex += '\\hline\n'
    latex += '$e$ & $f$ & $\lambda$ & $A_{train}$ & $A_{val}$ & $A_{test}$ \\\\\n'
    latex += '\\hline\n'

    mean = truncate(np.mean(np.array([stat['test_acc'] for stat in stats])))
    std = truncate(np.std(np.array([stat['test_acc'] for stat in stats])))

    # Build body
    for stat in stats_truncated:
        latex += f"{stat['hyperparameters']['extract']} & {stat['hyperparameters']['freeze']} & {stat['hyperparameters']['lambda']} & {stat['train_acc'][-1]} & {stat['val_acc'][-1]} & {stat['test_acc']} \\\\\n"
    latex += '\\hline\n'
    latex += '\\end{tabular}\n'
    latex += '\\caption{'
    latex += f"Average performance ${mean} \pm {std}$"
    latex += '}\n'
    latex += '\\label{table:foobar}\n'
    latex += '\\end{table}\n'

    print(latex)

    with open(target_file, 'w') as f:
        f.write(latex)


def plot_accuracy_vs_lambda(stats, plots):
    fig, ax = plt.subplots(1, 1, constrained_layout=True)

    x = np.array([stat['hyperparameters']['lambda'] for stat in stats])
    y1 = np.array([stat['train_acc'][-1] for stat in stats])
    y2 = np.array([stat['val_acc'][-1] for stat in stats])
    y3 = np.array([stat['test_acc'] for stat in stats])
    x, y1, y2, y3 = zip(*sorted(zip(x, y1, y2, y3)))
    x = np.array(x)
    y1 = np.array(y1)
    y2 = np.array(y2)
    y3 = np.array(y3)

    ax.plot(x, y1, '.b-', label="$A_{train}$")
    ax.plot(x, y2, '.r-', label="$A_{val}$")
    #ax.plot(x, y3, '.g-', label="$A_{test}$")
    ax.legend()
    ax.grid(True)

    ax.set_xscale("log")
    ax.set_yscale("linear")
    ax.set_xlabel("$\lambda$")
    ax.set_ylabel("Accuracy")

    plt.savefig(os.path.join(plots, f'lambda_study.png'))
    plt.close()


def plot_accuracy_vs_units(stats, plots):
    fig, ax = plt.subplots(1, 1, constrained_layout=True)

    x = np.array([stat['hyperparameters']['units'] for stat in stats])
    y1 = np.array([stat['train_acc'][-1] for stat in stats])
    y2 = np.array([stat['val_acc'][-1] for stat in stats])
    y3 = np.array([stat['test_acc'] for stat in stats])
    x, y1, y2, y3 = zip(*sorted(zip(x, y1, y2, y3)))
    x = np.array(x)
    y1 = np.array(y1)
    y2 = np.array(y2)
    y3 = np.array(y3)

    ax.plot(x, y1, '.b-', label="$A_{train}$")
    ax.plot(x, y2, '.r-', label="$A_{val}$")
    #ax.plot(x, y3, '.g-', label="$A_{test}$")
    ax.legend()
    ax.grid(True)

    ax.set_xscale("log", basex=2)
    ax.set_yscale("linear")
    ax.set_xlabel("$u$")
    ax.set_ylabel("Accuracy")

    plt.savefig(os.path.join(plots, f'units_study.png'))
    plt.close()


def plot_accuracy_vs_dropout(stats, plots):
    fig, ax = plt.subplots(1, 1, constrained_layout=True)

    x = np.array([stat['hyperparameters']['dropout'] for stat in stats])
    y1 = np.array([stat['train_acc'][-1] for stat in stats])
    y2 = np.array([stat['val_acc'][-1] for stat in stats])
    y3 = np.array([stat['test_acc'] for stat in stats])
    x, y1, y2, y3 = zip(*sorted(zip(x, y1, y2, y3)))
    x = np.array(x)
    y1 = np.array(y1)
    y2 = np.array(y2)
    y3 = np.array(y3)

    ax.plot(x, y1, '.b-', label="$A_{train}$")
    ax.plot(x, y2, '.r-', label="$A_{val}$")
    #ax.plot(x, y3, '.g-', label="$A_{test}$")
    ax.legend()
    ax.grid(True)

    ax.set_xscale("linear")
    ax.set_yscale("linear")
    ax.set_xlabel("$d$")
    ax.set_ylabel("Accuracy")

    plt.savefig(os.path.join(plots, f'dropout_study.png'))
    plt.close()


def plot_accuracy_loss_vs_epochs(stats, plots):
    for stat in stats:
        fig, ax = plt.subplots(1, 2, constrained_layout=True, figsize=(11, 4))

        x = np.array(list(range(1, len(stat['train_acc']))))
        y1 = np.array(stat['train_acc'])
        y2 = np.array(stat['val_acc'])
        y3 = np.array(stat['train_loss'])
        y4 = np.array(stat['val_loss'])
        x, y1, y2, y3, y4 = zip(*sorted(zip(x, y1, y2, y3, y4)))
        x = np.array(x)
        y1 = np.array(y1)
        y2 = np.array(y2)
        y3 = np.array(y3)
        y4 = np.array(y4)

        ax[0].plot(x, y1, '.b-', label="$A_{train}$")
        ax[0].plot(x, y2, '.r-', label="$A_{val}$")
        ax[0].legend()
        ax[0].grid(True)
        ax[0].set_xscale("linear")
        ax[0].set_yscale("linear")
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Accuracy")

        ax[1].plot(x, y3, '.b-', label="$J_{train}$")
        ax[1].plot(x, y4, '.r-', label="$J_{val}$")
        ax[1].legend()
        ax[1].grid(True)
        ax[1].set_xscale("linear")
        ax[1].set_yscale("linear")
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Cost")

        plt.savefig(os.path.join(plots, f'{stat["id"]}_epochs.png'))
        plt.close()


def main(experiments, plots):
    helpers.create_or_recreate_dir(plots)

    stats = os.path.join(experiments, 'stats.json')
    with open(stats) as f:
        stats = json.loads(f.read())
    stats = truncate_floats(stats)

    # Tabulate top-20 results sorted by performance
    tabulate(sorted(stats, key=lambda x: x['test_acc'], reverse=True), os.path.join(plots, 'table.tex'))

    # Study progress over epochs
    try:
        plot_accuracy_loss_vs_epochs(stats, plots)
    except:
        pass
    # Study effect of lambda
    try:
        plot_accuracy_vs_lambda(stats, plots)
    except:
        pass
    # Study effect of units
    try:
        plot_accuracy_vs_units(stats, plots)
    except:
        pass
    # Study effect of dropout
    try:
        plot_accuracy_vs_dropout(stats, plots)
    except:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments', type=helpers.is_dir, required=True)
    parser.add_argument('--plots', type=helpers.is_dir, required=True)
    args = parser.parse_args()

    main(args.experiments, args.plots)
