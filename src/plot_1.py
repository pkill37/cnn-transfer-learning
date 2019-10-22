import os
import re
import argparse
import csv
import json

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import helpers


def truncate_floats(stats):
    def truncate(f):
        return float("{:.2e}".format(f))

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
    latex += '|c'*(len(stats_truncated[0]['hyperparameters']) + 3) + '|'
    latex += ' }\n'
    latex += '\\hline\n'
    for h in stats_truncated[0]['hyperparameters']:
        latex += f"{h} & "
    latex += "train acc & val acc & test acc \\\\\n"
    latex += '\\hline\n'

    # Build body
    for stat in stats_truncated:
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


def plot_accuracy_vs_lambda(stats, plots):
    extract = [18, 14, 10, 6, 3]
    freeze = [18, 14, 10, 6, 3, 0]
    cases = []
    for e in extract:
        case = []
        for f in freeze:
            if f <= e:
                case.append((e, f))
        cases.append(case)

    target_dir = os.path.join(plots, 'lambda_study')
    helpers.create_or_recreate_dir(target_dir)
    for case in cases:
        fig, axs = plt.subplots(1, len(case), constrained_layout=True, figsize=(3*len(case), 3))
        for i in range(len(case)):
            extract, freeze = case[i]
            stats_filtered = list(filter(lambda s: s['hyperparameters']['extract'] == extract and s['hyperparameters']['freeze'] == freeze, stats))
            x = np.array([stat['hyperparameters']['lambda'] for stat in stats_filtered])
            y1 = np.array([stat['train_acc'][-1] for stat in stats_filtered])
            y2 = np.array([stat['val_acc'][-1] for stat in stats_filtered])
            y3 = np.array([stat['test_acc'] for stat in stats_filtered])
            x, y1, y2, y3 = zip(*sorted(zip(x, y1, y2, y3)))
            x = np.array(x)
            y1 = np.array(y1)
            y2 = np.array(y2)
            y3 = np.array(y3)

            axs[i].set_title(f'Accuracy when $e = {extract}$ and $f = {freeze}$')
            axs[i].plot(x, y1, '.b-', label="$A_{train}$")
            axs[i].plot(x, y2, '.r-', label="$A_{val}$")
            axs[i].plot(x, y3, '.g-', label="$A_{test}$")
            axs[i].legend()
            axs[i].grid(True)

            axs[i].set_xscale("log")
            axs[i].set_yscale("linear")

            axs[i].set_xlabel("L2-regularization Strength")
            axs[i].set_ylabel("Accuracy")

        plt.savefig(os.path.join(target_dir, f'{extract}.png'))
        plt.close()


def plot_accuracy_vs_extract_freeze(stats, plots):
    target_dir = os.path.join(plots, 'extract_freeze_study')
    helpers.create_or_recreate_dir(target_dir)
    lambdas = list(set([stat['hyperparameters']['lambda'] for stat in stats]))
    print(lambdas)
    for i, l2 in enumerate(lambdas):
        stats_filtered = list(filter(lambda s: s['hyperparameters']['lambda'] == l2, stats))
        print('len', len(stats_filtered))

        x = np.array([stat['hyperparameters']['extract'] for stat in stats_filtered])
        y = np.array([stat['hyperparameters']['freeze'] for stat in stats_filtered])
        z = np.array([stat['test_acc'] for stat in stats_filtered])
        x, y, z = zip(*sorted(zip(x, y, z)))
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)

        X, Y = np.meshgrid(x, y)
        Z = np.outer(z.T, z)

        fig, ax = plt.subplots(1, 1)
        ax.set_title(f'Accuracy versus \nnumber of extracted and frozen layers when $\lambda = {l2}$')
        ax.set_xscale("linear")
        ax.set_yscale("linear")
        ax.set_xlabel('Layer up to which weights were extracted from')
        ax.set_ylabel('Layer up to which weights were frozen')

        surf = ax.contourf(X, Y, Z, levels=50)
        fig.colorbar(surf)

        plt.savefig(os.path.join(target_dir, f'{l2}.png'))
        plt.close()


def plot_accuracy_vs_freeze(stats, plots):
    target_dir = os.path.join(plots, 'freeze_study')
    helpers.create_or_recreate_dir(target_dir)
    lambdas = list(set([stat['hyperparameters']['lambda'] for stat in stats]))
    extracts = list(set([stat['hyperparameters']['extract'] for stat in stats]))
    for i, l2 in enumerate(lambdas):
        for j, extract in enumerate(extracts):
            stats_filtered = list(filter(lambda s: s['hyperparameters']['lambda'] == l2 and s['hyperparameters']['extract'] == extract, stats))

            x = np.array([stat['hyperparameters']['freeze'] for stat in stats_filtered])
            y = np.array([stat['test_acc'] for stat in stats_filtered])
            x, y = zip(*sorted(zip(x, y)))
            x = np.array(x)
            y = np.array(y)

            fig, ax = plt.subplots(1, 1)
            ax.set_title(f'Accuracy versus frozen layers when $e = {extract}$ and $\lambda = {l2}$')
            ax.plot(x, y)

            ax.set_xscale("linear")
            ax.set_yscale("linear")
            ax.set_xlabel('Layer up to which weights were frozen')
            ax.set_ylabel('Accuracy on the test set')

            plt.savefig(os.path.join(target_dir, f'extract{extract}_lambda{l2}.png'))
            plt.close()


def main(experiments, plots):
    helpers.create_or_recreate_dir(plots)

    stats = os.path.join(experiments, 'stats.json')
    with open(stats) as f:
        stats = json.loads(f.read())
    stats = truncate_floats(stats)

    # Tabulate top-20 results sorted by performance
    stats = sorted(stats, key=lambda x: x['test_acc'], reverse=True)
    tabulate(stats[:20], os.path.join(plots, 'top20.tex'))

    # Study effect of lambda
    plot_accuracy_vs_lambda(stats, plots)

    # Study effect of extract and freeze
    plot_accuracy_vs_extract_freeze(stats, plots)

    # Study effect of freeze
    plot_accuracy_vs_freeze(stats, plots)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments', type=helpers.is_dir, required=True)
    parser.add_argument('--plots', type=helpers.is_dir, required=True)
    args = parser.parse_args()

    main(args.experiments, args.plots)
