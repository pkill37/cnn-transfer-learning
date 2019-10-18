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


def plot_lambda(stats, plots):
    extract = [18, 14, 10, 6, 3]
    freeze = [18, 14, 10, 6, 3, 0]
    cases = []
    for e in extract:
        case = []
        for f in freeze:
            if f <= e:
                case.append((e, f))
        cases.append(case)

    for case in cases:
        fig, axs = plt.subplots(1, len(case), constrained_layout=True, figsize=(3*len(case), 3))
        for i in range(len(case)):
            extract, freeze = case[i]
            stats_filtered = list(filter(lambda s: s['hyperparameters']['extract'] == extract and s['hyperparameters']['freeze'] == freeze, stats))
            x = np.array(sorted([stat['hyperparameters']['lambda'] for stat in stats_filtered]))
            y1 = np.array(sorted([stat['train_acc'][-1] for stat in stats_filtered]))
            y2 = np.array(sorted([stat['val_acc'][-1] for stat in stats_filtered]))
            y3 = np.array(sorted([stat['test_acc'] for stat in stats_filtered]))

            axs[i].set_title(f'Extract {extract} and freeze {freeze} layers')
            axs[i].plot(x, y1, '.b-', label="$A_{train}$")
            axs[i].plot(x, y2, '.r-', label="$A_{val}$")
            axs[i].plot(x, y3, '.g-', label="$A_{test}$")
            axs[i].legend()
            axs[i].grid(True)

            axs[i].set_xscale("log")
            axs[i].set_yscale("linear")

            axs[i].set_xlabel("L2-regularization Strength")
            axs[i].set_ylabel("Accuracy")

        plt.savefig(os.path.join(plots, f'{extract}.png'))
        plt.close()


def main(experiments, plots):
    stats = os.path.join(experiments, 'stats.json')
    with open(stats) as f:
        stats = json.loads(f.read())

    # Tabulate top-20 results sorted by performance
    stats = sorted(stats, key=lambda x: x['test_acc'], reverse=True)
    stats = stats[:20]
    tabulate(stats, os.path.join(plots, 'top20.tex'))

    # Plot accuracy vs lambda
    plot_graphs(stats, plots)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments', type=helpers.is_dir, required=True)
    parser.add_argument('--plots', type=helpers.is_dir, required=True)
    args = parser.parse_args()

    main(args.experiments, args.plots)
