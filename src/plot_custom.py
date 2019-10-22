import os
import re
import argparse
import csv
import json

import numpy as np
import matplotlib.pyplot as plt

import helpers


def truncate_floats(stats):
    def truncate(f): return float("{:.2e}".format(f))

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


def plot_accuracy_vs_units_lambda(stats, plots):
    x = np.array([stat['hyperparameters']['lambda'] for stat in stats])
    y = np.array([stat['hyperparameters']['units'] for stat in stats])
    z = np.array([stat['test_acc'] for stat in stats])
    x, y, z = zip(*sorted(zip(x, y, z)))
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    X, Y = np.meshgrid(x, y)
    Z = np.outer(z.T, z)

    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    ax.set_xscale("log")
    ax.set_yscale("linear")
    ax.set_xlabel("L2-regularization strength")
    ax.set_ylabel("Number of neurons")

    surf = ax.contourf(X, Y, Z, levels=50)
    fig.colorbar(surf)

    target_dir = os.path.join(plots, 'lambda_units_study')
    helpers.create_or_recreate_dir(target_dir)
    plt.savefig(os.path.join(target_dir, f'lambda_units_study.png'))
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
    plot_accuracy_vs_units_lambda(stats, plots)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments', type=helpers.is_dir, required=True)
    parser.add_argument('--plots', type=helpers.is_dir, required=True)
    args = parser.parse_args()

    main(args.experiments, args.plots)
