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

            axs[i].set_title(f'$e = {extract}$ and $f = {freeze}$')
            axs[i].plot(x, y1, '.b-', label="$A_{train}$")
            axs[i].plot(x, y2, '.r-', label="$A_{val}$")
            axs[i].plot(x, y3, '.g-', label="$A_{test}$")
            axs[i].legend()
            axs[i].grid(True)

            axs[i].set_xscale("log")
            axs[i].set_yscale("linear")

            axs[i].set_xlabel("$\lambda$")
            axs[i].set_ylabel("Accuracy")

        plt.savefig(os.path.join(target_dir, f'{extract}.png'))
        plt.close()


def plot_accuracy_vs_extract_freeze(stats, plots):
    target_dir = os.path.join(plots, 'extract_freeze_study')
    helpers.create_or_recreate_dir(target_dir)
    lambdas = sorted(list(set([stat['hyperparameters']['lambda'] for stat in stats])))

    rows = 2
    cols = int(len(lambdas)/rows)
    fig, axs = plt.subplots(rows, cols, constrained_layout=True, figsize=(10, 5))

    for i in range(rows):
        for j in range(cols):
            idx = i*cols+j
            l2 = lambdas[idx]
            stats_filtered = list(filter(lambda s: s['hyperparameters']['lambda'] == l2, stats))

            x = np.array([stat['hyperparameters']['extract'] for stat in stats_filtered])
            y = np.array([stat['hyperparameters']['freeze'] for stat in stats_filtered])
            z = np.array([stat['test_acc'] for stat in stats_filtered])
            x, y, z = zip(*sorted(zip(x, y, z)))
            x = np.array(x)
            y = np.array(y)
            z = np.array(z)

            ax = axs[i,j]
            ax.set_title(f'$\lambda = {l2}$')
            ax.set_xscale("linear")
            ax.set_yscale("linear")
            ax.set_xticks([3,6,10,14,18])
            ax.set_yticks([0,3,6,10,14,18])
            ax.set_xlabel('$e$')
            ax.set_ylabel('$f$')
            surf = ax.tricontourf(x, y, z)
            #fig.colorbar(surf)

    plt.savefig(os.path.join(target_dir, f'extract_freeze_study.png'))
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
    stats_top = sorted(stats, key=lambda x: x['test_acc'], reverse=True)[:20]
    tabulate(stats_top, os.path.join(plots, 'top20.tex'))

    # Tabulate all
    stats_all = sorted(stats, key=lambda x: (x['hyperparameters']['extract'], x['hyperparameters']['freeze'],x['hyperparameters']['lambda']), reverse=True)
    tabulate(stats_all, os.path.join(plots, 'all.tex'))

    stats1 = list(filter(lambda x: x['hyperparameters']['extract'] == x['hyperparameters']['freeze'] == 18, stats_all))
    tabulate(stats1, os.path.join(plots, 'total_feature_extraction_no_fine_tuning.tex'))

    stats2 = list(filter(lambda x: x['hyperparameters']['extract'] in [14,10,6,3] and x['hyperparameters']['freeze'] == x['hyperparameters']['extract'], stats_all))
    tabulate(stats2, os.path.join(plots, 'partial_feature_extraction_no_fine_tuning.tex'))

    stats3 = list(filter(lambda x: x['hyperparameters']['extract'] == 18 and x['hyperparameters']['freeze'] == 0, stats_all))
    tabulate(stats3, os.path.join(plots, 'total_feature_extraction_total_fine_tuning.tex'))

    stats4 = list(filter(lambda x: x['hyperparameters']['extract'] == 18 and x['hyperparameters']['freeze'] in [14,10,6,3], stats_all))
    tabulate(stats4, os.path.join(plots, 'total_feature_extraction_partial_fine_tuning.tex'))

    stats5 = list(filter(lambda x: x['hyperparameters']['extract'] in [14,10,6,3] and x['hyperparameters']['freeze'] == 0, stats_all))
    tabulate(stats5, os.path.join(plots, 'partial_feature_extraction_total_fine_tuning.tex'))

    stats6 = list(filter(lambda x: x['hyperparameters']['extract'] in [14,10,6,3] and x['hyperparameters']['freeze'] < x['hyperparameters']['extract'], stats_all))
    tabulate(stats6, os.path.join(plots, 'partial_feature_extraction_partial_fine_tuning.tex'))




    # Study effect of lambda
    plot_accuracy_vs_lambda(stats, plots)

    # Study effect of extract and freeze
    plot_accuracy_vs_extract_freeze(stats, plots)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments', type=helpers.is_dir, required=True)
    parser.add_argument('--plots', type=helpers.is_dir, required=True)
    args = parser.parse_args()

    main(args.experiments, args.plots)
