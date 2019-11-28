import os
import re
import argparse
import csv
import json
import itertools

import numpy as np
import matplotlib.pyplot as plt

import helpers


def truncate(f):
    return float("{:.2e}".format(f))


def truncate_floats(stats):
    for stat in stats:
        for h in stat['hyperparameters']:
            if isinstance(stat['hyperparameters'][h], float):
                stat['hyperparameters'][h] = truncate(stat['hyperparameters'][h])
            stat['hyperparameters']['extract'] = int(stat['hyperparameters']['extract'])
            stat['hyperparameters']['freeze'] = int(stat['hyperparameters']['freeze'])
        stat['train_acc'][-1] = truncate(stat['train_acc'][-1])
        stat['val_acc'][-1] = truncate(stat['val_acc'][-1])
        stat['test_acc'] = truncate(stat['test_acc'])
        stat['auc'] = truncate(stat['auc'])
        stat['classification_report']['weighted avg']['precision'] = truncate(stat['classification_report']['weighted avg']['precision'])
        stat['classification_report']['weighted avg']['recall'] = truncate(stat['classification_report']['weighted avg']['recall'])
        stat['classification_report']['weighted avg']['f1-score'] = truncate(stat['classification_report']['weighted avg']['f1-score'])
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
    latex += '|c|c|c|c|c|c|c|c|c|c|'
    latex += ' }\n'
    latex += '\\hline\n'
    latex += '$e$ & $f$ & $\lambda$ & $A_{train}$ & $A_{val}$ & $A_{test}$ & Precision & Recall & F1-Score \\\\\n'
    latex += '\\hline\n'

    # Build body
    for stat in stats:
        extract = stat['hyperparameters']['extract']
        freeze = stat['hyperparameters']['freeze']
        lambd = stat['hyperparameters']['lambda']
        train_acc = stat['train_acc'][-1]
        val_acc = stat['val_acc'][-1]
        test_acc = stat['test_acc']
        precision = stat['classification_report']['weighted avg']['precision']
        recall = stat['classification_report']['weighted avg']['recall']
        f1 = stat['classification_report']['weighted avg']['f1-score']

        latex += f"{extract} & {freeze} & {lambd} & {train_acc} & {val_acc} & {test_acc} & {precision} & {recall} & {f1} \\\\\n"

    def _mean_std(stats):
        m = truncate(np.mean(stats))
        s = truncate(np.std(stats))
        return m, s

    mean_train_acc, std_train_acc = _mean_std([stat['train_acc'][-1] for stat in stats])
    mean_val_acc, std_val_acc     = _mean_std([stat['val_acc'][-1] for stat in stats])
    mean_test_acc, std_test_acc   = _mean_std([stat['test_acc'] for stat in stats])
    mean_precision, std_precision = _mean_std([stat['classification_report']['weighted avg']['precision'] for stat in stats])
    mean_recall, std_recall       = _mean_std([stat['classification_report']['weighted avg']['recall'] for stat in stats])
    mean_f1, std_f1               = _mean_std([stat['classification_report']['weighted avg']['f1-score'] for stat in stats])

    latex += '\\hline\n'
    latex += f" & & & ${mean_train_acc}\pm{std_train_acc}$ & ${mean_val_acc}\pm{std_val_acc}$ & ${mean_test_acc}\pm{std_test_acc}$ & ${mean_precision}\pm{std_precision}$ & ${mean_recall}\pm{std_recall}$ & ${mean_f1}\pm{std_f1}$ \\\\\n"
    latex += '\\hline\n'
    latex += '\\end{tabular}\n'
    latex += '\\caption{Foobar}\n'
    latex += '\\label{table:foobar}\n'
    latex += '\\end{table}\n'

    with open(target_file, 'w') as f:
        f.write(latex)


def plot_accuracy_vs_lambda(stats, target_file):
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

    plt.savefig(target_file)
    plt.close()


def plot_training(stat, target_file):
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

    plt.savefig(target_file)
    plt.close()


def plot_confusion_matrix(stat, target_file):
    cm = np.array(stat['confusion_matrix'])

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()

    classes = ['Non-melanoma', 'Melanoma']
    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(target_file)
    plt.close()


def plot_accuracy_vs_freeze(stats, target_file):
    lambdas = sorted(list(set([stat['hyperparameters']['lambda'] for stat in stats])))

    rows = 2
    cols = int(len(lambdas)/rows)
    fig, axs = plt.subplots(rows, cols, constrained_layout=True, figsize=(10, 5))

    for i in range(rows):
        for j in range(cols):
            idx = i*cols+j
            l2 = lambdas[idx]
            stats_filtered = list(filter(lambda s: s['hyperparameters']['lambda'] == l2, stats))

            x = np.array([stat['hyperparameters']['freeze'] for stat in stats_filtered])
            y2 = np.array([stat['test_acc'] for stat in stats_filtered])
            x, y2 = zip(*sorted(zip(x, y2)))
            x = np.array(x)
            y2 = np.array(y2)

            ax = axs[i,j]
            ax.set_title(f'$\lambda = {l2}$')
            ax.set_xscale("linear")
            ax.set_yscale("linear")
            ax.set_xlabel('$f$')
            ax.set_ylabel('$Accuracy$')
            ax.set_xticks([0,3,6,10,14,18])
            ax.plot(x, y2, '.g-', label="$A_{test}$")

    plt.savefig(target_file)
    plt.close()


def main(experiments):
    # Read stats
    stats = []
    for d in os.listdir(experiments):
        d = os.path.join(experiments, d)
        if os.path.isdir(d):
            with open(os.path.join(d, 'stats.json')) as f: stats.append(json.loads(f.read()))
    stats = truncate_floats(stats)

    # Individual plots
    for stat in stats:
        plot_training(stat, os.path.join(experiments, stat['id'], 'training.png'))
        plot_confusion_matrix(stat, os.path.join(experiments, stat['id'], 'confusion_matrix.png'))

    # All results grouped and sorted by (extract,freeze,lambda)
    #stats_all = sorted(stats, key=lambda x: (x['hyperparameters']['extract'], x['hyperparameters']['freeze'], x['hyperparameters']['lambda']), reverse=True)
    stats_all = sorted(stats, key=lambda x: x['hyperparameters']['mfraction'], reverse=True)
    tabulate(stats_all, os.path.join(experiments, 'table_all.tex'))

    # Total feature extraction with no fine tuning
    stats_total = list(filter(lambda x: x['hyperparameters']['extract'] == x['hyperparameters']['freeze'] == 18, stats))
    tabulate(stats_total, os.path.join(experiments, 'table_total.tex'))
    plot_accuracy_vs_lambda(stats_total, os.path.join(experiments, 'lambda_total.png'))

    # Total feature extraction with fine tuning overall study
    stats_finetuning = list(filter(lambda x: x['hyperparameters']['extract'] == 18 and x['hyperparameters']['freeze'] < 18, stats))
    plot_accuracy_vs_freeze(stats_finetuning, os.path.join(experiments, 'freeze_finetuning.png'))
    # Total feature extraction with fine tuning 14
    stats_finetuning14 = list(filter(lambda x: x['hyperparameters']['extract'] == 18 and x['hyperparameters']['freeze'] == 14, stats))
    tabulate(stats_finetuning14, os.path.join(experiments, 'table_finetuning_14.tex'))
    plot_accuracy_vs_lambda(stats_finetuning14, os.path.join(experiments, 'lambda_finetuning_14.png'))
    # Total feature extraction with fine tuning 10
    stats_finetuning10 = list(filter(lambda x: x['hyperparameters']['extract'] == 18 and x['hyperparameters']['freeze'] == 10, stats))
    tabulate(stats_finetuning10, os.path.join(experiments, 'table_finetuning_10.tex'))
    plot_accuracy_vs_lambda(stats_finetuning10, os.path.join(experiments, 'lambda_finetuning_10.png'))
    # Total feature extraction with fine tuning 6
    stats_finetuning6 = list(filter(lambda x: x['hyperparameters']['extract'] == 18 and x['hyperparameters']['freeze'] == 6, stats))
    tabulate(stats_finetuning6, os.path.join(experiments, 'table_finetuning_6.tex'))
    plot_accuracy_vs_lambda(stats_finetuning6, os.path.join(experiments, 'lambda_finetuning_6.png'))
    # Total feature extraction with fine tuning 3
    stats_finetuning3 = list(filter(lambda x: x['hyperparameters']['extract'] == 18 and x['hyperparameters']['freeze'] == 3, stats))
    tabulate(stats_finetuning3, os.path.join(experiments, 'table_finetuning_3.tex'))
    plot_accuracy_vs_lambda(stats_finetuning3, os.path.join(experiments, 'lambda_finetuning_3.png'))
    # Total feature extraction with fine tuning 0
    stats_finetuning0 = list(filter(lambda x: x['hyperparameters']['extract'] == 18 and x['hyperparameters']['freeze'] == 0, stats))
    tabulate(stats_finetuning0, os.path.join(experiments, 'table_finetuning_0.tex'))
    plot_accuracy_vs_lambda(stats_finetuning0, os.path.join(experiments, 'lambda_finetuning_0.png'))

    # Partial feature extraction
    stats_partial = list(filter(lambda x: x['hyperparameters']['extract'] < 18, stats))
    tabulate(stats_partial, os.path.join(experiments, 'table_partial.tex'))
    plot_accuracy_vs_lambda(stats_partial, os.path.join(experiments, 'lambda_partial.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments', type=helpers.is_dir, required=True)
    args = parser.parse_args()
    main(args.experiments)
