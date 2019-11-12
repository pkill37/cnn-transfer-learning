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

# TODO: generalize into any dict with float values or keys
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

    # Build header
    latex = '\\begin{table}[ht]\n'
    latex += '\\centering\n'
    latex += '\\begin{tabular}{ '
    latex += '|c|c|c|c|c|c|c|c|c|c|c|c|'
    latex += ' }\n'
    latex += '\\hline\n'
    latex += '$e$ & $f$ & $u$ & $\lambda$ & $d$ & $p$ & $A_{train}$ & $A_{val}$ & $A_{test}$ & Precision & Recall & F1-Score \\\\\n'
    latex += '\\hline\n'

    mean = truncate(np.mean(np.array([stat['test_acc'] for stat in stats])))
    std = truncate(np.std(np.array([stat['test_acc'] for stat in stats])))

    # Build body
    for stat in stats:
        latex += f"{stat['hyperparameters']['extract']} & {stat['hyperparameters']['freeze']} & {stat['hyperparameters']['units']} & {stat['hyperparameters']['lambda']} & {stat['hyperparameters']['dropout']} & {stat['hyperparameters']['patience']} & {stat['train_acc'][-1]} & {stat['val_acc'][-1]} & {stat['test_acc']} & {stat['classification_report']['weighted avg']['precision']} & {stat['classification_report']['weighted avg']['recall']} & {stat['classification_report']['weighted avg']['f1-score']} \\\\\n"
    latex += '\\hline\n'
    latex += '\\end{tabular}\n'
    latex += '\\caption{'
    latex += f"Average performance ${mean} \pm {std}$"
    latex += '}\n'
    latex += '\\label{table:foobar}\n'
    latex += '\\end{table}\n'

    with open(target_file, 'w') as f:
        f.write(latex)


def plot_accuracy_vs_lambda(stats, experiments):
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

    plt.savefig(os.path.join(experiments, f'lambda_study.png'))
    plt.close()


def plot_accuracy_vs_units(stats, experiments):
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

    plt.savefig(os.path.join(experiments, f'units_study.png'))
    plt.close()


def plot_accuracy_vs_dropout(stats, experiments):
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

    plt.savefig(os.path.join(experiments, f'dropout_study.png'))
    plt.close()


def plot_training(stats, experiments):
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

        plt.savefig(os.path.join(experiments, stat['id'], 'training_study.png'))
        plt.close()


def plot_accuracy_vs_dropout(stats, experiments):
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

    plt.savefig(os.path.join(experiments, f'dropout_study.png'))
    plt.close()


def plot_confusion_matrix(stats, experiments):
    for stat in stats:
        print(stat['id'], stat['confusion_matrix'])
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
        plt.savefig(os.path.join(experiments, stat['id'], 'confusion_matrix.png'))
        plt.close()


def main(experiments):
    # Read stats
    stats = []
    for d in os.listdir(experiments):
        d = os.path.join(experiments, d)
        if os.path.isdir(d):
            with open(os.path.join(d, 'stats.json')) as f: stats.append(json.loads(f.read()))
    stats = truncate_floats(stats)

    # Tabulate results sorted by performance on the test set
    tabulate(
        sorted(stats, key=lambda x: x['test_acc'], reverse=True),
        os.path.join(experiments, 'table.tex')
    )

    # Study test set performance
    try: plot_confusion_matrix(stats, experiments)
    except: pass

    # Study progress over epochs
    try: plot_training(stats, experiments)
    except: pass

    # Study effect of lambda
    try: plot_accuracy_vs_lambda(stats, experiments)
    except: pass

    # Study effect of units
    try: plot_accuracy_vs_units(stats, experiments)
    except: pass

    # Study effect of dropout
    try: plot_accuracy_vs_dropout(stats, experiments)
    except: pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments', type=helpers.is_dir, required=True)
    args = parser.parse_args()
    main(args.experiments)
