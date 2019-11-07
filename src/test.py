import os
import re
import argparse
import csv
import json

import tensorflow as tf
import numpy as np
import sklearn.metrics

import helpers
import data


def test(model, test):
    m = tf.keras.models.load_model(model)
    x_test, y_test = data.load(test)
    x_test = tf.keras.applications.vgg16.preprocess_input(x_test)
    #x_test = data.standardize(x_test, np.mean(x_test, axis=(0,1,2)), np.std(x_test, axis=(0,1,2)))

    y_pred = m.predict(
        x=x_test,
        batch_size=32,
        verbose=1,
    )

    # Keep raw probability scores
    y_pred.reshape(-1)
    y_scores = y_pred

    # Threshold default predictions at 0.5
    threshold = 0.5
    y_pred[y_pred <= threshold] = 0. # not melanoma
    y_pred[y_pred > threshold] = 1. # melanoma

    np.savez_compressed(os.path.join(os.path.dirname(model), 'predictions'), y_true=y_test, y_pred=y_pred, y_scores=y_scores)


def reduce_experiment(experiment):
    def read_csv(train_csv):
        csv_stats = {
            'acc': [],
            'val_acc': [],
            'loss': [],
            'val_loss': [],
            'time': [],
        }

        with open(train_csv, 'r', newline='') as f:
            for i, row in enumerate(csv.DictReader(f)):
                csv_stats['acc'].append(float(row['acc']))
                csv_stats['val_acc'].append(float(row['val_acc']))
                csv_stats['loss'].append(float(row['loss']))
                csv_stats['val_loss'].append(float(row['val_loss']))
                csv_stats['time'].append(float(row['time']))

        return csv_stats


    def read_predictions(predictions):
        data = np.load(predictions)
        y_true = data['y_true']
        y_pred = data['y_pred']
        y_scores = data['y_scores']
        return y_true, y_pred, y_scores


    def hyperparameters_from_dirname(s):
        groups = re.findall(r'([a-zA-Z]+)([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)', s)
        return { str(group[0]): float(group[1]) for group in groups }

    y_true, y_pred, y_scores = read_predictions(os.path.join(experiment, 'predictions.npz'))
    csv_stats = read_csv(os.path.join(experiment, 'train.csv'))

    return {
        'id': os.path.basename(experiment),
        'hyperparameters': hyperparameters_from_dirname(os.path.basename(experiment)),
        'train_loss': csv_stats['loss'],
        'val_loss': csv_stats['val_loss'],
        'train_acc': csv_stats['acc'],
        'val_acc': csv_stats['val_acc'],
        'test_acc': sklearn.metrics.accuracy_score(y_true, y_pred),
        'auc': sklearn.metrics.roc_auc_score(y_true, y_scores),
        'classification_report': sklearn.metrics.classification_report(y_true, y_pred, target_names=['0', '1'], output_dict=True),
    }


def main(experiment, test_set):
    test(os.path.join(experiment, 'model.h5'), test_set)
    stats = reduce_experiment(experiment)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>', stats['id'], stats['test_acc'], stats['auc'])

    with open(os.path.join(experiment, 'stats.json'), 'w') as f:
        json.dump(stats, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=helpers.is_dir, required=True)
    parser.add_argument('--test-set', type=helpers.is_file, required=True)
    args = parser.parse_args()

    main(args.experiment, args.test_set)
