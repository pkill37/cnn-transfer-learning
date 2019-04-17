import os
import argparse

import tensorflow as tf

import data
import helpers
import metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=helpers.is_file, required=True)
    parser.add_argument('--test', type=helpers.is_file, required=True)
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model, custom_objects={
        'p': metrics.precision(),
        'r': metrics.recall(),
        'f1': metrics.f1_score(),
        'tp': metrics.true_positive(),
        'tn': metrics.true_negative(),
        'fp': metrics.false_positive(),
        'fn': metrics.false_negative(),
    })

    x_test, y_test = data.load(args.test)

    scores = model.evaluate(
        x=x_test,
        y=y_test,
        batch_size=32,
        verbose=1,
    )

    for score, metric in zip(scores, model.metrics_names):
        print("%s: %0.4f" % (metric, score))
