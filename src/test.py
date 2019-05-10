import os
import argparse
import json

import tensorflow as tf
import numpy as np

import data
import helpers


def test(model, test):
    m = tf.keras.models.load_model(model)
    x_test, y_test = data.load(test)

    y_pred = m.predict(
        x=x_test,
        batch_size=32,
        verbose=1,
    )

    y_pred.reshape(-1)
    y_pred[y_pred <= 0.5] = 0.
    y_pred[y_pred > 0.5] = 1.

    np.savez_compressed(os.path.join(os.path.dirname(model), 'predictions'), y_true=y_test, y_pred=y_pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=helpers.is_file, required=True)
    parser.add_argument('--test', type=helpers.is_file, required=True)
    args = parser.parse_args()

    test(args.model, args.test)
