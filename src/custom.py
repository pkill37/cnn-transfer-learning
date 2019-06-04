import os
import argparse
import itertools

import tensorflow as tf
import numpy as np
import sklearn.model_selection

import metrics
import data
import helpers


LOSS = 'binary_crossentropy'
METRICS = metrics.METRICS
OPTIMIZER = lambda lr: tf.keras.optimizers.SGD(lr=10e-4, momentum=0.9, decay=10e-6, nesterov=True)
BATCH_SIZE = 64

IMG_WIDTH = 224
IMG_HEIGHT = 224
IMG_CHANNELS = 3


def custom(l2):
    tf.keras.backend.clear_session()

    m = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS), filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2)),
        tf.keras.layers.Dense(units=1, activation='sigmoid'),
    ])
    m.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)
    return m


def train(experiment, train, k, epochs):
    x, y = data.load(train)
    x = x[:10]
    y = y[:10]
    x = data.standardize(x, np.mean(x, axis=(0,1,2)), np.std(x, axis=(0,1,2)))

    outer_scores = []
    for outer, (outer_trainval, outer_test) in enumerate(sklearn.model_selection.StratifiedKFold(n_splits=k, shuffle=True)):
        x_trainval, y_trainval = x[outer_trainval], y[outer_trainval]
        x_test, y_test = x[outer_test], y[outer_test]

        inner_mean_scores = []

        params = dict(l2=np.logspace(-5, 5, 5))
        for (l2,) in itertools.product(*params.values()):

            inner_scores = []

            for inner, (inner_train, inner_test) in enumerate(sklearn.model_selection.StratifiedKFold(n_splits=k, shuffle=True)):
                x_train, y_train = x_trainval[inner_train], y_trainval[inner_train]
                x_val, y_val = x_train[inner_test], y_train[inner_test]


                # Train on (x_train, y_train)
                model = custom(l2)
                callbacks = [
                    tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=30, verbose=0, mode='min', baseline=None),
                    tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(experiment, 'model.hdf5'), monitor='loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=1),
                    tf.keras.callbacks.CSVLogger(filename=os.path.join(experiment, 'train.csv'), separator=',', append=False),
                ]
                model.fit(x=x_train, y=y_train, batch_size=BATCH_SIZE, epochs=epochs, verbose=1, callbacks=callbacks, shuffle=True)

                # Evaluate on (x_val, y_val)
                scores = model.evaluate(...)
                inner_scores.append(scores)

            # Compute inner test accuracy
            inner_mean_scores.append(np.mean(inner_scores))

        # Select parameter set with maximum accuracy
        index, value = max(enumerate(inner_mean_scores), key=operator.itemgetter(1))
        print('Best parameter : %i' % (params[index]))
        model = custom(l2)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=30, verbose=0, mode='min', baseline=None),
            tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(experiment, 'model.hdf5'), monitor='loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=1),
            tf.keras.callbacks.CSVLogger(filename=os.path.join(experiment, 'train.csv'), separator=',', append=False),
        ]
        model.fit(x=x_train, y=y_train, batch_size=BATCH_SIZE, epochs=epochs, verbose=1, callbacks=callbacks, shuffle=True)

    # show the prediction error estimate produced by nested CV
    print('Unbiased prediction error: %.4f' % (np.mean(outer_scores)))

    # Train final model on entire dataset without setting data aside for testing
    model = custom(l2)
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=30, verbose=0, mode='min', baseline=None)]
    model.fit(x=x, y=y, batch_size=BATCH_SIZE, epochs=epochs, verbose=1, callbacks=callbacks, shuffle=True)
    tf.keras.models.save_model(model, os.path.join(experiment, 'model.hdf5')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=helpers.is_dir, required=True)
    parser.add_argument('--train', type=helpers.is_file, required=True)
    parser.add_argument('--k', type=int, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    args = parser.parse_args()

    train(args.experiment, args.train, args.k, args.epochs)
