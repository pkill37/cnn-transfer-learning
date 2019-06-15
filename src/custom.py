import os
import argparse
import itertools

import tensorflow as tf
import numpy as np
import sklearn

import data
import helpers


EPSILON = 10e-3
IMG_WIDTH = 224
IMG_HEIGHT = 224
IMG_CHANNELS = 3

LOSS = 'binary_crossentropy'
METRICS = ['accuracy']
OPTIMIZER = tf.keras.optimizers.SGD(lr=10e-4, momentum=0.9, decay=0.0, nesterov=False)
LR_DECAY = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=10, verbose=1, mode='max', min_delta=EPSILON, cooldown=0, min_lr=0)


def custom(l2):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS), filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(units=512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2)),
        tf.keras.layers.Dense(units=1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(l2)),
    ])
    model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)
    return model


def train_custom(experiment, x, y, epochs, batch_size):
    # Standardize and split data
    x = data.standardize(x, np.mean(x, axis=(0,1,2)), np.std(x, axis=(0,1,2)))
    x_train, x_validation, y_train, y_validation = sklearn.model_selection.train_test_split(x, y, test_size=0.2, shuffle=True, stratify=y)
    del x
    del y

    # Build hyperparameters search grid
    params = dict(l2=np.logspace(-10, 2, 10))
    params = itertools.product(*params.values())

    # Train a model for each hyperparameters setting
    for param in params:
        l2 = param
        model = custom(l2)

        run = os.path.join(experiment, f'lambda{l2}')
        helpers.create_or_recreate_dir(run)
        print(run)
        model_filename = os.path.join(run, 'model.h5')
        csv_filename = os.path.join(run, 'train.csv')

        callbacks = [
            LR_DECAY,
            helpers.TrainingTimeLogger(),
            tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=EPSILON, patience=30, verbose=1, mode='min', baseline=None),
            tf.keras.callbacks.ModelCheckpoint(filepath=model_filename, monitor='loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=1),
            tf.keras.callbacks.CSVLogger(filename=csv_filename, separator=',', append=False),
        ]

        model.fit(x=x_train, y=y_train, validation_data=(x_validation, y_validation), batch_size=batch_size, epochs=epochs, verbose=1, callbacks=callbacks, shuffle=True)
        del model


def main(experiments, train_set, epochs, batch_size):
    # Set PRNG seeds so that all runs have the same initial conditions
    helpers.seed()

    # Train
    x, y = data.load(train_set)
    train_custom(os.path.join(experiments, 'custom'), x, y, epochs, batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments', type=helpers.is_dir, required=True)
    parser.add_argument('--train-set', type=helpers.is_file, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    args = parser.parse_args()

    main(args.experiments, args.train_set, args.epochs, args.batch_size)
