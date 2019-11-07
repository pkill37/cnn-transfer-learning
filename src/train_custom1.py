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


def custom1(l2, units):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS), filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2)),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2)),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2)),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(units=units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2)),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Dense(units=units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2)),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Dense(units=1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(l2)),
    ])
    model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)
    return model


def train_one(experiments, x_train, y_train, x_validation, y_validation, epochs, batch_size, l2, units):
    run = os.path.join(experiments, f'units{units}_lambda{l2}')
    helpers.create_or_recreate_dir(run)
    print(run)
    model_filename = os.path.join(run, 'model.h5')
    csv_filename = os.path.join(run, 'train.csv')

    callbacks = [
        LR_DECAY,
        helpers.TrainingTimeLogger(),
        tf.keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=EPSILON, patience=30, verbose=1, mode='max', baseline=None),
        tf.keras.callbacks.CSVLogger(filename=csv_filename, separator=',', append=False),
    ]

    # Train
    model = custom1(l2, units)
    model.fit(x=x_train, y=y_train, validation_data=(x_validation, y_validation), batch_size=batch_size, epochs=epochs, verbose=1, callbacks=callbacks, shuffle=True)
    model.save(model_filename)
    del model


def main(experiments, train_set, epochs, batch_size, l2, units):
    # Set PRNG seeds so that all runs have the same initial conditions
    helpers.seed()

    # Standardize and split data
    x, y = data.load(train_set)
    x = data.standardize(x, np.mean(x, axis=(0,1,2)), np.std(x, axis=(0,1,2)))
    x_train, x_validation, y_train, y_validation = sklearn.model_selection.train_test_split(x, y, test_size=0.2, shuffle=True, stratify=y)
    del x
    del y

    # Train
    train_one(experiments, x_train, y_train, x_validation, y_validation, epochs, batch_size, l2, units)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments', type=helpers.is_dir, required=True)
    parser.add_argument('--train-set', type=helpers.is_file, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--l2', type=float, required=True)
    parser.add_argument('--units', type=int, required=True)
    args = parser.parse_args()

    main(args.experiments, args.train_set, args.epochs, args.batch_size, args.l2, args.units)
