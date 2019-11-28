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


def custom2(l2, units, dropout):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS), filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(units=units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2)),
        tf.keras.layers.Dropout(rate=dropout),
        tf.keras.layers.Dense(units=1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(l2)),
    ])
    return model


def main(experiments, train_set, epochs, batch_size, l2, units, dropout, patience):
    # Set PRNG seeds so that all runs have the same initial conditions
    helpers.seed()

    # Standardize and split data
    x, y = data.load(train_set)
    x = data.standardize(x, np.mean(x, axis=(0,1,2)), np.std(x, axis=(0,1,2)))
    x_train, x_validation, y_train, y_validation = sklearn.model_selection.train_test_split(x, y, test_size=0.15, shuffle=True, stratify=y)
    del x
    del y

    # Settings
    run = os.path.join(experiments, f'lambda{l2}_units{units}_dropout{dropout}_patience{patience}')
    helpers.create_or_recreate_dir(run)
    model_filename = os.path.join(run, 'model.h5')
    csv_filename = os.path.join(run, 'train.csv')

    callbacks = [
        helpers.TrainingTimeLogger(),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=10, verbose=1, mode='max', min_delta=EPSILON, cooldown=0, min_lr=0),
        tf.keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=EPSILON, patience=patience, verbose=1, mode='max', baseline=None),
        tf.keras.callbacks.CSVLogger(filename=csv_filename, separator=',', append=False),
    ]

    # Train
    model = custom2(l2, units, dropout)
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.SGD(lr=10e-4, momentum=0.9, decay=0.0, nesterov=False), metrics=['accuracy'])
    model.fit(x=x_train, y=y_train, validation_data=(x_validation, y_validation), batch_size=batch_size, epochs=epochs, verbose=1, callbacks=callbacks, shuffle=True)
    model.save(model_filename)
    del model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments', type=helpers.is_dir, required=True)
    parser.add_argument('--train-set', type=helpers.is_file, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--l2', type=float, required=True)
    parser.add_argument('--units', type=int, required=True)
    parser.add_argument('--dropout', type=float, required=True)
    parser.add_argument('--patience', type=int, required=True)
    args = parser.parse_args()
    main(args.experiments, args.train_set, args.epochs, args.batch_size, args.l2, args.units, args.dropout, args.patience)
