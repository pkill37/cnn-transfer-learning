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
OPTIMIZER = lambda lr: tf.keras.optimizers.SGD(lr=lr, momentum=0.9, decay=10e-6, nesterov=True)

IMG_WIDTH = 224
IMG_HEIGHT = 224
IMG_CHANNELS = 3


def cnn(lr=0.001, l2=0.001):
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
    m.compile(loss=LOSS, optimizer=OPTIMIZER(lr), metrics=METRICS)
    return m


def train(experiments_path, train, epochs, bs):
    #callbacks = [
    #    #tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=0, mode='min', baseline=None),
    #    tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(experiments_path, 'model.hdf5'), monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=1),
    #    tf.keras.callbacks.TensorBoard(log_dir=experiments_path, histogram_freq=0, write_graph=True, write_images=True),
    #    tf.keras.callbacks.CSVLogger(filename=os.path.join(experiments_path, 'log.csv'), separator=',', append=False),
    #]

    x_train, y_train = data.load(train)
    x_train = x_train[:100]
    y_train = y_train[:100]
    x_train = data.standardize(x_train, np.mean(x_train, axis=(0,1,2)), np.std(x_train, axis=(0,1,2)))

    params = dict(lr=np.logspace(-5, 1, 5), l2=np.logspace(-5, 5, 5))

    for ofold, (otrain, otest) in enumerate(sklearn.model_selection.StratifiedKFold(n_splits=3, shuffle=True)):
        x_trainval_fold, y_trainval_fold = x_train[otrain], y_train[otest]
        x_test_fold, y_test_fold = x_train[otest], y_train[otest]

        for (lr, l2) in itertools.product(*params.values()):

            for ifold, (itrain, itest) in enumerate(sklearn.model_selection.StratifiedKFold(n_splits=3, shuffle=True)):
                x_train_fold, y_train_fold = x_trainval_fold[itrain], y_trainval_fold[itest]
                x_val_fold, y_val_fold = x_train[itest], y_train_fold[itest]

                model.fit(
                    x=x_train_fold,
                    y=y_train_fold,
                    validation_data=(x_val_fold, y_val_fold),
                    batch_size=bs,
                    epochs=epochs,
                    verbose=1,
                    callbacks=callbacks,
                    shuffle=True,
                )

    index, value = max(enumerate(mean_scores), key=operator.itemgetter(1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments', type=helpers.is_dir, required=True)
    parser.add_argument('--train', type=helpers.is_file, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--bs', type=int, required=True)
    args = parser.parse_args()

    train(args.experiments, args.train, args.epochs, args.bs)
