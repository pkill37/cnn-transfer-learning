import os
import argparse

import tensorflow as tf
import numpy as np
import sklearn.model_selection

import metrics
import models
import data
import helpers


LOSS = 'binary_crossentropy'
METRICS = metrics.METRICS
OPTIMIZER = lambda lr: tf.keras.optimizers.SGD(lr=lr, momentum=0.9, decay=10e-6, nesterov=True)

IMG_WIDTH = 224
IMG_HEIGHT = 224
IMG_CHANNELS = 3


def cnn(lr=0.001, l1=0.001, l2=0.001):
    tf.keras.backend.clear_session()

    m = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS), filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=512, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1, l2=l2)),
        tf.keras.layers.Dense(units=1, activation='sigmoid', name='lol'),
    ])
    m.compile(loss=LOSS, optimizer=OPTIMIZER(lr), metrics=METRICS)
    return m


def train(experiments_path, train, validation, epochs, bs):
    #callbacks = [
    #    #tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=0, mode='min', baseline=None),
    #    tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(experiments_path, 'model.hdf5'), monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=1),
    #    tf.keras.callbacks.TensorBoard(log_dir=experiments_path, histogram_freq=0, write_graph=True, write_images=True),
    #    tf.keras.callbacks.CSVLogger(filename=os.path.join(experiments_path, 'log.csv'), separator=',', append=False),
    #]

    x_train, y_train = data.load(train)
    x_train = data.standardize(x_train, np.mean(x_train, axis=(0,1,2)), np.std(x_train, axis=(0,1,2)))

    estimator = tf.keras.wrappers.scikit_learn.KerasClassifier(
        build_fn=cnn,
        batch_size=bs,
        epochs=epochs,
        verbose=1,
        #callbacks=callbacks,
        shuffle=True,
    )

    param_grid = dict(lr=np.logspace(-5, 1, 5), l1=np.logspace(-5, 5, 11), l2=np.logspace(-5, 5, 11))
    gs = sklearn.model_selection.GridSearchCV(estimator=estimator, param_grid=param_grid, scoring='accuracy', n_jobs=1)
    skf = sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle=True)
    results = sklearn.model_selection.cross_val_score(gs, x_train, y_train, scoring='accuracy', cv=skf, n_jobs=1, error_score='raise')
    print("F1-score averaged over 3 folds: %.2f%% +- %.2f%%" % (results.mean(), results.std()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments', type=helpers.is_dir, required=True)
    parser.add_argument('--train', type=helpers.is_file, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--bs', type=int, required=True)
    args = parser.parse_args()

    train(args.experiments, args.train, args.validation, args.epochs, args.bs)
