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


def vgg19(extract_until=21, freeze_until=21, lr=0.001, l2=0.001):
    tf.keras.backend.clear_session()

    assert extract_until >= freeze_until

    input_tensor = tf.keras.layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
    vgg19 = tf.keras.applications.vgg19.VGG19(weights='imagenet', include_top=False, input_tensor=input_tensor)

    # Freeze
    for layer in vgg19.layers[:freeze_until]:
        layer.trainable = False

    # Extract
    x = vgg19.layers[extract_until].output

    # Classifier
    x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = tf.keras.layers.Dense(units=4096, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.Dense(units=4096, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.Dense(units=1, activation='sigmoid')(x)

    model = tf.keras.models.Model(inputs=input_tensor, outputs=x, name='vgg19')
    model.compile(loss=LOSS, optimizer=OPTIMIZER(lr), metrics=METRICS)
    return model


def train(experiments_path, train, validation, pretrained_model, extract_until, freeze_until, epochs, batch_size):
    #callbacks = [
    #    #tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=0, mode='min', baseline=None),
    #    tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(experiments_path, 'model.hdf5'), monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=1),
    #    tf.keras.callbacks.TensorBoard(log_dir=experiments_path, histogram_freq=0, write_graph=True, write_images=True),
    #    tf.keras.callbacks.CSVLogger(filename=os.path.join(experiments_path, 'log.csv'), separator=',', append=False),
    #]

    x_train, y_train = data.load(train)
    x_train = tf.keras.applications.vgg19.preprocess_input(x_train)

    estimator = tf.keras.wrappers.scikit_learn.KerasClassifier(
        build_fn=vgg19,
        batch_size=bs,
        epochs=epochs,
        verbose=1,
        #callbacks=callbacks,
        shuffle=True,
    )

    skf = sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle=True)
    results = sklearn.model_selection.cross_val_score(estimator, x_train, y_train, scoring='accuracy', cv=skf, n_jobs=1, error_score='raise')
    print("F1-score averaged over 3 folds: %.2f%% +- %.2f%%" % (results.mean(), results.std()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments', type=helpers.is_dir, required=True)
    parser.add_argument('--train', type=helpers.is_file, required=True)
    parser.add_argument('--pretrained-model', choices=['vgg19', 'inceptionv3'], required=True)
    parser.add_argument('--extract-until', type=int, required=True)
    parser.add_argument('--freeze-until', type=int, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--bs', type=int, required=True)
    args = parser.parse_args()

    train(args.experiments_path, args.train, args.validation, args.pretrained_model, args.extract_until, args.freeze_until, args.epochs, args.bs)
