import os
import argparse
import itertools

import tensorflow as tf
import numpy as np
import sklearn

import data
import helpers


EPSILON = 10e-3
IMG_WIDTH = 299
IMG_HEIGHT = 299
IMG_CHANNELS = 3

LOSS = 'binary_crossentropy'
METRICS = ['accuracy']
OPTIMIZER = tf.keras.optimizers.SGD(lr=10e-4, momentum=0.9, decay=0.0, nesterov=False)
OPTIMIZER_WARMUP = tf.keras.optimizers.Adam()
LR_DECAY = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=10, verbose=1, mode='max', min_delta=EPSILON, cooldown=0, min_lr=0)


def inceptionv3(l2):
    # Freeze pre-trained model layers
    inceptionv3 = tf.keras.applications.inception_v3.InceptionV3(weights='imagenet', input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS), include_top=False)
    for layer in inceptionv3.layers:
        layer.trainable = False

    # Classifier
    y = inceptionv3.output
    y = tf.keras.layers.GlobalAveragePooling2D()(y)
    y = tf.keras.layers.Dense(units=1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2))(y)
    y = tf.keras.layers.Dense(units=1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(l2))(y)

    model = tf.keras.models.Model(inputs=inceptionv3.input, outputs=y)
    return model


def train_one(experiments, x_train, y_train, x_validation, y_validation, epochs, batch_size, l2):
    run = os.path.join(experiments, f'lambda{l2}')
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

    # Warmup for a few epochs so that the classifier layers are reasonably trained using Adam
    model = inceptionv3(l2)
    model.compile(loss=LOSS, optimizer=OPTIMIZER_WARMUP, metrics=METRICS)
    model.fit(x=x_train, y=y_train, validation_data=(x_validation, y_validation), batch_size=batch_size, epochs=5, verbose=1, callbacks=callbacks, shuffle=True)
    
    # Finetune top 2 inception blocks cautiously using SGD
    for layer in model.layers[:249]: layer.trainable = False
    for layer in model.layers[249:]: layer.trainable = True
    model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)
    model.fit(x=x_train, y=y_train, validation_data=(x_validation, y_validation), batch_size=batch_size, epochs=epochs, verbose=1, callbacks=callbacks, shuffle=True)
    
    # Cleanup
    del model
    helpers.fix_layer0(model_filename, (None, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS), 'float32')


def main(experiments, train_set, epochs, batch_size, l2):
    # Set PRNG seeds so that all runs have the same initial conditions
    helpers.seed()

    # Standardize and split data
    x, y = data.load(train_set)
    x = tf.keras.applications.inception_v3.preprocess_input(x)
    x_train, x_validation, y_train, y_validation = sklearn.model_selection.train_test_split(x, y, test_size=0.15, shuffle=True, stratify=y)
    del x
    del y

    # Train
    train_one(experiments, x_train, y_train, x_validation, y_validation, epochs, batch_size, l2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments', type=helpers.is_dir, required=True)
    parser.add_argument('--train-set', type=helpers.is_file, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--l2', type=float, required=True)
    args = parser.parse_args()

    main(args.experiments, args.train_set, args.epochs, args.batch_size, args.l2)
