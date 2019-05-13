import os
import argparse

import tensorflow as tf
import numpy as np
import sklearn

import data
import helpers


# VGG's original training conditions per https://arxiv.org/pdf/1409.1556.pdf
LOSS = 'binary_crossentropy'
METRICS = ['accuracy']
OPTIMIZER = tf.keras.optimizers.SGD(lr=10e-2, momentum=0.9, decay=0.0, nesterov=False)
LR_DECAY = 0.1
L2 = 5*10e-4
DROPOUT = 0.5
BATCH_SIZE = 256
MIN_DELTA = 10e-3

IMG_WIDTH = 224
IMG_HEIGHT = 224
IMG_CHANNELS = 3


def vgg19(extract_until=21, freeze_until=21):
    assert extract_until >= freeze_until

    # Extract and freeze pre-trained model layers
    vgg19 = tf.keras.applications.vgg19.VGG19(weights='imagenet', input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS), include_top=False)
    model = tf.keras.models.Sequential()
    for i in range(0, extract_until+1): # i=0 is the input layer, i>0 are the actual model layers
        layer = vgg19.layers[i]
        layer.trainable = True if (i > freeze_until) and helpers.has_parameters(layer) else False
        layer.kernel_regularizer = tf.keras.regularizers.l2(L2) if helpers.has_parameters(layer) else None
        model.add(layer)

    # Classifier
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dense(units=512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2)))
    model.add(tf.keras.layers.Dropout(rate=DROPOUT))
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(L2)))
    model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)
    return model


def train(experiment, train, extract_until, freeze_until, epochs, bs):
    model = vgg19(extract_until, freeze_until)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=MIN_DELTA, patience=30, verbose=1, mode='min', baseline=None),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=LR_DECAY, patience=10, verbose=1, mode='min', min_delta=MIN_DELTA, cooldown=0, min_lr=0),
        tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(experiment, 'model.h5'), monitor='loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=1),
        tf.keras.callbacks.CSVLogger(filename=os.path.join(experiment, 'train.csv'), separator=',', append=False),
    ]

    x_train, y_train = data.load(train)
    x_train = tf.keras.applications.vgg19.preprocess_input(x_train)
    x_train, x_validation, y_train, y_validation = sklearn.model_selection.train_test_split(x_train, y_train, test_size=0.03, shuffle=True, stratify=y_train)

    model.fit(
        x=x_train,
        y=y_train,
        validation_data=(x_validation, y_validation),
        batch_size=bs,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks,
        shuffle=True,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=helpers.is_dir, required=True)
    parser.add_argument('--train', type=helpers.is_file, required=True)
    parser.add_argument('--extract-until', type=int, required=True)
    parser.add_argument('--freeze-until', type=int, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--bs', type=int, required=True)
    args = parser.parse_args()

    helpers.seed()
    train(args.experiment, args.train, args.extract_until, args.freeze_until, args.epochs, args.bs)
    helpers.fix_layer0(os.path.join(args.experiment, 'model.h5'), (None, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS), 'float32')
