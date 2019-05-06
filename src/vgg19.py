import os
import argparse

import tensorflow as tf
import numpy as np

import metrics
import data
import helpers


LOSS = 'binary_crossentropy'
METRICS = metrics.METRICS
OPTIMIZER = lambda lr: tf.keras.optimizers.SGD(lr=lr, momentum=0.9, decay=10e-6, nesterov=True)

IMG_WIDTH = 224
IMG_HEIGHT = 224
IMG_CHANNELS = 3


def vgg19(extract_until=21, freeze_until=21, lr=0.001, l2=0.001):
    assert extract_until >= freeze_until

    # Extract and freeze pre-trained model layers
    vgg19 = tf.keras.applications.vgg19.VGG19(weights='imagenet', input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS), include_top=False)
    vgg19.summary()

    model = tf.keras.models.Sequential()
    for i in range(0, extract_until+1): # i=0 is the input layer, i>0 are the actual model layers
        layer = vgg19.layers[i]
        layer.trainable = True if i > freeze_until else False
        model.add(layer)

    # Classifier
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dense(units=512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2)))
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    model.compile(loss=LOSS, optimizer=OPTIMIZER(lr), metrics=METRICS)
    return model


def train(experiment, train, extract_until, freeze_until, lr, l2, epochs, bs):
    model = vgg19(extract_until, freeze_until, lr, l2)
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(experiment, 'model.h5'), monitor='loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=1),
        tf.keras.callbacks.TensorBoard(log_dir=experiment, histogram_freq=0, write_graph=True, write_images=True),
        tf.keras.callbacks.CSVLogger(filename=os.path.join(experiment, 'log.csv'), separator=',', append=False),
    ]

    x_train, y_train = data.load(train)
    x_train = tf.keras.applications.vgg19.preprocess_input(x_train)

    model.fit(
        x=x_train,
        y=y_train,
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
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--l2', type=float, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--bs', type=int, required=True)
    args = parser.parse_args()

    train(args.experiment, args.train, args.extract_until, args.freeze_until, args.lr, args.l2, args.epochs, args.bs)
    helpers.fix_layer0(os.path.join(args.experiment, 'model.h5'), (None, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS), 'float32')
