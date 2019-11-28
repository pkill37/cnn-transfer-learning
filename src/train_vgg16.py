import os
import argparse

import tensorflow as tf
import sklearn

import data
import helpers


EPSILON = 10e-3
IMG_WIDTH = 224
IMG_HEIGHT = 224
IMG_CHANNELS = 3


def vgg16(extract_until, freeze_until, units, l2, dropout):
    assert extract_until >= freeze_until

    # Extract and freeze pre-trained model layers
    vgg16 = tf.keras.applications.vgg16.VGG16(weights='imagenet', input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS), include_top=False)
    model = tf.keras.models.Sequential()
    for i in range(0, extract_until+1): # i=0 is the input layer, i>0 are the actual model layers
        layer = vgg16.layers[i]
        layer.trainable = True if (i > freeze_until) and helpers.has_parameters(layer) else False
        layer.kernel_regularizer = tf.keras.regularizers.l2(l2) if helpers.has_parameters(layer) else None
        model.add(layer)

    # Classifier
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dense(units=units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2)))
    model.add(tf.keras.layers.Dropout(rate=dropout))
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(l2)))
    return model


def main(experiments, train_set, epochs, batch_size, extract_until, freeze_until, units, l2, dropout, patience, lr, m_fraction):
    # Set PRNG seeds so that all runs have the same initial conditions
    helpers.seed()

    # Standardize and split data
    x, y = data.load(train_set)
    x = tf.keras.applications.vgg16.preprocess_input(x)
    x_train, x_validation, y_train, y_validation = sklearn.model_selection.train_test_split(x, y, test_size=0.15, shuffle=True, stratify=y)
    s = int(len(y_train)*m_fraction)
    x_train = x_train[:s]
    y_train = y_train[:s]
    del x
    del y

    # Settings
    run = os.path.join(experiments, f'extract{extract_until:03}_freeze{freeze_until:03}_units{units}_lambda{l2}_dropout{dropout}_patience{patience}_lr{lr}_mfraction{m_fraction}')
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
    model = vgg16(extract_until, freeze_until, units, l2, dropout)
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.SGD(lr=lr, momentum=0.9, decay=0.0, nesterov=False), metrics=['accuracy'])
    model.fit(x=x_train, y=y_train, validation_data=(x_validation, y_validation), batch_size=batch_size, epochs=epochs, verbose=1, callbacks=callbacks, shuffle=True)
    model.save(model_filename)

    # Cleanup
    del model
    helpers.fix_layer0(model_filename, (None, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS), 'float32')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments', type=helpers.is_dir, required=True)
    parser.add_argument('--train-set', type=helpers.is_file, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--extract-until', type=int, required=True)
    parser.add_argument('--freeze-until', type=int, required=True)
    parser.add_argument('--units', type=int, required=True)
    parser.add_argument('--l2', type=float, required=True)
    parser.add_argument('--dropout', type=float, required=True)
    parser.add_argument('--patience', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--m-fraction', type=float, required=True)
    args = parser.parse_args()
    main(args.experiments, args.train_set, args.epochs, args.batch_size, args.extract_until, args.freeze_until, args.units, args.l2, args.dropout, args.patience, args.lr, args.m_fraction)
