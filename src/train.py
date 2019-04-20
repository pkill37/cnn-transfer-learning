import os
import argparse

import tensorflow as tf
import numpy as np

import metrics
import models
import data
import helpers


def train(experiments_path, train, validation, pretrained_model, extract_until, freeze_until, epochs, batch_size, lr, l1, l2, dropout):
    model, preprocess_input, (img_height, img_width) = getattr(models, pretrained_model)(extract_until=extract_until, freeze_until=freeze_until, lr=lr, l1=l1, l2=l2, dropout=dropout)
    model.summary()

    callbacks = [
        #tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=0, mode='min', baseline=None),
        tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(experiments_path, 'model.hdf5'), monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=1),
        tf.keras.callbacks.TensorBoard(log_dir=experiments_path, histogram_freq=0, write_graph=True, write_images=True),
        tf.keras.callbacks.CSVLogger(filename=os.path.join(experiments_path, 'log.csv'), separator=',', append=False),
    ]

    x_train, y_train = data.load(train)
    x_validation, y_validation = data.load(validation)

    x_train = preprocess_input(x_train)
    x_validation = preprocess_input(x_validation)

    model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks,
        validation_data=(x_validation, y_validation),
        shuffle=True,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments-path', type=helpers.is_dir, required=True)
    parser.add_argument('--train', type=helpers.is_file, required=True)
    parser.add_argument('--validation', type=helpers.is_file, required=True)
    parser.add_argument('--pretrained-model', choices=['vgg19', 'inceptionv3'], required=True)
    parser.add_argument('--extract-until', type=int, required=True)
    parser.add_argument('--freeze-until', type=int, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--l1', type=float, required=True)
    parser.add_argument('--l2', type=float, required=True)
    parser.add_argument('--dropout', type=float, required=True)
    args = parser.parse_args()

    train(args.experiments_path, args.train, args.validation, args.pretrained_model, args.extract_until, args.freeze_until, args.epochs, args.batch_size, args.lr, args.l1, args.l2, args.dropout)
