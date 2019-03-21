import os
import argparse
import multiprocessing
import tensorflow as tf
import metrics
import models
import data
import helpers


def train(experiments_path, train, validation, pretrained_model, extract_until, freeze_until, epochs, batch_size, lr, l1, l2):
    helpers.seed()

    model, preprocess_input, (img_height, img_width) = getattr(models, pretrained_model)(extract_until=extract_until, freeze_until=freeze_until, l1=l1, l2=l2)
    model.summary()

    callbacks = [
        tf.keras.callbacks.LearningRateScheduler(lambda epoch: lr*(0.1**int(epoch/10))),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_f1', factor=0.2, patience=10, verbose=0, mode='max', min_delta=0.0001, cooldown=0, min_lr=0.001),
        tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(experiments_path, 'best.hdf5'), monitor='val_f1', verbose=1, save_best_only=True, save_weights_only=False, mode='max', period=1),
        tf.keras.callbacks.TensorBoard(log_dir=experiments_path, histogram_freq=0, write_graph=True, write_images=True),
        tf.keras.callbacks.CSVLogger(filename=os.path.join(experiments_path, 'training_log.csv'), separator=',', append=False),
    ]

    x_train, y_train, class_weights = data.load_dataset(train)
    x_validation, y_validation, _ = data.load_dataset(validation)

    model.fit_generator(
        generator=data.BinaryLabelImageSequence(x_train, y_train, batch_size, True, preprocess_input),
        epochs=epochs,
        validation_data=data.BinaryLabelImageSequence(x_validation, y_validation, batch_size, False, preprocess_input),
        class_weight=class_weights,
        shuffle=True,
        verbose=1,
        callbacks=callbacks,
        workers=multiprocessing.cpu_count()-1 or 1,
        use_multiprocessing=True,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments-path', type=str, required=True)
    parser.add_argument('--train', type=str, required=True)
    parser.add_argument('--validation', type=str, required=True)
    parser.add_argument('--pretrained-model', choices=['vgg16', 'inceptionv3', 'resnet50'], required=True)
    parser.add_argument('--extract-until', type=int, required=True)
    parser.add_argument('--freeze-until', type=int, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--l1', type=float, required=True)
    parser.add_argument('--l2', type=float, required=True)
    args = parser.parse_args()

    train(args.experiments_path, args.train, args.validation, args.pretrained_model, args.extract_until, args.freeze_until, args.epochs, args.batch_size, args.lr, args.l1, args.l2)
