import os
import argparse
import multiprocessing
import tensorflow as tf
import metrics
import models
import data
import helpers


def train(experiment, images_path, descriptions_path, pretrained_model, extract_until, freeze_until, epochs, batch_size, lr):
    helpers.seed()

    if pretrained_model == 'vgg16':
        model, preprocess_input, (img_height, img_width) = models.vgg16(extract_until=extract_until, freeze_until=freeze_until)
    elif pretrained_model == 'inceptionv3':
        model, preprocess_input, (img_height, img_width) = models.inceptionv3(extract_until=extract_until, freeze_until=freeze_until)
    elif pretrained_model == 'resnet50':
        model, preprocess_input, (img_height, img_width) = models.resnet50(extract_until=extract_until, freeze_until=freeze_until)
    model.summary()

    callbacks = [
        tf.keras.callbacks.LearningRateScheduler(lambda epoch: lr*(0.1**int(epoch/10))),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_f1_score', factor=0.2, patience=10, verbose=0, mode='max', min_delta=0.0001, cooldown=0, min_lr=0.001),
        tf.keras.callbacks.ModelCheckpoint(filepath=experiment+'best.hdf5', monitor='val_f1_score', verbose=1, save_best_only=True, save_weights_only=False, mode='max', period=1),
        tf.keras.callbacks.TensorBoard(log_dir=experiment + 'tensorboard', histogram_freq=0, write_graph=True, write_images=True),
        tf.keras.callbacks.CSVLogger(filename=experiment + 'training_log.csv', separator=',', append=False),
    ]

    train_generator, validation_generator, _ = data.generators(
        images_path=images_path,
        descriptions_path=descriptions_path,
        img_height=img_height,
        img_width=img_width,
        split=(0.8, 0.1, 0.1),
        batch_size=batch_size,
        augmentation=True,
        preprocess_input=preprocess_input,
    )

    model.fit_generator(
        generator=train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        shuffle=True,
        verbose=1,
        callbacks=callbacks,
        workers=multiprocessing.cpu_count()-1 or 1,
        use_multiprocessing=True,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str)
    parser.add_argument('--images-path', type=str)
    parser.add_argument('--descriptions-path', type=str)
    parser.add_argument('--pretrained-model', choices=['vgg16', 'inceptionv3', 'resnet50'])
    parser.add_argument('--extract-until', type=int)
    parser.add_argument('--freeze-until', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--lr', type=float)
    args = parser.parse_args()

    train(args.experiment, args.images_path, args.descriptions_path, args.pretrained_model, args.extract_until, args.freeze_until, args.epochs, args.batch_size, args.lr)
