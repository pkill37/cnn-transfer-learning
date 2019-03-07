import os
import argparse
import multiprocessing
import tensorflow as tf
import metrics
import models
import data
import helpers


if __name__ == '__main__':
    helpers.seed()

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str)
    parser.add_argument('--images-path', type=str)
    parser.add_argument('--descriptions-path', type=str)
    parser.add_argument('--img-height', type=int)
    parser.add_argument('--img-width', type=int)
    parser.add_argument('--model', choices=['vgg16', 'inceptionv3', 'resnet152'])
    parser.add_argument('--nb-layers', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--lr', type=float)
    args = parser.parse_args()

    if args.model == 'vgg16':
        model = models.vgg16(img_height=args.img_height, img_width=args.img_width, nb_layers=args.nb_layers)
    elif args.model == 'inceptionv3':
        model = models.inceptionv3(img_height=args.img_height, img_width=args.img_width, nb_layers=args.nb_layers)
    elif args.model == 'resnet152':
        model = models.resnet152(img_height=args.img_height, img_width=args.img_width, nb_layers=args.nb_layers)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_f1_score', min_delta=0, patience=20, verbose=0, mode='max', baseline=None),
        tf.keras.callbacks.LearningRateScheduler(lambda epoch: args.lr*(0.1**int(epoch/10))),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_f1_score', factor=0.2, patience=10, verbose=0, mode='max', min_delta=0.0001, cooldown=0, min_lr=0.001),
        tf.keras.callbacks.ModelCheckpoint(filepath=args.experiment+'best.hdf5', monitor='val_f1_score', verbose=1, save_best_only=True, save_weights_only=False, mode='max', period=1),
        tf.keras.callbacks.TensorBoard(log_dir=args.experiment + 'tensorboard', histogram_freq=0, write_graph=True, write_images=True),
        tf.keras.callbacks.CSVLogger(filename=args.experiment + 'training_log.csv', separator=',', append=False),
    ]

    train_generator, validation_generator, _ = data.generators(
        images_path=args.images_path,
        descriptions_path=args.descriptions_path,
        img_height=args.img_height,
        img_width=args.img_width,
        split=(0.8, 0.1, 0.1),
        batch_size=args.batch_size,
        augmentation=True,
    )

    model.fit_generator(
        generator=train_generator,
        epochs=args.epochs,
        validation_data=validation_generator,
        shuffle=True,
        verbose=1,
        callbacks=callbacks,
        workers=multiprocessing.cpu_count()-1 or 1,
        use_multiprocessing=True,
    )
