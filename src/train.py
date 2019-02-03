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

    pwd = os.path.realpath(__file__)
    out_dir = os.path.abspath(os.path.join(pwd, '../../out/')) + '/'
    data_dir = os.path.abspath(os.path.join(pwd, '../../data/')) + '/'

    parser = argparse.ArgumentParser()
    parser.add_argument('--images_path', type=str, default=data_dir+'images/')
    parser.add_argument('--descriptions_path', type=str, default=data_dir+'descriptions/')
    parser.add_argument('--experiment', type=str, default=out_dir)
    parser.add_argument('--img_height', type=int, default=224)
    parser.add_argument('--img_width', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--nb_layers', type=int, default=0) # how many layers to freeze
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--loss', type=str, default='binary_crossentropy')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--augmentation', default=False, action='store_true')
    args = parser.parse_args()

    model = models.vgg16(
        img_height=args.img_height,
        img_width=args.img_width,
        loss=args.loss,
        metrics=[metrics.true_positive(), metrics.true_negative(), metrics.false_negative(), metrics.false_positive(), metrics.precision(), metrics.recall(), metrics.f1_score(), 'accuracy'],
        optimizer=args.optimizer,
        dropout=args.dropout,
        nb_layers=args.nb_layers,
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_f1_score', min_delta=0, patience=20, verbose=0, mode='max', baseline=None),
        tf.keras.callbacks.LearningRateScheduler(lambda epoch: args.learning_rate*(0.1**int(epoch/10))),
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
        augmentation=args.augmentation,
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
