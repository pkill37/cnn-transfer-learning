import os
import multiprocessing
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import data
import helpers
import metrics


if __name__ == '__main__':
    helpers.seed()

    pwd = os.path.realpath(__file__)
    out_dir = os.path.abspath(os.path.join(pwd, '../../out/')) + '/'
    data_dir = os.path.abspath(os.path.join(pwd, '../../data/')) + '/'

    parser = argparse.ArgumentParser()
    parser.add_argument('--images_path', type=str, default=data_dir+'images/')
    parser.add_argument('--descriptions_path', type=str, default=data_dir+'descriptions/')
    parser.add_argument('--model', type=str, default=out_dir+'experiment_21/best.hdf5')
    parser.add_argument('--img_height', type=int, default=224)
    parser.add_argument('--img_width', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()

    # Load best trained model
    model = tf.keras.models.load_model(args.model, custom_objects={
        'true_positive': metrics.true_positive(),
        'true_negative': metrics.true_negative(),
        'false_negative': metrics.false_negative(),
        'false_positive': metrics.false_positive(),
        'precision': metrics.precision(),
        'recall': metrics.recall(),
        'f1_score': metrics.f1_score()
    })

    # Build test generator
    _, __, test_generator = data.generators(
        images_path=args.images_path,
        descriptions_path=args.descriptions_path,
        img_height=args.img_height,
        img_width=args.img_width,
        split=(0.8, 0.1, 0.1),
        batch_size=args.batch_size,
        augmentation=False,
    )

    # Evaluate model on test generator
    scores = model.evaluate_generator(
        generator=test_generator,
        verbose=1,
        #workers=multiprocessing.cpu_count()-1 or 1,
        #use_multiprocessing=True,
    )
    for score, metric in zip(scores, model.metrics_names):
        print("%s : %0.4f" % (metric, score))

    # Visualize the model's predictions
    plt.ion()
    for x_batch, y_batch in test_generator:
        y_pred = model.predict_on_batch(x_batch)
        for i in range(len(x_batch)):
            plt.imshow(x_batch[i,:,:,:])
            plt.title(f'Class {y_batch}')
            input('Press [Enter] to predict another mini-batch...')
            plt.close()
