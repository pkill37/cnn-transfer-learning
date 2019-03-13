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

    parser = argparse.ArgumentParser()
    parser.add_argument('--images-path', type=str)
    parser.add_argument('--descriptions-path', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--img-height', type=int)
    parser.add_argument('--img-width', type=int)
    parser.add_argument('--batch-size', type=int)
    args = parser.parse_args()

    # Load best trained model
    model = tf.keras.models.load_model(args.model, custom_objects={
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
        workers=multiprocessing.cpu_count()-1 or 1,
        use_multiprocessing=True,
    )
    for score, metric in zip(scores, model.metrics_names):
        print("%s: %0.4f" % (metric, score))
