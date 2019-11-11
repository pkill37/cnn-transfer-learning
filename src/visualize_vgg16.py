import os
import argparse

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import helpers


def main(img_filename):
    # load the model
    model = tf.keras.applications.vgg16.VGG16()
    idxs = [3, 6, 10, 14, 18]
    outputs = [model.layers[i].output for i in idxs]
    model = tf.keras.models.Model(inputs=model.inputs, outputs=outputs)

    # load the image with the required shape
    img = tf.keras.preprocessing.image.load_img(img_filename, target_size=(224, 224))
    # convert the image to an array
    img = tf.keras.preprocessing.image.img_to_array(img)
    # expand dimensions so that it represents a single 'sample'
    img = np.expand_dims(img, axis=0)
    # prepare the image (e.g. scale pixel values for the vgg)
    img = tf.keras.applications.vgg16.preprocess_input(img)
    # get feature map for first hidden layer
    feature_maps = model.predict(img)
    # plot the output from each block
    square = 8
    for block, fmap in enumerate(feature_maps):
        # plot all 64 maps in an 8x8 squares
        ix = 1
        for _ in range(square):
            for __ in range(square):
                # specify subplot and turn of axis
                ax = plt.subplot(square, square, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                # plot filter channel in grayscale
                plt.imshow(fmap[0, :, :, ix-1])
                ix += 1

        # save the figure
        plt.savefig(f'block{block+1}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-filename', type=helpers.is_file, required=True)
    args = parser.parse_args()
    main(args.img_filename)
