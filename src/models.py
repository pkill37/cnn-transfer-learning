import tensorflow as tf


def vgg16(img_height, img_width, loss, optimizer, metrics, dropout, nb_layers=None):
    # Freeze the model's first nb_layers layers
    vgg16 = tf.keras.applications.vgg16.VGG16(weights='imagenet')
    for layer in vgg16.layers[1:nb_layers+1]:
        print(layer)
        layer.trainable = False

    # Dropout after each existing fully-connected layer
    fc1 = vgg16.get_layer('fc1')
    fc2 = vgg16.get_layer('fc2')
    dropout1 = tf.keras.layers.Dropout(rate=dropout)
    dropout2 = tf.keras.layers.Dropout(rate=dropout)
    x = dropout1(fc1.output)
    x = fc2(x)
    x = dropout2(x)

    # Install our own fully-connected layer for binary classification
    x = tf.keras.layers.Dense(units=1, activation='sigmoid')(x)

    # Stitch model
    model = tf.keras.models.Model(inputs=vgg16.input, outputs=x)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model
