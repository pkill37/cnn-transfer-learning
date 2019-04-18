import tensorflow as tf

import metrics


LOSS = 'binary_crossentropy'
METRICS = metrics.METRICS
OPTIMIZER = lambda lr: tf.keras.optimizers.SGD(lr=lr, momentum=0.9, decay=10e-6, nesterov=True)
IMG_SHAPE = { 'vgg19': (224, 224), 'inceptionv3': (299, 299) }


def extract(model, extract_until):
    return model.layers[extract_until].output


def freeze(model, freeze_until):
    for layer in model.layers[:freeze_until]:
        layer.trainable = False
    return model


def classifier(x, l1, l2, dropout):
    x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = tf.keras.layers.Dense(units=4096, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1, l2=l2))(x)
    x = tf.keras.layers.Dropout(rate=dropout)(x)
    x = tf.keras.layers.Dense(units=4096, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1, l2=l2))(x)
    x = tf.keras.layers.Dropout(rate=dropout)(x)
    x = tf.keras.layers.Dense(units=1, activation='sigmoid')(x)
    return x


def vgg19(extract_until, freeze_until, lr, l1, l2, dropout):
    assert extract_until >= freeze_until

    input_tensor = tf.keras.layers.Input(shape=(*IMG_SHAPE['vgg19'], 3))
    vgg19 = tf.keras.applications.vgg19.VGG19(weights='imagenet', include_top=False, input_tensor=input_tensor)

    x = freeze(vgg19, freeze_until)
    x = extract(vgg19, extract_until)
    x = classifier(x, l1, l2, dropout)

    model = tf.keras.models.Model(inputs=input_tensor, outputs=x, name='vgg19')
    model.compile(loss=LOSS, optimizer=OPTIMIZER(lr), metrics=METRICS)
    return model, tf.keras.applications.vgg19.preprocess_input, IMG_SHAPE['vgg19']


def inceptionv3(extract_until, freeze_until, lr, l1, l2, dropout):
    assert extract_until >= freeze_until

    input_tensor = tf.keras.layers.Input(shape=(*IMG_SHAPE['inceptionv3'], 3))
    inceptionv3 = tf.keras.applications.inception_v3.InceptionV3(weights='imagenet', include_top=False, input_tensor=input_tensor)

    x = freeze(inceptionv3, freeze_until)
    x = extract(inceptionv3, extract_until)
    x = classifier(x, l1, l2, dropout)

    model = tf.keras.models.Model(inputs=input_tensor, outputs=x, name='inceptionv3')
    model.compile(loss=LOSS, optimizer=OPTIMIZER(lr), metrics=METRICS)
    return model, tf.keras.applications.inception_v3.preprocess_input, IMG_SHAPE['inceptionv3']
