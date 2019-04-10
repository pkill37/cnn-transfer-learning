import tensorflow as tf
import metrics


LOSS = 'binary_crossentropy'
METRICS = [metrics.true_positive(), metrics.true_negative(), metrics.false_positive(), metrics.false_negative(), metrics.precision(), metrics.recall(), metrics.f1_score()]
OPTIMIZER = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)
IMG_SHAPE = { 'vgg19': (224, 224), 'inceptionv3': (299, 299) }


def extract(model, extract_until):
    return model.layers[extract_until].output


def freeze(model, freeze_until):
    for layer in model.layers[:freeze_until]:
        layer.trainable = False
    return model


def classifier(x, dropout):
    x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(units=1, activation='sigmoid')(x)
    return x


def vgg19(extract_until, freeze_until, lr, dropout):
    assert extract_until >= freeze_until

    input_tensor = tf.keras.layers.Input(shape=(*IMG_SHAPE['vgg19'], 3))
    vgg19 = tf.keras.applications.vgg19.VGG19(weights='imagenet', include_top=False, input_tensor=input_tensor)

    x = freeze(vgg19, freeze_until)
    x = extract(vgg19, extract_until)
    x = classifier(x, dropout)

    model = tf.keras.models.Model(inputs=input_tensor, outputs=x, name='vgg19')
    model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)
    return model, tf.keras.applications.vgg19.preprocess_input, IMG_SHAPE['vgg19']


def inceptionv3(extract_until, freeze_until, lr, dropout):
    assert extract_until >= freeze_until

    input_tensor = tf.keras.layers.Input(shape=(*IMG_SHAPE['inceptionv3'], 3))
    inceptionv3 = tf.keras.applications.inception_v3.InceptionV3(weights='imagenet', include_top=False, input_tensor=input_tensor)

    x = freeze(inceptionv3, freeze_until)
    x = extract(inceptionv3, extract_until)
    x = classifier(x, dropout)

    model = tf.keras.models.Model(inputs=input_tensor, outputs=x, name='inceptionv3')
    model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)
    return model, tf.keras.applications.inception_v3.preprocess_input, IMG_SHAPE['inceptionv3']
