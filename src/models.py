import tensorflow as tf
import metrics


LOSS = 'binary_crossentropy'
METRICS = [metrics.precision(), metrics.recall(), metrics.f1_score()]


def extract(model, extract_until):
    return model.layers[extract_until].output


def freeze(model, freeze_until):
    for layer in model.layers[:freeze_until]:
        layer.trainable = False
    return model


def classifier(x, l1, l2):
    x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = tf.keras.layers.Dense(units=1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1, l2=l2))(x)
    return x


def vgg16(extract_until, freeze_until, l1, l2):
    assert extract_until >= freeze_until

    img_height = img_width = 224
    input_tensor = tf.keras.layers.Input(shape=(img_height, img_width, 3))
    vgg16 = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)

    x = freeze(vgg16, freeze_until)
    x = extract(vgg16, extract_until)
    x = classifier(x, l1, l2)

    model = tf.keras.models.Model(inputs=input_tensor, outputs=x, name='vgg16')
    model.compile(loss=LOSS, optimizer='adam', metrics=METRICS)
    return model, tf.keras.applications.vgg16.preprocess_input, (img_height, img_width)


def inceptionv3(extract_until, freeze_until):
    assert extract_until >= freeze_until

    img_height = img_width = 299
    input_tensor = tf.keras.layers.Input(shape=(img_height, img_width, 3))
    inceptionv3 = tf.keras.applications.inception_v3.InceptionV3(weights='imagenet', include_top=False, input_tensor=input_tensor)
    x = freeze(inceptionv3, freeze_until)
    x = extract(inceptionv3, extract_until)
    x = classifier(x)

    model = tf.keras.models.Model(inputs=input_tensor, outputs=x, name='inceptionv3')
    model.compile(loss=LOSS, optimizer='adam', metrics=METRICS)
    return model, tf.keras.applications.inception_v3.preprocess_input, (img_height, img_width)
