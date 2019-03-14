import tensorflow as tf
import metrics


LOSS = 'binary_crossentropy'
METRICS = [metrics.precision(), metrics.recall(), metrics.f1_score()]


def freeze(model, nb_layers):
    # Freeze the model's first nb_layers layers
    for layer in model.layers[:nb_layers]:
        layer.trainable = False
    return model


def ensemble(models, model_input):
    x = [model(model_input) for model in models]
    x = tf.keras.layers.Average()(x)

    # Compile the model
    model = tf.keras.models.Model(inputs=model_input, outputs=x, name='ensemble')
    model.compile(loss=LOSS, optimizer='adam', metrics=METRICS)
    return model


def vgg16(extract_until, freeze_until):
    assert extract_until >= freeze_until

    img_height = img_width = 224
    input_tensor = tf.keras.layers.Input(shape=(img_height, img_width, 3))
    vgg16 = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)
    freeze(vgg16, freeze_until)

    x = vgg16.layers[extract_until].output
    x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(units=1, activation='sigmoid')(x)

    model = tf.keras.models.Model(inputs=input_tensor, outputs=x, name='vgg16')
    model.compile(loss=LOSS, optimizer='adam', metrics=METRICS)
    return model, tf.keras.applications.vgg16.preprocess_input, (img_height, img_width)


def inceptionv3(extract_until, freeze_until):
    assert extract_until >= freeze_until

    img_height = img_width = 299
    input_tensor = tf.keras.layers.Input(shape=(img_height, img_width, 3))
    inceptionv3 = tf.keras.applications.inception_v3.InceptionV3(weights='imagenet', include_top=False, input_tensor=input_tensor)
    freeze(inceptionv3, freeze_until)

    x = inceptionv3.layers[extract_until].output
    x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(units=1, activation='sigmoid')(x)

    model = tf.keras.models.Model(inputs=input_tensor, outputs=x, name='inceptionv3')
    model.compile(loss=LOSS, optimizer='adam', metrics=METRICS)
    return model, tf.keras.applications.inception_v3.preprocess_input, (img_height, img_width)


def resnet50(extract_until, freeze_until):
    assert extract_until >= freeze_until

    img_height = img_width = 224
    input_tensor = tf.keras.layers.Input(shape=(img_height, img_width, 3))
    resnet50 = tf.keras.applications.resnet50.ResNet50(weights='imagenet', include_top=False, input_tensor=input_tensor)
    freeze(resnet50, freeze_until)

    x = resnet50.layers[extract_until].output
    x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(units=1, activation='sigmoid')(x)

    model = tf.keras.models.Model(inputs=input_tensor, outputs=x, name='resnet50')
    model.compile(loss=LOSS, optimizer='adam', metrics=METRICS)
    return model, tf.keras.applications.resnet50.preprocess_input, (img_height, img_width)
