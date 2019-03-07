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


def vgg16(img_height, img_width, nb_layers=None):
    input_tensor = tf.keras.layers.Input(shape=(img_height, img_width, 3))
    vgg16 = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)
    freeze(vgg16, nb_layers)

    x = vgg16.output
    x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(units=1, activation='sigmoid')(x)

    # Compile model
    model = tf.keras.models.Model(inputs=input_tensor, outputs=x, name='vgg16')
    model.compile(loss=LOSS, optimizer='adam', metrics=METRICS)
    return model


def inceptionv3(img_height, img_width, nb_layers=None):
    input_tensor = tf.keras.layers.Input(shape=(img_height, img_width, 3))
    inceptionv3 = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_tensor=input_tensor)
    freeze(inceptionv3, nb_layers)

    x = inceptionv3.output
    x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(units=1, activation='sigmoid')(x)

    # Compile model
    model = tf.keras.models.Model(inputs=input_tensor, outputs=x, name='inceptionv3')
    model.compile(loss=LOSS, optimizer='adam', metrics=METRICS)
    return model


def resnet152(img_height, img_width, nb_layers=None):
    input_tensor = tf.keras.layers.Input(shape=(img_height, img_width, 3))
    resnet152 = tf.keras.applications.ResNet152(weights='imagenet', include_top=False, input_tensor=input_tensor)
    freeze(resnet152, nb_layers)

    x = resnet152.output
    x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(units=1, activation='sigmoid')(x)

    # Compile model
    model = tf.keras.models.Model(inputs=input_tensor, outputs=x, name='resnet152')
    model.compile(loss=LOSS, optimizer='adam', metrics=METRICS)
    return model
