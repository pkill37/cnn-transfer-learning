import os
import shutil
import time

import tensorflow as tf


def is_dir(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


def is_file(string):
    if os.path.isfile(string):
        return string
    else:
        raise FileNotFoundError(string)


def smallest(a, b):
    if len(a) > len(b):
        return b
    elif len(a) < len(b):
        return a
    else:
        return None


def biggest(a, b):
    if len(a) > len(b):
        return a
    elif len(a) < len(b):
        return b
    else:
        return None


def create_or_recreate_dir(dir_path):
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


def create_if_none(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


def validate_exists_and_dir(dir_path, arg_name):
    if not os.path.exists(dir_path):
        raise ValueError("{0} {1} does not exist".format(arg_name, dir_path))

    if not os.path.isdir(dir_path):
        raise ValueError("{0} {1} is not a dir".format(arg_name, dir_path))


def touch(fname, times=None):
    with open(fname, 'a'):
        os.utime(fname, times)


def seed():
    from random import seed
    seed(1)
    import numpy.random
    numpy.random.seed(2)
    from tensorflow import set_random_seed
    set_random_seed(3)


# https://github.com/keras-team/keras/issues/10417
def fix_layer0(filename, batch_input_shape, dtype):
    import h5py
    import json
    with h5py.File(filename, 'r+') as f:
        model_config = json.loads(f.attrs['model_config'].decode('utf-8'))
        layer0 = model_config['config']['layers'][0]['config']
        layer0['batch_input_shape'] = batch_input_shape
        layer0['dtype'] = dtype
        f.attrs['model_config'] = json.dumps(model_config).encode('utf-8')


def has_parameters(layer):
    return len(layer.get_weights()) > 0


class TrainingTimeLogger(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        logs['time'] = time.time() - self.epoch_time_start
