import os


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
