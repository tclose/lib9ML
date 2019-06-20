import os
from numpy.random import RandomState


try:
    RAND_SEED = os.environ['NINML_TEST_RAND_SEED']
except KeyError:
    RAND_SEED = None

random_state = RandomState(RAND_SEED)
