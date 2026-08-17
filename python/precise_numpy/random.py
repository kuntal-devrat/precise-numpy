"""Random number generation, mirroring numpy.random's core API."""

from precise_numpy._precise_numpy import seed, rand, randn, randint, uniform, normal
from precise_numpy._precise_numpy import random_sample

# Aliases matching numpy.random names.
random = random_sample
random_sample = random_sample
rand = rand
randn = randn
randint = randint
uniform = uniform
normal = normal
seed = seed

__all__ = [
    "seed",
    "rand",
    "randn",
    "randint",
    "uniform",
    "normal",
    "random",
    "random_sample",
]
