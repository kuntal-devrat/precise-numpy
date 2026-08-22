"""Random number generation, mirroring numpy.random's core API."""

from precise_numpy._precise_numpy import normal, rand, randint, randn, random_sample, seed, uniform

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
