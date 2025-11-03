# This function was borrowed as-is from the original pykan implementation
# to be able to compare the models on an equal footing
import torch
import numpy as np

from dataset import create_dataset
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from jax import random, grad, vmap, jit

from KAN import KAN
from jax import vmap
import math

import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import scipy.io
from jax.flatten_util import ravel_pytree


def yL(x, noise_level = 0):
    
    def yH_s(x):
        u = 0.5*(0.1*jnp.exp(x[0]+x[1])-x[3]*jnp.sin(12*jnp.pi*x[2]) + x[2])
        return u

    
    u = np.zeros(len(x))

    for i in np.arange(len(x)):
        u[i] = 1.2*yH_s(x[i, :])-0.5
    
    key = jax.random.PRNGKey(10)

    key, subkey = random.split(key)
    
    noise = noise_level*random.normal(key, [len(x)])

    
    return u + noise

def yH(x, noise_level = 0):

    key = jax.random.PRNGKey(10)

    u = np.zeros(len(x))
    for i in np.arange(len(x)):
            u[i] = 0.5*(0.1*jnp.exp(x[i, 0]+x[i, 1])-x[i, 3]*jnp.sin(12*jnp.pi*x[i, 2]) + x[i, 2])
    noise = noise_level*random.normal(key, [len(x)])

    return u+noise





def create_dataset_LF(noise_level = 0):
    key = jax.random.PRNGKey(10)

    key, subkey = random.split(key)

    in_data_LF = random.uniform(key, [25000, 4])

    s_data_LF = yL(in_data_LF, noise_level)

    return in_data_LF, s_data_LF

def create_dataset_HF(noise_level = 0):
    key = jax.random.PRNGKey(1)

    key, subkey = random.split(key)
    in_data_HF = random.uniform(key,  [150, 4])
    s_data_HF = yH(in_data_HF, noise_level)

    return in_data_HF, s_data_HF

def create_dataset_test():
    key = jax.random.PRNGKey(100000)

    key, subkey = random.split(key)
    in_data_HF = random.uniform(key,  [100000, 4])
    s_data_HF = yH(in_data_HF, noise_level=0)
    s_data_LF = yL(in_data_HF, noise_level=0)

    return in_data_HF, s_data_HF, s_data_LF

