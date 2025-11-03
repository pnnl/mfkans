# This function was borrowed as-is from the original pykan implementation
# to be able to compare the models on an equal footing
import torch
import numpy as np

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from jax import random, grad, vmap, jit

from KAN import KAN
from jax import vmap
import math

#import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import scipy.io
from jax.flatten_util import ravel_pytree


def yL(x, k, noise_level):
    
    c = 2*k*jnp.pi
    u = jnp.sin(c*x**2)
    f = 2*c*(jnp.cos(c*x**2)-2*c*x**2*jnp.sin(c*x**2))
    
    key = jax.random.PRNGKey(10)
    key, subkey = random.split(key)    
    noise = noise_level*random.normal(key, [len(x)])
    return u, f

def yH(x, k, noise_level):
    c = 2*k*jnp.pi
    u = jnp.sin(c*x**2)
    f = 2*c*(jnp.cos(c*x**2)-2*c*x**2*jnp.sin(c*x**2))
    
    key = jax.random.PRNGKey(10)
    key, subkey = random.split(key)    
    noise = noise_level*random.normal(key, [len(x)])
    return u, f





def create_dataset_LF(k=2, noise_level = 0):
    key = jax.random.PRNGKey(10)

    key, subkey = random.split(key)

    in_data_LF = random.uniform(key, [1000, 1])

    s_data_LF, f = yL(in_data_LF, k, noise_level)

    return in_data_LF, s_data_LF, f

def create_dataset_HF(k=4, noise_level = 0):
    key = jax.random.PRNGKey(1)

    key, subkey = random.split(key)
    in_data_HF = random.uniform(key,  [1000, 1])
    s_data_HF, f = yH(in_data_HF, k, noise_level)

    return in_data_HF, s_data_HF, f

def create_dataset_test(k_L, k_H):
    in_data_HF = np.linspace(0, 1, 200).reshape(-1, 1)
    s_data_HF, f_HF = yH(in_data_HF, k_H, 0)
    s_data_LF, f_LF = yL(in_data_HF, k_L, 0)

    return in_data_HF, s_data_HF, s_data_LF, f_HF, f_LF

