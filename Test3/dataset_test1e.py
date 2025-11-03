# This function was borrowed as-is from the original pykan implementation
# to be able to compare the models on an equal footing
import torch
import numpy as np

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from jax import random, grad, vmap, jit

from jax import vmap
import math

import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import scipy.io
from jax.flatten_util import ravel_pytree


def yL(x, y):
    u = np.zeros(len(x))
    for i in np.arange(len(x)):
        u[i] = jnp.sin(12*jnp.pi*x[i])
    return u

def yH(x, y):
    def yL_s(x, y):
        u = jnp.sin(12*jnp.pi*x)
        return u
    
    u = np.zeros(len(x))
    for i in np.arange(len(x)):
            u[i] = 2*(yL_s(x[i], y[i]))+ jnp.sin(12*y[i]) 

    return u





def create_dataset_LF():
    x_data_LF = jnp.linspace(0, 1, 100)
    y_data_LF = jnp.linspace(0, 1, 100)
    XX, YY = jnp.meshgrid(x_data_LF, y_data_LF)
    XX = XX.reshape(-1, 1)
    YY = YY.reshape(-1, 1)
    in_data_LF = jnp.concatenate([XX, YY], axis=1)
    s_data_LF = yL(in_data_LF[:, 0], in_data_LF[:, 1])

    return in_data_LF, s_data_LF

def create_dataset_HF():
    x_data_LF = jnp.linspace(.01, .99, 15)
    y_data_LF = jnp.linspace(.01, .99, 20)
    
    x_data_LF = jnp.linspace(.01, .99, 10)
    y_data_LF = jnp.linspace(.01, .99, 15)
    
    
    XX, YY = jnp.meshgrid(x_data_LF, y_data_LF)
    XX = XX.reshape(-1, 1)
    YY = YY.reshape(-1, 1)
    in_data_HF = jnp.concatenate([XX, YY], axis=1)
    s_data_HF = yH(in_data_HF[:, 0], in_data_HF[:, 1])

    return in_data_HF, s_data_HF

def create_dataset_test():
    x_data_LF = jnp.linspace(0, 1, 100)
    y_data_LF = jnp.linspace(0, 1, 100)
    XX, YY = jnp.meshgrid(x_data_LF, y_data_LF)
    XX = XX.reshape(-1, 1)
    YY = YY.reshape(-1, 1)
    in_data_HF = jnp.concatenate([XX, YY], axis=1)
    
    s_data_HF = yH(in_data_HF[:, 0], in_data_HF[:, 1])
    return in_data_HF, s_data_HF

