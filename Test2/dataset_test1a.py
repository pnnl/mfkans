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


def create_dataset_LF():
    t = jnp.linspace(0, 1, 51).reshape(-1, 1)
    x = jnp.sin(8*jnp.pi*t)#+0.5*t
    return t, x

def create_dataset_HF():
    t = jnp.linspace(0, 0.98, 14).reshape(-1, 1)
    x = (t-jnp.sqrt(2))*jnp.sin(8*jnp.pi*t)**2
        
    return t, x

def create_dataset_test():
    t = jnp.linspace(0, 1, 200).reshape(-1, 1)
    x = (t-jnp.sqrt(2))*jnp.sin(8*jnp.pi*t)**2

    return t, x

