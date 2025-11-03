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

#import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import scipy.io
from jax.flatten_util import ravel_pytree


def create_dataset_LF(Test=False, N=100):
    
    xin = jnp.zeros([N, 28**2])
    xout = jnp.zeros([N, 1])
    if Test:
        filename_psi = "data/UE_CM_14_psi_test.txt"
        filename_in = "data/mnist_img_test.txt"
    else:
        filename_psi = "data/UE_CM_14_psi_train.txt"
        filename_in = "data/mnist_img_train.txt"

    with open(filename_psi, 'r') as f:
        lines = f.read().split('\n')
        xout = jnp.asarray([float(i) for i in lines[0:N]]).reshape(-1, 1)/200

    with open(filename_in, 'r') as f:
        lines = f.read().split()
        xin = jnp.asarray([float(i) for i in lines[0:(784*N)]]).reshape(N, -1)/256
        
    return xin, xout

def create_dataset_HF(Test=False, N=100):
    
    xin = jnp.zeros([N, 28**2])
    xout = jnp.zeros([N, 1])
    if Test:
        filename_psi = "data/UE_psi_test.txt"
        filename_in = "data/mnist_img_test.txt"
    else:
        filename_psi = "data/UE_psi_train.txt"
        filename_in = "data/mnist_img_train.txt"

    with open(filename_psi, 'r') as f:
        lines = f.read().split('\n')
        xout = jnp.asarray([float(i) for i in lines[0:N]]).reshape(-1, 1)/200

    with open(filename_in, 'r') as f:
        lines = f.read().split()
        xin = jnp.asarray([float(i) for i in lines[0:(784*N)]]).reshape(N, -1)/256
        
    return xin, xout

def create_dataset_test(LF=True, N=100):
    
   # xout = jnp.zeros([N, 1])
    filename_in = "data/mnist_img_test.txt"

    if LF:
        filename_psi = "data/UE_CM_14_psi_test.txt"
    else:
        filename_psi = "data/UE_psi_test.txt"

    with open(filename_psi, 'r') as f:
        lines = f.read().split('\n')
        xout = jnp.asarray([float(i) for i in lines[0:N]]).reshape(-1, 1)/200

    with open(filename_in, 'r') as f:
        lines = f.read().split()
        xin = jnp.asarray([int(i) for i in lines[0:(784*N)]]).reshape(N, -1)/256
   
        
    return xin, xout


def create_dataset_test_out(LF=True, N=100):
    
   # xout = jnp.zeros([N, 1])

    if LF:
        filename_psi = "data/UE_CM_14_psi_test.txt"
    else:
        filename_psi = "data/UE_psi_test.txt"

    with open(filename_psi, 'r') as f:
        lines = f.read().split('\n')
        xout = jnp.asarray([float(i) for i in lines[0:N]]).reshape(-1, 1)


        
    return xout

def create_dataset_train_out(LF=True, N=100):
    
   # xout = jnp.zeros([N, 1])

    if LF:
        filename_psi = "data/UE_CM_14_psi_train.txt"
    else:
        filename_psi = "data/UE_psi_train.txt"

    with open(filename_psi, 'r') as f:
        lines = f.read().split('\n')
        xout = jnp.asarray([float(i) for i in lines[0:N]]).reshape(-1, 1)


        
    return xout

