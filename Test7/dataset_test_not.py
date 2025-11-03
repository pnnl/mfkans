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
import pickle



def create_dataset_LF(noise_level = 0):
    with open('HF_data_cylinder.pkl', 'rb') as f:
        w = pickle.load(f)
      
    mask1 = np.arange(1, 257, 3)
    mask2 = np.arange(0, 129, 3)
    
    x = np.linspace(0, 200, len(mask1))/200
    y = np.linspace(0, 100, len(mask2))/100
    t = np.linspace(0, 201, 201)/201
    X, T, Y = np.meshgrid(x, t, y)
    
    w = w[200:401, :, :]
    w = w[:, mask1, :]
    w = w[:, :, mask2]
    
    Nt = X.shape[0]
     
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)
    T = T.reshape(-1, 1)
    
    N = len(mask1)*len(mask2)
    w = w.reshape(-1,  1)
    
    in_data = np.concatenate([X, Y, T], axis=1)

    return in_data, w


def create_dataset_test_LF(noise_level = 0):
    with open('HF_data_cylinder.pkl', 'rb') as f:
        w = pickle.load(f)
      
    mask1 = np.arange(1, 257, 3)
    mask2 = np.arange(0, 129, 3)
    
#    x = np.linspace(0, 200, len(mask1))
#    y = np.linspace(0, 100, len(mask2))
#    t = np.linspace(201, 401, 201)

    x = np.linspace(0, 200, len(mask1))/200
    y = np.linspace(0, 100, len(mask2))/100
    t = np.linspace(0, 201, 201)/201


    X, T, Y = np.meshgrid(x, t, y)
    
    w = w[200:401, :, :]
    w = w[:, mask1, :]
    w = w[:, :, mask2]
    
    Nt = X.shape[0]
     
    X = X.reshape(201, -1, 1)
    Y = Y.reshape(201, -1, 1)
    T = T.reshape(201, -1, 1)
    
    N = len(mask1)*len(mask2)
    w = w.reshape(201, -1,  1)
    
    in_data = np.concatenate([X, Y, T], axis=2)

    return in_data, w



def create_dataset_HF(noise_level = 0):
    with open('HF_data_cylinder.pkl', 'rb') as f:
        w = pickle.load(f)
      
    mask1 = np.arange(1, 257, 1)
    mask2 = np.arange(0, 129, 1)
    
#    x = np.linspace(0, 200, len(mask1))
#    y = np.linspace(0, 100, len(mask2))
#    t = np.linspace(201, 301, 101)
    x = np.linspace(0, 200, len(mask1))/200
    y = np.linspace(0, 100, len(mask2))/100
    t = np.linspace(0, 101, 101)/201

    X, T, Y = np.meshgrid(x, t, y)
    
    w = w[200:301, :, :]
    w = w[:, mask1, :]
    w = w[:, :, mask2]
    
    Nt = X.shape[0]
     
    X = X.reshape( -1, 1)
    Y = Y.reshape(  -1, 1)
    T = T.reshape(-1, 1)
    
    N = len(mask1)*len(mask2)
    w = w.reshape(-1,  1)
    
    in_data = np.concatenate([X, Y, T], axis=1)

#    out_points = np.concatenate([X, Y, T], axis=1)
    
    mask1 = np.arange(1, 257, 3)
    mask2 = np.arange(0, 129, 3)
    
    
    with open('HF_data_cylinder.pkl', 'rb') as f:
        wLF = pickle.load(f)
    wLF = wLF[200:301, :, :]
    wLF = wLF[:, mask1, :]
    wLF = wLF[:, :, mask2]
    
    
    mask1 = np.arange(1, 257, 1)
    mask2 = np.arange(0, 129, 1)
    t = np.arange(101)
    x = (mask1-1)/3
    y = mask2/3
#x = np.linspace(0, 200, len(mask1))
#y = np.linspace(0, 100, len(mask2))
#t = np.linspace(0, 201, 201)/201
    X, T, Y = np.meshgrid(x, t, y)

    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1) 
    T = T.reshape(-1, 1)
    out_points = np.concatenate([T, X, Y], axis=1)

    
    out = jax.scipy.ndimage.map_coordinates(wLF, out_points.T, order=1)
    #inter = scipy.interpolate.LinearNDInterpolator(in_points, wLF.reshape(-1))
    
   # out = inter(in_data)
    out = out.reshape(-1, 1)
    in_data = np.concatenate([in_data[:, 0:2], out], axis=1)
    return in_data, w



def create_dataset_test():
    with open('HF_data_cylinder.pkl', 'rb') as f:
        w = pickle.load(f)
      
    mask1 = np.arange(1, 257, 1)
    mask2 = np.arange(0, 129, 1)
    x = np.linspace(0, 200, len(mask1))/200
    y = np.linspace(0, 100, len(mask2))/100
    t = np.linspace(0, 201, 201)/201

    X, T, Y = np.meshgrid(x, t, y)
    
    w = w[200:401, :, :]
    w = w[:, mask1, :]
    w = w[:, :, mask2]
     
    X = X.reshape(201, -1, 1)
    Y = Y.reshape(201, -1, 1)
    T = T.reshape(201, -1, 1)
    w = w.reshape(201, -1,  1)    
    in_data = np.concatenate([X, Y, T], axis=2)

    mask1 = np.arange(1, 257, 3)
    mask2 = np.arange(0, 129, 3)    
    with open('HF_data_cylinder.pkl', 'rb') as f:
        wLF = pickle.load(f)
    wLF = wLF[200:401, :, :]
    wLF = wLF[:, mask1, :]
    wLF = wLF[:, :, mask2]
    
    mask1 = np.arange(1, 257, 1)
    mask2 = np.arange(0, 129, 1)
    t = np.arange(201)
    x = (mask1-1)/3
    y = mask2/3
    X, T, Y = np.meshgrid(x, t, y)

    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1) 
    T = T.reshape(-1, 1)
    out_points = np.concatenate([T, X, Y], axis=1)

    
    out = jax.scipy.ndimage.map_coordinates(wLF, out_points.T, order=1)
    
    #inter = scipy.interpolate.LinearNDInterpolator(in_points, wLF.reshape(-1))
    
   # out = inter(out_points)
    out = out.reshape(201, -1, 1)
    in_data = np.concatenate([in_data[:, :, 0:2], out], axis=2)

    return in_data,  w

