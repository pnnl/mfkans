import torch
import numpy as np
import sys
sys.path.insert(0, '../')

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

from SF_funcs_only import *
from dataset_test1a import *
import pickle

t_data_test, s_data_test = create_dataset_HF(Test=False, N=10000)
X_test = t_data_test
    

for N in [200, 400, 1000, 5000, 10000, 20000, 60000]:
    print(N)

    # Training epochsy
    num_epochs_LF = 10001
    
    # Epochs at which to change the grid & learning rate
    boundaries = [0]
    # Learning rate scales
    scales = [1.0]
    # Grid sizes to use
    grid_vals = [3]
    # Initial learning rate
    init_lr = 0.01
    
    # Corresponding dicts
    lr_scales = dict(zip(boundaries, scales))
    grid_upds = dict(zip(boundaries, grid_vals))
    
    # Create a piecewise constant schedule
    layer_dims = [784, 64,  1]
    LF_model = SF_KAN(layer_dims, boundaries, scales, grid_vals, init_lr, 0)
    
    
    with open('HF_' + str(N) + '.pkl', 'rb') as f:
        params_LF = pickle.load(f)
    

    
    preds_HF = LF_model.simple_out_fn(X_test, params_LF)
    
    
    scipy.io.savemat("Test7_HF_" + str(N) + "_train.mat", 
                     {'preds_HF':preds_HF[:, 0, 0]}
                     , format='4')
        