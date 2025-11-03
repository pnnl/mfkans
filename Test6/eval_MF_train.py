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

from SF_funcs import *
from MF_funcs import *

from dataset_test1a import *

import pickle


t_data_test, s_data_test = create_dataset_HF(Test=False, N=10000)
# Training epochsy
num_epochs_HF = 60001

# Epochs at which to change the grid & learning rate
boundaries = [0]
# Learning rate scales
scales = [1.0]
# Grid sizes to use
grid_vals = [6]
# Initial learning rate
init_lr = 0.005
lr_scales = dict(zip(boundaries, scales))
grid_upds = dict(zip(boundaries, grid_vals))

# Create a piecewise constant schedule
layer_dims = [784,  64,  1]
LF_model = SF_KAN(layer_dims, boundaries, scales, grid_vals, init_lr, 0)

reload = True

if reload:
    with open('LF.pkl', 'rb') as f:
        params_LF = pickle.load(f)
    train_losses_LF  = [0]


# Epochs at which to change the grid & learning rate
boundaries = [0]
# Learning rate scales
scales = [1.0]
# Grid sizes to use
grid_vals = [5]
# Initial learning rate
init_lr = 0.1
lr_scales = dict(zip(boundaries, scales))
grid_upds = dict(zip(boundaries, grid_vals))

layer_dims_nl = [785,  64,  1]
layer_dims_l = [1,  1]

HF_model = MF_KAN(layer_dims_nl, layer_dims_l, LF_model, params_LF, boundaries, scales, 
                  grid_vals, init_lr, 0)


#t_data_LF, s_data_LF = create_dataset_LF(Test=False, N=10000)
for N in [200, 400, 1000, 5000, 10000, 20000, 60000]:
    print(N)

    with open('MF_64_' + str(N) + '.pkl', 'rb') as f:
        params_MF = pickle.load(f)

    preds_HF = HF_model.simple_out_fn(t_data_test, params_MF)
    
    
    scipy.io.savemat("Test7_64_" + str(N) + "_train.mat", 
                     {'preds_HF':preds_HF[:, 0]}
                     , format='4')
        