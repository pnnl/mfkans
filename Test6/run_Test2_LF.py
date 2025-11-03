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

#import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import scipy.io

from SF_funcs_only import *
from dataset_test1a import *
import pickle


t_data_LF, s_data_LF = create_dataset_LF(Test=False, N=60000)


# Training epochsy
num_epochs_LF = 30001

# Epochs at which to change the grid & learning rate
boundaries = [0]
# Learning rate scales
scales = [1.0]
# Grid sizes to use
grid_vals = [5]
# Initial learning rate
init_lr = 0.01

# Corresponding dicts
lr_scales = dict(zip(boundaries, scales))
grid_upds = dict(zip(boundaries, grid_vals))

# Create a piecewise constant schedule
layer_dims = [784, 64,  1]
LF_model = SF_KAN(layer_dims, boundaries, scales, grid_vals, init_lr, 0)
LF_model.train(num_epochs_LF, t_data_LF, s_data_LF)
train_losses = LF_model.train_losses
params_LF = LF_model.get_params()

with open('LF.pkl', 'wb') as f:
    pickle.dump(params_LF, f)

#t_data_LF_test, s_data_LF_test = create_dataset_test(LF=True, N=10000)
#t_data_test, s_data_test = create_dataset_test(LF=False, N=10000)
#X_test = t_data_test


#preds_HF = LF_model.simple_out_fn(X_test, params_LF)
#preds_LF = LF_model.simple_out_fn(t_data_LF_test, params_LF)


scipy.io.savemat("Test7_LF_loss.mat", 
                 {
                 'train_losses':train_losses}
                 , format='4')
    
