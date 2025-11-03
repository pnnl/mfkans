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

from SF_funcs_MLP import *
from MF_funcs_MLP import *

from dataset_test1a import *


t_data_LF, s_data_LF = create_dataset_LF()
t_data_test, s_data_test = create_dataset_test()
t_data_HF, s_data_HF = create_dataset_HF()
X_test = t_data_test


    

# Training epochsy
num_epochs_LF = 60001
num_epochs_HF = 60001

# Epochs at which to change the grid & learning rate
boundaries = [0]
# Learning rate scales
scales = [1.0]
# Grid sizes to use
grid_vals = [6]
# Initial learning rate
init_lr = 0.001

# Corresponding dicts
lr_scales = dict(zip(boundaries, scales))
grid_upds = dict(zip(boundaries, grid_vals))

# Create a piecewise constant schedule
layer_dims = [1, 30, 30, 30,  1]
layer_dims = [1, 12, 12,  1]
LF_model = SF_KAN(layer_dims, boundaries, scales, grid_vals, init_lr, 0)
LF_model.train(num_epochs_LF, t_data_LF, s_data_LF)
train_losses_LF = LF_model.train_losses

params_LF = LF_model.get_params(LF_model.opt_state)


N_LF = 0
for i in np.arange(len(layer_dims)-1):
    N_LF += layer_dims[i]*layer_dims[i+1]
    N_LF += layer_dims[i+1]



preds_LF = LF_model.simple_out_fn(X_test, params_LF)


# Epochs at which to change the grid & learning rate
boundaries = [0]
# Learning rate scales
scales = [1.0]
# Grid sizes to use
grid_vals = [5]
# Initial learning rate
init_lr = 0.005

layer_dims_nl = [2,5,5, 1]
layer_dims_l = [2, 1]

N_HFnl = 0
for i in np.arange(len(layer_dims_nl)-1):
     N_HFnl +=layer_dims_nl[i]*layer_dims_nl[i+1]
     N_HFnl += layer_dims_nl[i+1]

N_HFl = 0
for i in np.arange(len(layer_dims_l)-1):
     N_HFl += layer_dims_l[i]*layer_dims_l[i+1]
     N_HFl += layer_dims_l[i+1]




HF_model = MF_KAN(layer_dims_nl, layer_dims_l, LF_model, params_LF, boundaries, scales, 
                  grid_vals, init_lr, 1)
HF_model.train(num_epochs_HF, t_data_HF, s_data_HF)
train_losses_HF = HF_model.train_losses
params_HF = HF_model.get_params(HF_model.opt_state)

preds_HF = HF_model.simple_out_fn(X_test, params_HF)
train_losses = HF_model.train_losses
eval_losses_HF = HF_model.eval_losses_HF

preds_nl, preds_lin, alpha, preds_LF_on_HF = HF_model.both_out(X_test, params_HF)


scipy.io.savemat("Test2_w1_MLP.mat", 
                 {'X_train_HF':t_data_HF,
                 'y_train_HF':s_data_HF,
                 'X_train_LF':t_data_LF,
                 'y_train_LF':s_data_LF,
                 'X_test':X_test,
                 'preds_LF':preds_LF,
                 'preds_HF':preds_HF,
                 'train_losses':train_losses,
                 'train_losses_LF':train_losses_LF,
                 'eval_losses_HF':eval_losses_HF}
                 , format='4')
    



# Epochs at which to change the grid & learning rate
boundaries = [0]
# Learning rate scales
scales = [1.0]
# Grid sizes to use
grid_vals = [6]
# Initial learning rate
init_lr = 0.001

# Corresponding dicts
lr_scales = dict(zip(boundaries, scales))
grid_upds = dict(zip(boundaries, grid_vals))

# Create a piecewise constant schedule
layer_dims = [1, 30, 30,  1]
num_epochs_LF = 50001

LF_model = SF_KAN(layer_dims, boundaries, scales, grid_vals, init_lr, 0)
LF_model.train(num_epochs_LF, t_data_HF, s_data_HF)
train_losses = LF_model.train_losses
params_LF = LF_model.get_params(LF_model.opt_state)

preds_HF = LF_model.simple_out_fn(X_test, params_LF)


scipy.io.savemat("Test2_HF_MLP.mat", 
                 {'X_train_HF':t_data_HF,
                 'y_train_HF':s_data_HF,
                 'X_test':X_test,
                 'preds_HF':preds_HF,
                 'train_losses':train_losses}
                 , format='4')
    

