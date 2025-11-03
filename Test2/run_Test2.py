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


t_data_LF, s_data_LF = create_dataset_LF()
t_data_test, s_data_test = create_dataset_test()
t_data_HF, s_data_HF = create_dataset_HF()
X_test = t_data_test


    

# Training epochsy
num_epochs_LF = 40001
num_epochs_HF = 100001

# Epochs at which to change the grid & learning rate
boundaries = [0, 15000]
# Learning rate scales
scales = [1.0, .4]
# Grid sizes to use
grid_vals = [6, 12]
# Initial learning rate
init_lr = 0.001

# Corresponding dicts
lr_scales = dict(zip(boundaries, scales))
grid_upds = dict(zip(boundaries, grid_vals))

# Create a piecewise constant schedule
layer_dims = [1, 5,  1]
LF_model = SF_KAN(layer_dims, boundaries, scales, grid_vals, init_lr, 0)
LF_model.train(num_epochs_LF, t_data_LF, s_data_LF)
train_losses_LF = LF_model.train_losses
params_LF = LF_model.get_params()

preds_LF = LF_model.simple_out_fn(X_test, params_LF)


num_params_LF = len(params_LF['params']['bias_0']) + len(params_LF['params']['bias_1']) + \
    len(params_LF['params']['layers_0']['c_basis'].flatten()) + len(params_LF['params']['layers_1']['c_basis'].flatten())\
    + len(params_LF['params']['layers_0']['c_res'].flatten()) + len(params_LF['params']['layers_0']['c_spl'].flatten())\
    + len(params_LF['params']['layers_1']['c_res'].flatten()) +len(params_LF['params']['layers_1']['c_spl'].flatten())




# Epochs at which to change the grid & learning rate
boundaries = [0, 25000]
# Learning rate scales
scales = [1.0, .7]
# Grid sizes to use
grid_vals = [5, 8]
# Initial learning rate
init_lr = 0.005

layer_dims_nl = [2, 8,  1]
layer_dims_l = [2, 1]





HF_model = MF_KAN(layer_dims_nl, layer_dims_l, LF_model, params_LF, boundaries, scales, 
                  grid_vals, init_lr, 1)
HF_model.train(num_epochs_HF, t_data_HF, s_data_HF)
train_losses_HF = HF_model.train_losses
params_HF = HF_model.get_params()

num_params_HF_nl = len(params_HF[0]['params']['bias_0']) + len(params_HF[0]['params']['bias_1']) + \
    len(params_HF[0]['params']['layers_0']['c_basis'].flatten()) + len(params_HF[0]['params']['layers_1']['c_basis'].flatten())\
    + len(params_HF[0]['params']['layers_0']['c_res'].flatten()) \
    + len(params_HF[0]['params']['layers_1']['c_res'].flatten()) 

num_params_HF_l = len(params_HF[1]['params']['layers_0']['c_basis'].flatten()) 




preds_HF = HF_model.simple_out_fn(X_test, params_HF)
train_losses = HF_model.train_losses
eval_losses_HF = HF_model.eval_losses_HF

preds_nl, preds_lin, alpha, preds_LF_on_HF = HF_model.both_out(X_test, params_HF)


scipy.io.savemat("Test2_w1.mat", 
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
    

HF_model = MF_KAN(layer_dims_nl, layer_dims_l, LF_model, params_LF, boundaries, scales, 
                  grid_vals, init_lr, 0)
HF_model.train(num_epochs_HF, t_data_HF, s_data_HF)
train_losses_HF = HF_model.train_losses
params_HF = HF_model.get_params()

preds_HF = HF_model.simple_out_fn(X_test, params_HF)
train_losses = HF_model.train_losses
eval_losses_HF = HF_model.eval_losses_HF

preds_nl, preds_lin, alpha, preds_LF_on_HF = HF_model.both_out(X_test, params_HF)


scipy.io.savemat("Test2_w0.mat", 
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
layer_dims = [1, 5,  1]
num_epochs_LF = 50001

LF_model = SF_KAN(layer_dims, boundaries, scales, grid_vals, init_lr, 0)
LF_model.train(num_epochs_LF, t_data_HF, s_data_HF)
train_losses = LF_model.train_losses
params_LF = LF_model.get_params()

preds_HF = LF_model.simple_out_fn(X_test, params_LF)


scipy.io.savemat("Test2_HF.mat", 
                 {'X_train_HF':t_data_HF,
                 'y_train_HF':s_data_HF,
                 'X_test':X_test,
                 'preds_HF':preds_HF,
                 'train_losses':train_losses}
                 , format='4')
    

