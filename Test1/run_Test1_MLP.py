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


def yL(x):
    if x < 0.5:
        y = 0.5*(6*x-2)**2 * jnp.sin(12*x-4) + 10*(x-0.5)-5
        
    else:
        y = 3 + 0.5*(6*x-2)**2 * jnp.sin(12*x-4) + 10*(x-0.5)-5
    y = y/10
    return y 

def yH(x):
     y = 2*(yL(x))-2*x + 2
    
     return y

    

t_data_LF = jnp.linspace(0, 1, 50)
s_data_LF = np.zeros(50)
for i in np.arange(50):
    s_data_LF[i] = yL(t_data_LF[i])
t_data_HF = jnp.linspace(.1, .93, 5)
s_data_HF = np.zeros(5)
for i in np.arange(5):
    s_data_HF[i] = yH(t_data_HF[i])
t_data_LF = t_data_LF.reshape(-1, 1)
t_data_HF = t_data_HF.reshape(-1, 1)
s_data_HF = s_data_HF.reshape(-1, 1)
s_data_LF = s_data_LF.reshape(-1, 1)
X_test = np.linspace(0, 1, 200).reshape(-1, 1)


# Training epochsy
num_epochs_LF = 100001
num_epochs_HF = 50001

# Epochs at which to change the grid & learning rate
boundaries = [0]
# Learning rate scales
scales = [1.0]
# Grid sizes to use
grid_vals = [5]
# Initial learning rate
init_lr = 0.001
# Corresponding dicts
lr_scales = dict(zip(boundaries, scales))
grid_upds = dict(zip(boundaries, grid_vals))

# Create a piecewise constant schedule
layer_dims = [1, 6, 6,  1]
N_LF = 0
for i in np.arange(len(layer_dims)-1):
    N_LF += layer_dims[i]*layer_dims[i+1]
    N_LF += layer_dims[i+1]



LF_model = SF_KAN(layer_dims, boundaries, scales, grid_vals, init_lr, 0)
t = LF_model.train(num_epochs_LF, t_data_LF, s_data_LF)
train_losses_LF = LF_model.train_losses
params_LF = LF_model.get_params(LF_model.opt_state)




preds_LF = LF_model.simple_out_fn(X_test, params_LF)



# Epochs at which to change the grid & learning rate
boundaries = [0]
# Learning rate scales
scales = [1.0]
# Grid sizes to use
grid_vals = [3]
# Corresponding dicts
lr_scales = dict(zip(boundaries, scales))
grid_upds = dict(zip(boundaries, grid_vals))


layer_dims_nl = [2, 4, 4, 1]
layer_dims_l = [2,1]
init_lr = 0.001



HF_model = MF_KAN(layer_dims_nl, layer_dims_l, LF_model, params_LF, boundaries, scales, 
                  grid_vals, init_lr, 0)
HF_model.train(num_epochs_HF, t_data_HF, s_data_HF)
train_losses_HF = HF_model.train_losses
params_HF = HF_model.get_params(HF_model.opt_state)

N_HFnl = 0
for i in np.arange(len(layer_dims_nl)-1):
     N_HFnl +=layer_dims_nl[i]*layer_dims_nl[i+1]
     N_HFnl += layer_dims_nl[i+1]

N_HFl = 0
for i in np.arange(len(layer_dims_l)-1):
     N_HFl += layer_dims_l[i]*layer_dims_l[i+1]
     N_HFl += layer_dims_l[i+1]


preds_HF = HF_model.simple_out_fn(X_test, params_HF)
train_losses = HF_model.train_losses
eval_losses_HF = HF_model.eval_losses_HF

preds_nl, preds_lin, alpha, preds_LF_on_HF = HF_model.both_out(X_test, params_HF)


scipy.io.savemat("Test1_w0_MLP.mat", 
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


# Training epochsy
num_epochs_LF = 100001
# Epochs at which to change the grid & learning rate
boundarieslr = [0]
boundaries = [0]
# Learning rate scales
scales = [1.0]
# Grid sizes to use
grid_vals = [2]
# Initial learning rate
init_lr = 0.001

# Corresponding dicts
lr_scales = dict(zip(boundarieslr, scales))
grid_upds = dict(zip(boundaries, grid_vals))

# Create a piecewise constant schedule
layer_dims = [1, 20, 20, 20, 20,  1]
LF_model = SF_KAN(layer_dims, boundaries, scales, grid_vals, init_lr, 0)
LF_model.train(num_epochs_LF, t_data_HF, s_data_HF)
train_losses = LF_model.train_losses
params_LF = LF_model.get_params(LF_model.opt_state)

preds_HF = LF_model.simple_out_fn(X_test, params_LF)


scipy.io.savemat("Test1_HF_MLP.mat", 
                 {'X_train_HF':t_data_HF,
                 'y_train_HF':s_data_HF,
                 'X_test':X_test,
                 'preds_HF':preds_HF,
                 'train_losses':train_losses}
                 , format='4')






#### Train with w=300

t_data_LF = jnp.linspace(0, 1, 300)
s_data_LF = np.zeros(300)
for i in np.arange(300):
    s_data_LF[i] = yL(t_data_LF[i])
t_data_LF = t_data_LF.reshape(-1, 1)
s_data_LF = s_data_LF.reshape(-1, 1)

# Training epochsy
num_epochs_LF = 100001
num_epochs_HF = 50001

# Epochs at which to change the grid & learning rate
boundaries = [0]
# Learning rate scales
scales = [1.0]
# Grid sizes to use
grid_vals = [5]
# Initial learning rate
init_lr = 0.001
# Corresponding dicts
lr_scales = dict(zip(boundaries, scales))
grid_upds = dict(zip(boundaries, grid_vals))

# Create a piecewise constant schedule
layer_dims = [1, 6, 6,  1]
LF_model = SF_KAN(layer_dims, boundaries, scales, grid_vals, init_lr, 0)
LF_model.train(num_epochs_LF, t_data_LF, s_data_LF)
train_losses_LF = LF_model.train_losses
params_LF = LF_model.get_params(LF_model.opt_state)

preds_LF = LF_model.simple_out_fn(X_test, params_LF)



# Epochs at which to change the grid & learning rate
boundaries = [0]
# Learning rate scales
scales = [1.0]
# Grid sizes to use
grid_vals = [3]
# Corresponding dicts
lr_scales = dict(zip(boundaries, scales))
grid_upds = dict(zip(boundaries, grid_vals))

init_lr = 0.001


HF_model = MF_KAN(layer_dims_nl, layer_dims_l, LF_model, params_LF, boundaries, scales, 
                  grid_vals, init_lr, 0)
HF_model.train(num_epochs_HF, t_data_HF, s_data_HF)
train_losses_HF = HF_model.train_losses
params_HF = HF_model.get_params(HF_model.opt_state)

preds_HF = HF_model.simple_out_fn(X_test, params_HF)
train_losses = HF_model.train_losses
eval_losses_HF = HF_model.eval_losses_HF

preds_nl, preds_lin, alpha, preds_LF_on_HF = HF_model.both_out(X_test, params_HF)


scipy.io.savemat("Test1_w0_300_MLP.mat", 
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


