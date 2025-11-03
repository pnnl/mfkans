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
from dataset_test1e import *



#t_data_LF, s_data_LF = create_dataset_LF()
t_data_test, s_data_test, _ = create_dataset_test()
X_test = t_data_test
t_data_HF, s_data_HF = create_dataset_HF(noise_level=0.03)


# Training epochsy
num_epochs_LF = 50000

# Epochs at which to change the grid & learning rate

init_lr = 0.008

# Create a piecewise constant schedule
layer_dims = [4, 30, 30, 30, 1]
LF_model = SF_KAN(layer_dims,  init_lr, 0)
LF_model.train(num_epochs_LF, t_data_HF, s_data_HF)
train_losses = LF_model.train_losses
params_LF = LF_model.get_params(LF_model.opt_state)

preds_HF = LF_model.simple_out_fn(X_test, params_LF)


scipy.io.savemat("Test4_HF_noise_MLP.mat", 
                 {'X_train_HF':t_data_HF,
                 'y_train_HF':s_data_HF,
                 'X_test':X_test,
                 'preds_HF':preds_HF,
                 'train_losses':train_losses}
                 , format='4')




t_data_HF, s_data_HF = create_dataset_HF(noise_level=0.0)


LF_model = SF_KAN(layer_dims, init_lr, 0)
LF_model.train(num_epochs_LF, t_data_HF, s_data_HF)
train_losses = LF_model.train_losses
params_LF = LF_model.get_params(LF_model.opt_state)

preds_HF = LF_model.simple_out_fn(X_test, params_LF)


scipy.io.savemat("Test4_HF_MLP.mat", 
                 {'X_train_HF':t_data_HF,
                 'y_train_HF':s_data_HF,
                 'X_test':X_test,
                 'preds_HF':preds_HF,
                 'train_losses':train_losses}
                 , format='4')
    

    