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
from dataset_test5 import *



for k_HF in [4, 12]:
    
    t_data_test, s_data_test, _, f_test, _= create_dataset_test(2, k_HF)
    X_test = t_data_test
    t_data_HF, s_data_HF, f_HF = create_dataset_HF(k_HF, 0)
    t_data_bc = jnp.asarray([0, 1]).reshape(-1, 1)
    s_data_bc = jnp.asarray([0, 0]).reshape(-1, 1)
    
    
    
    # Training epochsy
    num_epochs_HF = 60000
    
    # Epochs at which to change the grid & learning rate
    boundaries = [0, 20000, 40000]
    # Learning rate scales
    scales = [1.0, .8, .8,]
    # Grid sizes to use
    grid_vals = [6, 12, 18]
    # Initial learning rate
    init_lr = 0.0005
    
    # Corresponding dicts
    lr_scales = dict(zip(boundaries, scales))
    grid_upds = dict(zip(boundaries, grid_vals))
    
    
    
    # Create a piecewise constant schedule
    layer_dims = [1, 8, 8, 1]
    LF_model = SF_KAN(layer_dims, boundaries, scales, grid_vals, init_lr, 0)
    LF_model.train(num_epochs_HF, t_data_HF, f_HF, t_data_bc, s_data_bc)
    train_losses = LF_model.train_losses
    params_LF = LF_model.get_params()
    
    preds_HF = LF_model.simple_out_fn(X_test, params_LF)
    
    
    scipy.io.savemat("Test5_HF_" + str(k_HF) + ".mat", 
                     {'X_train_HF':t_data_HF,
                     'y_train_HF':s_data_HF,
                     'X_test':X_test,
                     'preds_HF':preds_HF,
                     'Y_test':s_data_test,
                     'f_test':f_test,
                     'train_losses':train_losses}
                     , format='4')
        
