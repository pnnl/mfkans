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

from tqdm import trange, tqdm
import scipy.io

from SF_funcs_MLP import *
from dataset_test5 import *



for k_HF in [4, 12]:
    
    t_data_test, s_data_test, _, f_test, _= create_dataset_test(2, k_HF)
    X_test = t_data_test
    t_data_HF, s_data_HF, f_HF = create_dataset_HF(k_HF, 0)
    t_data_bc = jnp.asarray([0, 1]).reshape(-1, 1)
    s_data_bc = jnp.asarray([0, 0]).reshape(-1, 1)
    
    
    
    # Training epochsy
    num_epochs_HF = 60000
    

    init_lr = 0.0005
    

    
    # Create a piecewise constant schedule
    
    layer_dims = [1, 100, 100, 1]
#    layer_dims = [1, 40, 40, 1]


    LF_model = SF_KAN(layer_dims,  init_lr, 0)
    LF_model.train(num_epochs_HF, t_data_HF, f_HF, t_data_bc, s_data_bc)
    train_losses = LF_model.train_losses
    params_LF = LF_model.get_params(LF_model.opt_state)
    
    preds_HF = LF_model.simple_out_fn(X_test, params_LF)
    
    
    scipy.io.savemat("Test5_HF_" + str(k_HF) + "_MLP_big.mat", 
                     {'X_train_HF':t_data_HF,
                     'y_train_HF':s_data_HF,
                     'X_test':X_test,
                     'preds_HF':preds_HF,
                     'Y_test':s_data_test,
                     'f_test':f_test,
                     'train_losses':train_losses}
                     , format='4')
        

    layer_dims = [1, 40, 40, 1]


    LF_model = SF_KAN(layer_dims,  init_lr, 0)
    LF_model.train(num_epochs_HF, t_data_HF, f_HF, t_data_bc, s_data_bc)
    train_losses = LF_model.train_losses
    params_LF = LF_model.get_params(LF_model.opt_state)
    
    preds_HF = LF_model.simple_out_fn(X_test, params_LF)
    
    
    scipy.io.savemat("Test5_HF_" + str(k_HF) + "_MLP_small.mat", 
                     {'X_train_HF':t_data_HF,
                     'y_train_HF':s_data_HF,
                     'X_test':X_test,
                     'preds_HF':preds_HF,
                     'Y_test':s_data_test,
                     'f_test':f_test,
                     'train_losses':train_losses}
                     , format='4')
