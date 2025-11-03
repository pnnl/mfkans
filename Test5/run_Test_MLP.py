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
from MF_funcs_MLP import *

from dataset_test5 import *
import pickle 


for k_LF in [4]:
    for k_HF in [12]:
        t_data_test, s_data_test, s_data_test_LF, f_test, f_test_LF = create_dataset_test(k_LF, k_HF)
        X_test = t_data_test
        t_data_HF, s_data_HF, f_HF = create_dataset_HF(k_HF, 0)
        t_data_LF, s_data_LF, f_LF = create_dataset_LF(k_LF, 0)
        t_data_bc = jnp.asarray([0, 1]).reshape(-1, 1)
        s_data_bc = jnp.asarray([0, 0]).reshape(-1, 1)
        
            
        
        # Training epochsy
        num_epochs_HF = 60000
        
        num_epochs_LF = 60000
        

        init_lr = 0.0005

        # Create a piecewise constant schedule
        layer_dims =  [1, 40, 40, 1]
#        layer_dims = [1,20, 20, 1]
        LF_model = SF_KAN(layer_dims, init_lr, 0)
        
        reload = False
        
        if reload:
            with open('LF_' + str(k_LF) + '.pkl', 'rb') as f:
                params_LF = pickle.load(f)
            train_losses_LF  = [0]
        else:
            LF_model.train(num_epochs_LF, t_data_LF, f_LF, t_data_bc, s_data_bc)
            train_losses_LF = LF_model.train_losses
            params_LF = LF_model.get_params(LF_model.opt_state)
            with open('LF_' + str(k_LF) + '.pkl', 'wb') as f:
                pickle.dump(params_LF, f)
            
        preds_LF = LF_model.simple_out_fn(X_test, params_LF)
        

        init_lr = 0.01
        

        layer_dims_nl = [2, 50, 50,  1]
#        layer_dims_nl = [2, 30, 30,  1]
        layer_dims_l = [2, 1]
        layer_dims_l = [2, 1]
        
        HF_model = MF_KAN(layer_dims_nl, layer_dims_l, LF_model, params_LF, init_lr, 0)
        HF_model.train(num_epochs_HF, t_data_HF, f_HF, t_data_bc, s_data_bc)
        train_losses_HF = HF_model.train_losses
        params_HF = HF_model.get_params((HF_model.opt_state))
        
        preds_HF = HF_model.simple_out_fn(X_test, params_HF)
        train_losses = HF_model.train_losses
        eval_losses_HF = HF_model.eval_losses_HF
        
        preds_nl, preds_lin, alpha, preds_LF_on_HF = HF_model.both_out(X_test, params_HF)
        
        
        scipy.io.savemat("Test5_" + str(k_HF) + "_MLP_big.mat", 
                         {'X_train_HF':t_data_HF,
                         'y_train_HF':s_data_HF,
                         'X_train_LF':t_data_LF,
                         'y_train_LF':s_data_LF,
                         'X_test':X_test,
                         'Y_test':s_data_test,
                         'Y_test_LF':s_data_test_LF,
                         'preds_LF':preds_LF,
                         'preds_HF':preds_HF,
                         'train_losses':train_losses,
                         'train_losses_LF':train_losses_LF,
                         'eval_losses_HF':eval_losses_HF}
                         , format='4')
            
#        layer_dims =  [1, 40, 40, 1]
        layer_dims = [1,20, 20, 1]
        LF_model = SF_KAN(layer_dims, init_lr, 0)
        
        reload = False
        
        if reload:
            with open('LF_' + str(k_LF) + '.pkl', 'rb') as f:
                params_LF = pickle.load(f)
            train_losses_LF  = [0]
        else:
            LF_model.train(num_epochs_LF, t_data_LF, f_LF, t_data_bc, s_data_bc)
            train_losses_LF = LF_model.train_losses
            params_LF = LF_model.get_params(LF_model.opt_state)
            with open('LF_' + str(k_LF) + '.pkl', 'wb') as f:
                pickle.dump(params_LF, f)
            
        preds_LF = LF_model.simple_out_fn(X_test, params_LF)
        

        init_lr = 0.01
        

#        layer_dims_nl = [2, 50, 50,  1]
        layer_dims_nl = [2, 30, 30,  1]
        layer_dims_l = [2, 1]
        layer_dims_l = [2, 1]
        
        HF_model = MF_KAN(layer_dims_nl, layer_dims_l, LF_model, params_LF, init_lr, 0)
        HF_model.train(num_epochs_HF, t_data_HF, f_HF, t_data_bc, s_data_bc)
        train_losses_HF = HF_model.train_losses
        params_HF = HF_model.get_params((HF_model.opt_state))
        
        preds_HF = HF_model.simple_out_fn(X_test, params_HF)
        train_losses = HF_model.train_losses
        eval_losses_HF = HF_model.eval_losses_HF
        
        preds_nl, preds_lin, alpha, preds_LF_on_HF = HF_model.both_out(X_test, params_HF)
        
        
        scipy.io.savemat("Test5_" + str(k_HF) + "_MLP_small.mat", 
                         {'X_train_HF':t_data_HF,
                         'y_train_HF':s_data_HF,
                         'X_train_LF':t_data_LF,
                         'y_train_LF':s_data_LF,
                         'X_test':X_test,
                         'Y_test':s_data_test,
                         'Y_test_LF':s_data_test_LF,
                         'preds_LF':preds_LF,
                         'preds_HF':preds_HF,
                         'train_losses':train_losses,
                         'train_losses_LF':train_losses_LF,
                         'eval_losses_HF':eval_losses_HF}
                         , format='4')
            
