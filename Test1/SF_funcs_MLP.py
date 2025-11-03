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
from functools import partial

import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import scipy.io
from jax.nn import relu, elu, selu, swish
import itertools
from jax.example_libraries import optimizers


def DNN(branch_layers, activation=jnp.tanh):

    def xavier_init_j(key, d_in, d_out):
        glorot_stddev = 1. / np.sqrt((d_in + d_out) / 2.)
        W = glorot_stddev * random.normal(key, (d_in, d_out))
        b = np.zeros(d_out)
        return W, b
    def init(rng_key1):
        def init_layer(key, d_in, d_out):
            k1, k2 = random.split(key)
            W, b = xavier_init_j(k1, d_in, d_out)
            return W, b
        key1, *keys1 = random.split(rng_key1, len(branch_layers))
        branch_params = list(map(init_layer, keys1, branch_layers[:-1], branch_layers[1:]))
        return branch_params
        
    def apply(params, u):
      #  print(u.shape)
        for k in range(len(branch_layers)-2):
            W_b, b_b = params[k]
            
            u = activation(jnp.dot(u, W_b) + b_b)

        W_b, b_b = params[-1]
        u = jnp.dot(u, W_b) + b_b
      #  print(u.shape)

        return u

    return init, apply



class SF_KAN:
    def __init__(self, layer_dims, boundaries, scales, grid_vals, init_lr, norm_weight=0): 
        
        self.init_low, self.apply_low = DNN(layer_dims)
        params = self.init_low(random.PRNGKey(12))
        self.itercount = itertools.count()

        

        key = jax.random.PRNGKey(10)
        self.variable_params = params


        self.train_losses =[]
        
        self.norm_weight = norm_weight

        self.opt_init, \
        self.opt_update, \
        self.get_params = optimizers.adam(init_lr)
 
        self.opt_state = self.opt_init(params)

    def op_net(self, params, xLF):
        preds1 = self.apply_low(params, xLF)
        return preds1
    
    def simple_out_fn(self, xLF, params ):
        
        preds_HF = self.op_net(params, xLF)
        # Define the prediction loss
        return preds_HF

    def simple_loss_fn(self, params, xLF, yLF):
        predsLF  = self.op_net(params, xLF)
        loss_pred = (jnp.mean((predsLF.flatten()-yLF.flatten())**2))
        
        loss = loss_pred 
        return loss
    
    @partial(jit, static_argnums=(0,))
    def train_step(self,  i, opt_state, xLF, yLF):
        params = self.get_params(opt_state)

       # loss, grads = jax.value_and_grad(self.simple_loss_fn)(params,  xLF, yLF,  states)

        g = grad(self.simple_loss_fn)(params, xLF, yLF)
        return self.opt_update(i, g, opt_state)


    
    def train(self, num_epochs, t_data, s_data):
        pbar = trange(num_epochs)

        for epoch in pbar:
            self.opt_state = self.train_step(next(self.itercount), self.opt_state, 
                                      t_data, s_data)

          # self.variables, self.opt_state, loss= self.train_step(params, states, self.opt_state, t_data, s_data)
            

            if epoch % 100 == 0:
                params = self.get_params(self.opt_state)

                loss_value = self.simple_loss_fn(params, t_data, s_data)

                pbar.set_postfix({'Loss': "{0:.6f}".format(loss_value)})
                self.train_losses.append(loss_value)
            elapsed = pbar.format_dict['elapsed']
        return elapsed

    
