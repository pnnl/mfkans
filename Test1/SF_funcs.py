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

class SF_KAN:
    def __init__(self, layer_dims, boundaries, scales, grid_vals, init_lr, norm_weight=0): 
        
        self.modelLF = KAN(layer_dims=layer_dims, k=3, const_spl=False, const_res=False, add_bias=True, grid_e=0.02, j='0')
        
        lr_scales = dict(zip(boundaries, scales))
        self.grid_upds = dict(zip(boundaries, grid_vals))

        schedule = optax.piecewise_constant_schedule(
            init_value=init_lr,
            boundaries_and_scales=lr_scales
        )
        
        key = jax.random.PRNGKey(10)
        self.variables = self.modelLF.init(key, jnp.ones([1, 1]))
        self.variable_params = self.variables['params']

        self.optimizer = optax.adam(learning_rate=schedule, nesterov=True)
        self.opt_state = self.optimizer.init(self.variable_params)

        self.train_losses =[]
        
        self.norm_weight = norm_weight

    def get_params(self):
        return self.variables
        
    def interpolate_moments(self, mu_old, nu_old, new_shape):
        old_shape = mu_old.shape
        size = old_shape[0]
        old_j = old_shape[1]
        new_j = new_shape[1]
        
        # Create new indices for the second dimension
        old_indices = jnp.linspace(0, old_j - 1, old_j)
        new_indices = jnp.linspace(0, old_j - 1, new_j)
    
        # Vectorize the interpolation function for use with vmap
        interpolate_fn = lambda old_row: jnp.interp(new_indices, old_indices, old_row)
    
        # Apply the interpolation function to each row using vmap
        mu_new = vmap(interpolate_fn)(mu_old)
        nu_new = vmap(interpolate_fn)(nu_old)
        
        return mu_new, nu_new
    
    @partial(jit, static_argnums=(0,))
    def smooth_state_transition(self, old_state, variables):
    
        # Copy old state
        adam_count = old_state[0].count
        adam_mu, adam_nu = old_state[0].mu, old_state[0].nu
        
    
        # Get all layer-related keys, so that we do not touch the other parameters
        
        layer_keys = {k for k in adam_mu.keys() if k.startswith('layers_')}
            
        for key in layer_keys:
            # Find the c_basis shape for this layer
            c_shape = variables['params'][key]['c_basis'].shape
            # Get new mu and nu
            mu_new0, nu_new0 = self.interpolate_moments(adam_mu[key]['c_basis'], adam_nu[key]['c_basis'], c_shape)
            # Set them
            adam_mu[key]['c_basis'], adam_nu[key]['c_basis'] = mu_new0, nu_new0
    
        # Make new adam state
        adam_state = optax.ScaleByAdamState(adam_count, adam_mu, adam_nu)
        # Make new scheduler state if using scheduling, otherwise an empty state
        extra_state = optax.ScaleByScheduleState(adam_count)
        # Make new total state
        new_state = (adam_state, extra_state)
    
        return new_state

    
    def op_net(self, params, xLF, state):
        param_cur = params
        state_cur = state
        variables1 = {'params' : param_cur, 'state' : state_cur}
        preds1, spl_regs1 = self.modelLF.apply(variables1, xLF)
        return preds1, spl_regs1
    
    def simple_out_fn(self, xLF, varis ):
        params = varis['params'] 
        state = varis['state']
        
        preds_HF, _ = self.op_net(params, xLF,  state)
        # Define the prediction loss
        return preds_HF

    def simple_loss_fn(self, params, xLF, yLF, state):
        predsLF, spl_regs  = self.op_net(params, xLF, state)
        loss_pred = (jnp.mean((predsLF.flatten()-yLF.flatten())**2))
        
        # Calculate the regularization loss
        loss_reg = 0.0
        for spl_reg in spl_regs:
            # L1 regularization loss
            phis = spl_reg.reshape(-1)
            L1 = jnp.sum(phis)

            mu_1 = 1.0
            loss_reg += (mu_1 * L1)# + (mu_2 * Entropy)
            
        loss = loss_pred + self.norm_weight*loss_reg
        return loss
    
    @partial(jit, static_argnums=(0,))
    def train_step(self,  params, states, opt_state, xLF, yLF):

        loss, grads = jax.value_and_grad(self.simple_loss_fn)(params,  xLF, yLF,  states)

        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
    
        new_variables = {'params': params, 'state': states}
        return new_variables, opt_state, loss


    
    def train(self, num_epochs, t_data, s_data):
        pbar = trange(num_epochs)

        for epoch in pbar:
            # Check if we're in an update epoch
            if epoch in self.grid_upds.keys():
                print(f"Epoch {epoch+1}: Performing grid update")
                # Get grid size
                G_new = self.grid_upds[epoch]

                # Perform the update
                updated_variables1 = self.modelLF.apply(self.variables, t_data, G_new, 
                                                         method=self.modelLF.update_grids)
                self.variables = updated_variables1.copy()
                self.opt_state = self.smooth_state_transition(self.opt_state, self.variables)
                
            # Calculate the loss
            params = self.variables['params'] 
            states = self.variables['state']
            
            self.variables, self.opt_state, loss= self.train_step(params, states, self.opt_state, t_data, s_data)
            

            if epoch % 100 == 0:
                pbar.set_postfix({'Loss': "{0:.6f}".format(loss)})
                self.train_losses.append(loss)
            elapsed = pbar.format_dict['elapsed']
        return elapsed

    
