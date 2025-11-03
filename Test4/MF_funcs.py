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

class MF_KAN:
    def __init__(self, layer_dims_nl, layer_dims_l, SF_model, SF_variables, boundaries, scales, 
                 grid_vals, init_lr, norm_weight): 
        
        self.modelHF_nl = KAN(layer_dims=layer_dims_nl, k=3, const_spl=1.0, const_res=False, add_bias=True, grid_e=0.02, j='0')
        self.modelHF_lin = KAN(layer_dims=layer_dims_l, k=1, const_spl=1.0, const_res=0.0, add_bias=False, grid_e=1.0, j='0')
        
        self.lr_scales = dict(zip(boundaries, scales))
        self.grid_upds = dict(zip(boundaries, grid_vals))

        schedule = optax.piecewise_constant_schedule(
            init_value=init_lr,
            boundaries_and_scales=self.lr_scales
        )
        
        key = jax.random.PRNGKey(10)

        variables2 = self.modelHF_nl.init(key, jnp.ones([1, 5]))
        variable_params = [variables2['params']]
        variables = [variables2]
        
        key, subkey = random.split(key)
        variables2 = self.modelHF_lin.init(key, jnp.ones([1, 5]))
        variable_params.append(variables2['params'])
        variables.append(variables2)

        alpha = np.abs(random.uniform(key, shape=[1]))
        alpha = jnp.asarray([.2])
        variables.append(alpha)
        variable_params.append(alpha)
        
        self.variables = variables
        self.SF_variables = SF_variables
        self.modelLF = SF_model


        
        self.optimizer = optax.adam(learning_rate=schedule, nesterov=True)
        self.opt_state = self.optimizer.init(variable_params)

        self.train_losses =[]
        self.eval_losses_HF = []
        
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
        
        for i in np.arange(len(variables)-1):
            layer_keys = {k for k in adam_mu[i].keys() if k.startswith('layers_')}
            
            for key in layer_keys:
                # Find the c_basis shape for this layer
                c_shape = variables[i]['params'][key]['c_basis'].shape
                # Get new mu and nu
                mu_new0, nu_new0 = self.interpolate_moments(adam_mu[i][key]['c_basis'], adam_nu[i][key]['c_basis'], c_shape)
                # Set them
                adam_mu[i][key]['c_basis'], adam_nu[i][key]['c_basis'] = mu_new0, nu_new0
    
        # Make new adam state
        adam_state = optax.ScaleByAdamState(adam_count, adam_mu, adam_nu)
        # Make new scheduler state if using scheduling, otherwise an empty state
        extra_state = optax.ScaleByScheduleState(adam_count)
        # Make new total state
        new_state = (adam_state, extra_state)
    
        return new_state

    
    def op_net(self, params, xHF, state):
        preds_LF_on_HF = self.modelLF.simple_out_fn(xHF, self.SF_variables)
    
        invals = jnp.hstack([xHF, preds_LF_on_HF])
    
        param_cur = params[0]
        state_cur = state[0]
        variables1 = {'params' : param_cur, 'state' : state_cur}
        preds_nonlin, spl_regs2 = self.modelHF_nl.apply(variables1, invals)
        
        
        param_cur = params[1]
        state_cur = state[1]
        variables1 = {'params' : param_cur, 'state' : state_cur}
        preds_lin, spl_regs3 = self.modelHF_lin.apply(variables1, invals)
        
        alpha = params[-1]

        preds2 = jnp.abs(alpha)*preds_nonlin+(1-jnp.abs(alpha))*preds_lin
    
        return preds2, spl_regs2, spl_regs3
        
    def simple_out_fn(self, xHF, varis):
        params = []
        states = []
        for i in np.arange(len(varis)-1):
            params.append(varis[i]['params'])
            states.append(varis[i]['state'])
        alpha = varis[-1]
        params.append(alpha)

        preds_HF, _, _ = self.op_net(params, xHF, states)

        return preds_HF
    
    def op_both(self, params, xHF, state):
        preds_LF_on_HF = self.modelLF.simple_out_fn(xHF, self.SF_variables)
    
        invals = jnp.hstack([xHF, preds_LF_on_HF])
    
        param_cur = params[0]
        state_cur = state[0]
        variables1 = {'params' : param_cur, 'state' : state_cur}
        preds_nonlin, spl_regs2 = self.modelHF_nl.apply(variables1, invals)
        
        
        param_cur = params[1]
        state_cur = state[1]
        variables1 = {'params' : param_cur, 'state' : state_cur}
        preds_lin, spl_regs3 = self.modelHF_lin.apply(variables1, invals)
        

    
        return preds_nonlin, preds_lin, preds_LF_on_HF
    
    
    def both_out(self, xHF, varis):
        params = []
        states = []
        for i in np.arange(len(varis)-1):
            params.append(varis[i]['params'])
            states.append(varis[i]['state'])
        alpha = varis[-1]
        params.append(alpha)

        preds_nonlin, preds_lin, preds_LF_on_HF = self.op_both(params, xHF, states)
        return preds_nonlin, preds_lin, alpha, preds_LF_on_HF




    def simple_loss_fn(self, params, xHF, yHF, state):
        predsHF, spl_regs2, spl_regs3 = self.op_net(params, xHF, state)
    
        alpha = params[-1]

        # Calculate the regularization loss
        loss_reg = 0.0
        for spl_reg in spl_regs2:
            # L1 regularization loss
            phis = spl_reg.reshape(-1)
            L1 = jnp.sum(phis)
            # Return full regularization loss
            mu_1 = 1.0
            loss_reg += (mu_1 * L1)# + (mu_2 * Entropy)
                    # Define the prediction loss

        loss_pred = (jnp.mean((predsHF.flatten()-yHF.flatten())**2)) + 10*alpha[0]**4 \
                    + self.norm_weight*loss_reg
        return loss_pred
    
    
    @partial(jit, static_argnums=(0,))
    def train_step(self, params, states, opt_state, xLF, yLF):

        loss, grads = jax.value_and_grad(self.simple_loss_fn)(params,  xLF, yLF,  states)
        updates, opt_state = self.optimizer.update(grads,opt_state, params)
        params = optax.apply_updates(params, updates)
        
        new_variables = []
        for i in np.arange(len(params)-1):
            new_variables.append({'params': params[i], 'state': states[i]})
        new_variables.append(params[-1])
            
        return new_variables,opt_state, loss


    
    def train(self, num_epochs, t_data, s_data):
        pbar = trange(num_epochs)
        key = jax.random.PRNGKey(10)

        for epoch in pbar:
            # Check if we're in an update epoch
            if epoch in self.grid_upds.keys():
                print(f"Epoch {epoch+1}: Performing grid update")
                G_new = self.grid_upds[epoch]
                preds_LF_on_HF = self.modelLF.simple_out_fn(t_data, self.SF_variables)

                invals = jnp.hstack([t_data, preds_LF_on_HF])
                variables = self.variables[0]
                updated_variables1 = self.modelHF_nl.apply(variables, invals, 3, 
                                                         method=self.modelHF_nl.update_grids)
                
                new_variables= [updated_variables1]
                variables = self.variables[1]

                invals = jnp.asarray([[0, 0, 0, 0, -2], [1, 1, 1, 1, 2]])


      #          invals = jnp.asarray([[0, 1],[0, 1],[0, 1],[0, 1], [-2, 2]])


                updated_variables1 = self.modelHF_lin.apply(variables, invals, 1, 
                                                         method=self.modelHF_lin.update_grids)
                new_variables.append(updated_variables1)
                
                alpha = self.variables[2]
                new_variables.append(alpha)

                self.variables = new_variables.copy()
                self.opt_state = self.smooth_state_transition(self.opt_state, self.variables)
                

            # Calculate the loss
            params = []
            states = []
            for i in np.arange(len(self.variables)-1):
                params.append(self.variables[i]['params'])
                states.append(self.variables[i]['state'])
            alpha = self.variables[-1]
            params.append(alpha)
            key, subkey = random.split(key)

            inds = random.choice(key, len(t_data), (100,), replace=False)

            self.variables, self.opt_state, loss= self.train_step(params, states, self.opt_state, 
                                                                  t_data[inds, :], s_data[inds])

    
            

            
            if epoch % 1000 == 0:

                preds_HF = self.simple_out_fn(t_data, self.variables)
                loss_HF = (jnp.mean((preds_HF.flatten() - s_data.flatten()) ** 2))

                self.eval_losses_HF.append(loss_HF)
                self.train_losses.append(loss)
                pbar.set_postfix({'Loss': "{0:.6f}".format(loss),'Test HF': "{0:.6f}".format(loss_HF),
                                  'alpha': "{0:.6f}".format(alpha[0])})

                elapsed = pbar.format_dict['elapsed']
        return elapsed

      #  print(alpha)
