#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 22:10:58 2024

@author: howa549
"""

from phi.jax.flow import *
import pickle

cylinder = geom.infinite_cylinder(x=20, y=50, radius=10, inf_dim='z')
plot({"Top view": cylinder['x,y'], "Side view": cylinder['x,z']})


@jit_compile
def step(v, p, dt=1.):
    v = advect.semi_lagrangian(v, v, dt)
    return fluid.make_incompressible(v, cylinder, Solve(x0=p))

boundary = {'x-': vec(x=2, y=0, z=0), 'x+': ZERO_GRADIENT, 'y': PERIODIC, 'z': PERIODIC}
v0 = StaggeredGrid((8., 0, 0), boundary, x=256, y=128, z=8, bounds=Box(x=200, y=100, z=5))
v_trj, p_trj = iterate(step, batch(time=200), v0, None)


v_trj_2d = v_trj[{'z': 4, 'vector': 'x,y'}]
plot(v_trj_2d.time[0:].curl(), animate='time')


a = v_trj_2d.time[0:].curl()
b = a.numpy(order='time, x, y')

with open('HF_data.pkl', 'wb') as f:
    pickle.dump(b, f)