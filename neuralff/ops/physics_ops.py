# The MIT License (MIT)
#
# Copyright (c) 2021
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import torch
from .vector_ops import divergence, laplacian, gradient

def navier_stokes_loss(
        coords, velocity_field, pressure_field, 
        timestep, rho=1e-2, nu=1e-5, eps=1e-3, initial_velocity=None):
    """Computes the Navier Stokes equation loss.

    We use the following definition of the Navier-Stokes equation (from Bridson):
    
        eq 1 : du/dt + (div u) u + (1/rho) grad p = g + nu (Lap u)
        eq 2 : div u = 0

    Args:
        coords (torch.Tensor) : coordinates to enforce Navier-Stokes of shape [N, D]
        velocity_field (neuralff.Field) : velocity field function (aka u)
        pressure_field (neuralff.Field) : pressure field function (aka p)
        timestep (float) : delta t of the simulation
        rho (float) : the fluid density (kg/m^d)
        nu (float) : kinematic viscosity
        eps (float) : the "grid spacing" for finite diff
        initial_velocity (torch.Tensor) : if provided, will use this instead of sampling the initial velocity.
                                          This is useful for preconditioning the velocity field.

    Returns:
        (torch.Tensor) : per-point loss of the Navier-Stokes equation of shape [N, 1]
    """
    u = velocity_field.sample(coords)

    div_u = divergence(coords, velocity_field, eps=eps)

    if initial_velocity is None:
        dudt = (u - u.detach()) / timestep
    else:
        dudt = (u - initial_velocity) / timestep

    grad_p = (1.0/rho) * gradient(coords, pressure_field, eps=eps)
    
    g = torch.zeros_like(coords)
    g[...,1] = 9.8 * timestep

    diff = nu * laplacian(coords, velocity_field, eps=eps)
    

    momentum_term = ((g + diff - dudt - div_u * u - grad_p)**2).sum(-1, keepdim=True)
    #momentum_term = ((g-dudt)**2).sum(-1, keepdim=True)
    divergence_term = (div_u**2).sum(-1, keepdim=True)
    return momentum_term + divergence_term



