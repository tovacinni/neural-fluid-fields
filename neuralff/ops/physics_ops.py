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

def navier_stokes_loss(coords, velocity_field, pressure_field, timestep, rho=1.0, nu=1.0):
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

    Returns:
        (torch.Tensor) : per-point loss of the Navier-Stokes equation of shape [N, 1]
    """
    u = velocity_field.sample(coords)

    div_u = divergence(coords, velocity_field)

    dudt = (u - u.detach()) / timestep

    grad_p = (1.0/rho) * gradient(coords, pressure_field)
    
    g = torch.zeros_like(coords)
    g[...,1] = -9.8 * timestep

    diff = nu * laplacian(coords, velocity_field)

    return g + diff - dudt - div_u * u - grad_p + div_u




