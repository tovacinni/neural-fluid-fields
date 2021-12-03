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
from .grid_ops import sample_from_grid

def semi_lagrangian_backtrace(coords, grid, velocity_field, timestep):
    """Perform semi-Lagrangian backtracing at continuous coordinates.

    Warning: PyTorch follows rather stupid conventions, so the flow field is [w, h] but the grid is [h, w]

    Args:
        coords (torch.FloatTensor): continuous coordinates in normalized coords [-1, 1] of size [N, 2]
        grid (torch.FloatTensor): grid of features of size [H, W, F]
        velocity_field (neuralff.Field): Eulerian vector field. This can be any Field representation.
        timestep (float) : timestep of the simulation

    Returns
        (torch.FloatTensor): backtracked values of [N, F]
    """

    # Velocities aren't necessarily on the same grid as the coords
    velocities_at_coords = velocity_field.sample(coords)
    
    # Equation 3.22 from Doyub's book
    samples = sample_from_grid(coords - timestep * velocities_at_coords, grid)
    return samples

def semi_lagrangian_advection(coords, grid, velocity_field, timestep):
    """Performs advection and updates the grid.
    
    This method is similar to the `semi_lagrangian_backtrace` function, but assumes the `coords` are 
    perfectly aligned with the `grid`. 

    Args:
        coords (torch.FloatTensor): continuous coordinates in normalized coords [-1, 1] of size [H, W, 2]
        grid (torch.FloatTensor): grid of features of size [H, W, F]
        velocity_field (neuralff.Field): Eulerian vector field. This can be any Field representation.
        timestep (float) : timestep of the simulation

    Returns
        (torch.FloatTensor): advaceted grid of [H, W, F]
    """
    H, W = coords.shape[:2]
    samples = semi_lagrangian_backtrace(coords.reshape(-1, 2), grid, velocity_field, timestep)
    return samples.reshape(H, W, -1)


