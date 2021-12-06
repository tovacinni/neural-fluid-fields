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

def time_derivative(
        coords, velocity_field, timestep, initial_velocity=None):
    """Computes the time derivative (du/dt).

    Args:
        coords (torch.Tensor) : coordinates of shape [N, D]
        velocity_field (neuralff.Field) : velocity field function (aka u)
        timestep (float) : delta t of the simulation
        initial_velocity (torch.Tensor) : if provided, will use this instead of sampling the initial velocity.
                                          This is useful for preconditioning the velocity field.

    Returns:
        (torch.Tensor, torch.Tensor) : 
            - time derivative of shape [N, D]
            - velocity of shape [N, D]
    """
    u = velocity_field.sample(coords)
    if initial_velocity is None:
        dudt = (u - u.detach()) / timestep
    else:
        dudt = (u - initial_velocity) / timestep
    return dudt, u

def material_derivative(
        coords, velocity_field, timestep, eps=1e-3, initial_velocity=None):
    """Computes the material derivative.

    We use the following definition of the material derivative (from Bridson):

        eq 1 : du/dt + (div u) u

    Args:
        coords (torch.Tensor) : coordinates of shape [N, D]
        velocity_field (neuralff.Field) : velocity field function (aka u)
        timestep (float) : delta t of the simulation
        eps (float) : the "grid spacing" for finite diff
        initial_velocity (torch.Tensor) : if provided, will use this instead of sampling the initial velocity.
                                          This is useful for preconditioning the velocity field.

    Returns:
        (torch.Tensor, torch.Tensor) : 
            - material derivative of shape [N, 2]
            - divergence of shape [N, 1]
    """
    dudt, u = time_derivative(coords, velocity_field, timestep, initial_velocity=initial_velocity)
    div_u = divergence(coords, velocity_field, eps=eps)
    return dudt + div_u * u, div_u

def gravity(coords, timestep):
    """Returns the gravity vector constraint.

    Args:
        coords (torch.Tensor) : coordinates of shape [N, D]
        timestep (float) : delta t of the simulation
    
    Returns:
        (torch.Tensor) : gravity tensor of shape [N, D]
    """
    g = torch.zeros_like(coords)
    g[..., 1] = 9.8 * 1e-4
    return g

def divergence_free_loss(coords, velocity_field, eps=1e-3):
    """Computes the divergence-free equation loss.

        eq 1 : div u = 0

    Args:
        coords (torch.Tensor) : coordinates of shape [N, D]
        velocity_field (neuralff.Field) : velocity field function (aka u)
        eps (float) : the "grid spacing" for finite diff
    
    Returns:
        (torch.Tensor) : per-point loss of the divergence-free term of shape [N, 1]
    """
    div_u = divergence(coords, velocity_field, eps=eps)
    return (div_u**2.0).sum(-1, keepdim=True)

def advection_loss(coords, velocity_field, timestep, eps=1e-3, initial_velocity=None):
    """Computes the advection equation loss.

    We use the following definition (from Bridson)

        eq 1 : du/dt + (div u) u = 0
    
    Args:
        coords (torch.Tensor) : coordinates of shape [N, D]
        velocity_field (neuralff.Field) : velocity field function (aka u)
        timestep (float) : delta t of the simulation
        eps (float) : the "grid spacing" for finite diff
        initial_velocity (torch.Tensor) : if provided, will use this instead of sampling the initial velocity.
                                          This is useful for preconditioning the velocity field.

    Returns:
        (torch.Tensor) : per-point loss of the advection equation of shape [N, 1]
    """
    mat_d, div_u = material_derivative(coords, velocity_field, timestep, eps=eps, initial_velocity=initial_velocity)
    return (mat_d**2.0).sum(-1, keepdim=True)

def body_forces_loss(coords, velocity_field, timestep, eps=1e-3, initial_velocity=None):
    """Computes the body-forces equation loss.

    We use the following definition (from Bridson)

        eq 1 : du/dt = g
    
    Args:
        coords (torch.Tensor) : coordinates of shape [N, D]
        velocity_field (neuralff.Field) : velocity field function (aka u)
        timestep (float) : delta t of the simulation
        eps (float) : the "grid spacing" for finite diff
        initial_velocity (torch.Tensor) : if provided, will use this instead of sampling the initial velocity.
                                          This is useful for preconditioning the velocity field.

    Returns:
        (torch.Tensor) : per-point loss of the body forces equation of shape [N, 1]
    """
    dudt, u = time_derivative(coords, velocity_field, timestep, initial_velocity=initial_velocity)
    g = gravity(coords, timestep)
    return ((g - dudt)**2.0).sum(-1, keepdim=True)

def incompressibility_loss(coords, velocity_field, pressure_field, rho_field, timestep, eps=1e-3, initial_velocity=None):
    """Computes the incompressibility equation loss.

    We use the following definition (from Bridson)

        eq 1 : du/dt = (1/rho) grad p
        eq 2 : div u = 0

    Args:
        coords (torch.Tensor) : coordinates of shape [N, D]
        pressure_field (neuralff.Field) : pressure field function (aka p)
        rho_field (neuralff.Field) : the fluid density field (kg/m^d)
        timestep (float) : delta t of the simulation
        eps (float) : the "grid spacing" for finite diff
        initial_velocity (torch.Tensor) : if provided, will use this instead of sampling the initial velocity.
                                          This is useful for preconditioning the velocity field.

    Returns:
        (torch.Tensor) : per-point loss of the incompressibility equation of shape [N, 1]
    """
    dudt, u = time_derivative(coords, velocity_field, timestep, initial_velocity=initial_velocity)
    div_u = divergence(coords, velocity_field, eps=eps)
    rho = rho_field.sample(coords)
    grad_p = (1.0/rho) * gradient(coords, pressure_field, eps=eps)
    return ((grad_p - dudt)**2.0).sum(-1, keepdim=True) + 100.0 * (div_u**2.0).sum(-1, keepdim=True)

def euler_loss(
        coords, velocity_field, pressure_field, rho_field,
        timestep, eps=1e-6, initial_velocity=None):
    """Computes the incompressible Euler equation loss.

    We use the following definition of the Euler equation (from Bridson):
    
        eq 1 : du/dt + (div u) u + (1/rho) grad p = g
        eq 2 : div u = 0

    That is, this is the Navier-Stokes equation without the viscosity term.

    Args:
        coords (torch.Tensor) : coordinates to enforce Navier-Stokes of shape [N, D]
        velocity_field (neuralff.Field) : velocity field function (aka u)
        pressure_field (neuralff.Field) : pressure field function (aka p)
        rho_field (neuralff.Field) : the fluid density field (kg/m^d)
        timestep (float) : delta t of the simulation
        eps (float) : the "grid spacing" for finite diff
        initial_velocity (torch.Tensor) : if provided, will use this instead of sampling the initial velocity.
                                          This is useful for preconditioning the velocity field.
    
    Returns:
        (torch.Tensor) : per-point loss of the Euler equation of shape [N, 1]
    """
    mat_d, div_u = material_derivative(coords, velocity_field, timestep, eps=eps, initial_velocity=initial_velocity)

    rho = rho_field.sample(coords) 
    #grad_p = (1.0/rho) * gradient(coords, pressure_field, eps=eps)
    grad_p = gradient(coords, pressure_field, eps=eps)
    
    g = gravity(coords, timestep)
    
    #momentum_term = ((g - mat_d - grad_p)**2).sum(-1, keepdim=True)
    momentum_term = ((rho * mat_d + grad_p - rho * g)**2).sum(-1, keepdim=True)
    divergence_term = (div_u**2).sum(-1, keepdim=True)
    return momentum_term + 100.0 * divergence_term

def navier_stokes_loss(
        coords, velocity_field, pressure_field, rho_field,
        timestep, nu=1e-5, eps=1e-3, initial_velocity=None):
    """Computes the Navier Stokes equation loss.

    We use the following definition of the Navier-Stokes equation (from Bridson):
    
        eq 1 : du/dt + (div u) u + (1/rho) grad p = g + nu (Lap u)
        eq 2 : div u = 0

    Args:
        coords (torch.Tensor) : coordinates to enforce Navier-Stokes of shape [N, D]
        velocity_field (neuralff.Field) : velocity field function (aka u)
        pressure_field (neuralff.Field) : pressure field function (aka p)
        rho_field (neuralff.Field) : the fluid density field (kg/m^d)
        timestep (float) : delta t of the simulation
        nu (float) : kinematic viscosity
        eps (float) : the "grid spacing" for finite diff
        initial_velocity (torch.Tensor) : if provided, will use this instead of sampling the initial velocity.
                                          This is useful for preconditioning the velocity field.

    Returns:
        (torch.Tensor) : per-point loss of the Navier-Stokes equation of shape [N, 1]
    """
    mat_d, div_u = material_derivative(coords, velocity_field, timestep, eps=eps, initial_velocity=initial_velocity)

    rho = rho_field.sample(coords)
    grad_p = (1.0/rho) * gradient(coords, pressure_field, eps=eps)
    
    g = gravity(coords, timestep)

    diff = nu * laplacian(coords, velocity_field, eps=eps)
    
    momentum_term = ((g + diff - mat_d - grad_p)**2).sum(-1, keepdim=True)
    divergence_term = (div_u**2).sum(-1, keepdim=True)
    return momentum_term + divergence_term

