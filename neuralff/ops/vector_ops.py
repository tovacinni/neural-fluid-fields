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

def gradient(u, f, method='finitediff', eps=1e-7):
    """Compute the gradient of the scalar field `f` with respect to the input `u`.

    Args:
        u (torch.Tensor) : the input to the function f of shape [..., D]
        f (function) : some scalar field (must support autodiff if using method='autodiff')
        method (str) : method for calculating the gradient. 
            options: ['autodiff', 'finitediff']

        Returns:
            (torch.Tensor) : gradient of shape [..., D]

        Finite diff currently assumes that `u` represents a 2D vector field (D=2).

    """
    if method == 'autodiff':
        with torch.enable_grad():
            u = u.requires_grad_(True)
            v = f(u)
            grad = torch.autograd.grad(v, u, 
                                       grad_outputs=torch.ones_like(v), create_graph=True)[0]
    elif method == 'finitediff':
        assert(u.shape[-1] == 2 and "Finitediff only supports 2D vector fields")
        eps_x = torch.tensor([eps, 0.0], device=u.device)
        eps_y = torch.tensor([0.0, eps], device=u.device)

        grad = torch.cat([f(u + eps_x) - f(u - eps_x),
                          f(u + eps_y) - f(u - eps_y)], dim=-1)
        grad = grad / (eps*2.0)
    else:
        raise NotImplementedError

    return grad

def jacobian(u, f, method='finitediff', eps=1e-7):
    """Compute the Jacobian of the vector field `f` with respect to the input `u`. 

    Args:
        u (torch.Tensor) : the input to the function f of shape [..., D]
        f (function) : some vector field (must support autodiff if using method='autodiff') with output dim [F]
        method (str) : method for calculating the Jacobian. 
            options: ['autodiff', 'finitediff']

        Returns:
            (torch.Tensor) : Jacobian of shape [..., F, D]

        Finite diff currently assumes that `u` represents a 2D vector field.
    """
    if method == 'autodiff':
        raise NotImplementedError
        # The behaviour here is a bit mysterious to me...
        with torch.enable_grad():
            j = torch.autograd.functional.jacobian(f, u,  create_graph=True)
    elif method == 'finitediff':
        assert(u.shape[-1] == 2 and "Finitediff only supports 2D vector fields")
        eps_x = torch.tensor([eps, 0.0], device=u.device)
        eps_y = torch.tensor([0.0, eps], device=u.device)
        
        dfux = (f(u + eps_x) - f(u - eps_x))[..., None]
        dfuy = (f(u + eps_y) - f(u - eps_y))[..., None]

        # Check that the dims are ordered correctly
        return torch.cat([dfux, dfuy], dim=-1) / (eps*2.0)
    else:
        raise NotImplementedError
    return j

def divergence(u, f, method='finitediff', eps=1e-7):
    """Compute the divergence of the vector field `f` with respect to the input `u`.

    Args:
        u (torch.Tensor) : the input to the function f of shape [..., D]
        f (function) : some vector field (must support autodiff if using method='autodiff') with output dim [D]
        method (str) : method for calculating the Jacobian. 
            options: ['autodiff', 'finitediff']

        Finite diff currently assumes that `u` represents a 2D vector field.

        Returns:
            (torch.Tensor) : divergence of shape [..., 1]
    """
    if method == 'autodiff':
        raise NotImplementedError
        j = jacobian(u, f, method=method, eps=eps)
        return j.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1) # Make sure it returns the correct diagonal
    if method == 'finitediff':
        eps_x = torch.tensor([eps, 0.0], device=u.device)
        eps_y = torch.tensor([0.0, eps], device=u.device)
        dfxux = f(u + eps_x)[...,0:1] - f(u - eps_x)[...,0:1] 
        dfyuy = f(u + eps_y)[...,1:2] - f(u - eps_y)[...,1:2]
        return (dfxux + dfyuy)/(eps*2.0)

def curl(u, f, method='finitediff', eps=1e-7):
    """Compute the curl of the vector field `f` with respect to the input `u`.

    Args:
        u (torch.Tensor) : the input to the function f of shape [..., D]
        f (function) : some vector field (must support autodiff if using method='autodiff') with output dim [D]
        method (str) : method for calculating the curl. 
            options: ['autodiff', 'finitediff']

        Finite diff currently assumes that `u` represents a 2D vector field.

        Returns:
            (torch.Tensor) : curl of shape [..., 1]
    """
    if method == 'autodiff':
        raise NotImplementedError
    if method == 'finitediff':
        eps_x = torch.tensor([eps, 0.0], device=u.device)
        eps_y = torch.tensor([0.0, eps], device=u.device)
        dfyux = f(u + eps_x)[...,1:2] - f(u - eps_x)[...,1:2] 
        dfxuy = f(u + eps_y)[...,0:1] - f(u - eps_y)[...,0:1]
        return (dfyux+dfxuy)/(eps*2.0)

def laplacian(u, f, method='finitediff', eps=1e-7):
    """Compute the Laplacian of the vector field `f` with respect to the input `u`.

    Note: the Laplacian of a vector field is just the vector of Laplacians of its components.

    Args:
        u (torch.Tensor) : the input to the function f of shape [..., D]
        f (function) : some vector field (must support autodiff if using method='autodiff') with output dim [D]
        method (str) : method for calculating the Laplacian. 
            options: ['autodiff', 'finitediff']

        Finite diff currently assumes that `u` represents a 2D vector field.

        Returns:
            (torch.Tensor) : Laplacian of shape [..., 1]
    """
    if method == 'autodiff':
        raise NotImplementedError
    if method == 'finitediff':
        fu = 2.0 * f(u)
        dfux = f(u + eps_x) - fu + f(u - eps_x)
        dfuy = f(u + eps_y) - fu + f(u - eps_y)
        return (dfux + dfuy) / (eps**2.0)


