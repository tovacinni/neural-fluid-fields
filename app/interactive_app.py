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

import os
import sys
from contextlib import contextmanager
import importlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pycuda.driver
from pycuda.gl import graphics_map_flags
from glumpy import app, gloo, gl

from torch.profiler import profile, record_function, ProfilerActivity

from neuralff.model import BasicNetwork

from scipy import sparse
import scipy.sparse.linalg as linalg


class FeatureImage(nn.Module):
    def __init__(self, fdim, fsize, dim=0):
        super().__init__()
        self.fsize = fsize
        self.fdim = fdim
        self.fm = nn.Parameter(torch.randn(1, fdim, fsize+1, fsize+1) * 0.01)
        self.dims = [0,1,2]
        self.dims.remove(dim)
        self.padding_mode = 'reflection'

    def forward(self, x):
        N = x.shape[0]
        
        # This stuff needs to be fixed... eventually
        if len(x.shape) == 3:
            sample_coords = x.reshape(1, N, x.shape[1], 3) # [N, 1, 1, 3]    
            sample = F.grid_sample(self.fm, sample_coords[...,self.dims], 
                                   align_corners=True, padding_mode=self.padding_mode)[0,:,:,:].transpose(0,1)
        else:
            sample_coords = x.reshape(1, N, 1, 3) # [N, 1, 1, 3]    
            sample = F.grid_sample(self.fm, sample_coords[...,self.dims], 
                                    align_corners=True, padding_mode=self.padding_mode)[0,:,:,0].transpose(0,1)
        return sample


def sample_from_grid(coords, grid, padding_mode="reflection"):
    """Sample from a discrete grid at continuous coordinates.

    Args:
        coords (torch.FloatTensor): continuous coordinates in normalized coords [-1, 1] of size [N, 2]
        grid (torch.FloatTensor): grid of size [H, W, F].
    
    Returns:
        
    """
    N = coords.shape[0]
    sample_coords = x.reshape(1, N, 1, 3)
    samples = F.grid_sample(grid, sample_coords, align_corners=True,
                            padding_mode=self.padding_mode)[0,:,:,0].transpose(0,1)
    return samples




@contextmanager
def cuda_activate(img):
    """Context manager simplifying use of pycuda.gl.RegisteredImage"""
    mapping = img.map()
    yield mapping.array(0,0)
    mapping.unmap()

def create_shared_texture(w, h, c=4,
        map_flags=graphics_map_flags.WRITE_DISCARD,
        dtype=np.uint8):
    """Create and return a Texture2D with gloo and pycuda views."""
    tex = np.zeros((h,w,c), dtype).view(gloo.Texture2D)
    tex.activate() # force gloo to create on GPU
    tex.deactivate()
    cuda_buffer = pycuda.gl.RegisteredImage(int(tex.handle), tex.target, map_flags)
    return tex, cuda_buffer

backend = "glumpy.app.window.backends.backend_glfw"
importlib.import_module(backend)

def normalized_grid(height, width, device="cuda"):
    window_x = torch.linspace(-1, 1, steps=width, device=device) * (width / height)
    window_y = torch.linspace(1, -1, steps=height, device=device)
    coord = torch.stack(torch.meshgrid(window_x, window_y, indexing='ij')).permute(2,1,0)
    return coord

def grad_mat_generator(n):
    mat_size = (n - 1) * n
    result = np.zeros(mat_size)
    result[range(0, mat_size, n + 1)] = -1
    result[range(1, mat_size, n + 1)] = 1
    result = result.reshape([n - 1, n])
    return sparse.csr_matrix(result)

def laplacian_mat_generator(n):
    mat_size = (n - 1) * (n - 1)
    result = np.zeros(mat_size)
    result[range(0, mat_size, n)] = 2
    result[range(1, mat_size, n)] = -1
    result[range(n - 1, mat_size, n)] = -1
    result = result.reshape([n - 1, n - 1])
    return sparse.csr_matrix(result)

def interpolator_generator(n):
    mat_size = (n - 1) * n
    result = np.zeros(mat_size)
    result[range(0, mat_size, n + 1)] = 0.5
    result[range(1, mat_size, n + 1)] = 0.5
    result = result.reshape([n - 1, n])
    return sparse.csr_matrix(result)

def remove_divergence(v, x_mapper, y_mapper):
    # v = v_divergence_free + grad(w)
    # div(v) = laplacian(w)
    # w = laplacian \ div(v)

    # Map velocities from nodes to edges
    vx = y_mapper.dot(v[:,:,0].ravel()).reshape(self.width, self.height - 1)
    vy = x_mapper.dot(v[:,:,1].ravel()).reshape(self.width - 1, self.height)

    # Compute grad(w)
    divergence = x_gradient.dot(vx.ravel()) + y_gradient.dot(vy.ravel())
    w = factorized_laplacian(divergence.ravel())
    grad_w_x = x_gradient.T.dot(w).reshape(self.width, self.height - 1).ravel()
    grad_w_y = y_gradient.T.dot(w).reshape(self.width - 1, self.height).ravel()

    # Make the velocity divergence-free and map it back from edges to nodes
    v[:,:,0] -= y_mapper.T.dot(grad_w_x).reshape(self.width, self.height)
    v[:,:,1] -= x_mapper.T.dot(grad_w_y).reshape(self.width, self.height)

    return v



def advect_and_apply_changes(mass_matrix, v, X, Y, width, height):
    image = np.zeros(self.height, self.width)
    coordinates = np.stack([(X - v[:, :, 0] * time_step) * self.width, (Y - v[:, :, 1] * time_step) * self.height], axis=0)
    image[:, :, 0] = ndimage.map_coordinates(image[:, :, 0], coordinates, mode='wrap')
    image[:, :, 1] = ndimage.map_coordinates(image[:, :, 1], coordinates, mode='wrap')
    image[:, :, 2] = ndimage.map_coordinates(image[:, :, 2], coordinates, mode='wrap')
    # The mass matrix needs to be moved along the points
    mass_matrix = ndimage.map_coordinates(mass_matrix, coordinates, mode='wrap')
    v[:, :, 0] = ndimage.map_coordinates(v[:, :, 0], coordinates, mode='wrap')
    v[:, :, 1] = ndimage.map_coordinates(v[:, :, 1], coordinates, mode='wrap')
    return image, mass_matrix, v

def advect_and_apply_changes(particles, velocities):
    """Apply advection on the particles to animate the image.

    Args:
        feats (torch.FloatTensor) : 
        particles (torch.FloatTensor) : an array of (x_coord, y_coord, mass) of size [N, 3]
        velocities (torch.FloatTensor) : an Eulerian grid of velocities of size [H,W,2]
    """



class InteractiveApp(sys.modules[backend].Window):

    #def __init__(self, render_res=[720, 1024]):
    def __init__(self, render_res=[100, 200]):
        super().__init__(width=render_res[1], height=render_res[0], 
                         fullscreen=False, config=app.configuration.get_default())
        
        import pycuda.gl.autoinit
        import pycuda.gl
        assert torch.cuda.is_available()
        print('using GPU {}'.format(torch.cuda.current_device()))
        self.buffer = torch.zeros(*render_res, 4, device='cuda')

        self.render_res = render_res
        self.camera_origin = np.array([2.5, 2.5, 2.5])
        self.world_transform = np.eye(3)
    
        self.tex, self.cuda_buffer = create_shared_texture(self.width, self.height, 4)
        
        vertex = """
        uniform float scale;
        attribute vec2 position;
        attribute vec2 texcoord;
        varying vec2 v_texcoord;
        void main()
        {
            v_texcoord = texcoord;
            gl_Position = vec4(scale*position, 0.0, 1.0);
        } """
        
        fragment = """
        uniform sampler2D tex;
        varying vec2 v_texcoord;
        void main()
        {
            gl_FragColor = texture2D(tex, v_texcoord);
        } """
        
        self.screen = gloo.Program(vertex, fragment, count=4)
        self.screen['position'] = [(-1,-1), (-1,+1), (+1,-1), (+1,+1)]
        self.screen['texcoord'] = [(0,0), (0,1), (1,0), (1,1)]
        self.screen['scale'] = 1.0
        self.screen['tex'] = self.tex

        self.mode = "stable_fluids"

    def on_draw(self, dt):
        self.set_title(str(self.fps).encode("ascii"))
        tex = self.screen['tex']
        h,w = tex.shape[:2]

        # render with pytorch
        state = torch.zeros(*self.render_res, 4, device='cuda')

        coords = normalized_grid(*self.render_res)

        state[...,:2] = self.render(coords)
        state[...,3] = 1

        img = (255*state).byte().contiguous()

        # copy from torch into buffer
        assert tex.nbytes == img.numel()*img.element_size()
        with cuda_activate(self.cuda_buffer) as ary:
            cpy = pycuda.driver.Memcpy2D()
            cpy.set_src_device(img.data_ptr())
            cpy.set_dst_array(ary)
            cpy.width_in_bytes = cpy.src_pitch = cpy.dst_pitch = tex.nbytes//h
            cpy.height = h
            cpy(aligned=False)
            torch.cuda.synchronize()
        # draw to screen
        self.clear()
        self.screen.draw(gl.GL_TRIANGLE_STRIP)

    def on_close(self):
        pycuda.gl.autoinit.context.pop()

    ####################################
    # Application specific code
    ####################################

    def init_state(self):

        self.gravity = 1e-6
        self.time_step = 1e-1

    
        if self.mode == "stable_fluids":
            # Operator Initialization
            # For mapping the velocities from nodes to edges and vice versa
            self.y_mapper = sparse.kron(sparse.identity(self.width), interpolator_generator(self.height))
            self.x_mapper = sparse.kron(interpolator_generator(self.width), sparse.identity(self.height))
            # For Computing gradient, divergence, and Laplacian
            self.x_gradient = sparse.kron(grad_mat_generator(self.width), sparse.identity(self.height - 1))
            self.y_gradient = sparse.kron(sparse.identity(self.width - 1), grad_mat_generator(self.height))
            self.laplacian = sparse.kronsum(laplacian_mat_generator(self.height),laplacian_mat_generator(self.width))
            # Prefactorize the Laplacian matrix for better performance
            self.factorized_laplacian = linalg.factorized(self.laplacian)    
            self.velocities = np.zeros([self.height, self.width, 2])
            X, Y = np.meshgrid(np.linspace(0,1,self.width, endpoint=False), np.linspace(0,1,self.height, endpoint=False), indexing='ij')

            self.particles = 

            import pdb; pdb.set_trace()

        elif self.mode == "neuralff":
            self.net = BasicNetwork()
            self.net = self.net.to('cuda')
            self.net.eval()
            
    def render(self, coords):
        if self.mode == "stable_fluids":
            # Add external forces
            self.velocities[..., 0] += 10 * self.time_step * self.gravity
            
            # Remove divergence
            #self.velocities = remove_divergence(self.velocities, self.x_mapper, self.y_mapper)
    
            image, mass_matrix, v = advect_and_apply_changes( mass_matrix, v, X, Y, self.width, self.height)

        elif self.mode == "neuralff":
            return self.net(coords * 100)

if __name__=='__main__':
    app.use('glfw')
    window = InteractiveApp()
    window.init_state()
    app.run()

