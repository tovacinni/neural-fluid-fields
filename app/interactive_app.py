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

class Network(nn.Module):
    def __init__(self, 
        input_dim = 2, 
        output_dim = 2, 
        activation = torch.sin, 
        bias = True, 
        num_layers = 1, 
        hidden_dim = 128):
        
        super().__init__()
        self.activation = activation
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim, bias=bias))
            else: 
                layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
        self.layers = nn.ModuleList(layers)
        self.lout = nn.Linear(hidden_dim, output_dim, bias=bias)

    def forward(self, x):
        h = x
        for i, l in enumerate(self.layers):
            h = self.activation(l(h))
        out = torch.sigmoid(self.lout(h))
        return out

class InteractiveApp(sys.modules[backend].Window):

    def __init__(self, render_res=[720, 1024]):
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
    
    def init_state(self):

        self.net = Network()
        self.net = self.net.to('cuda')
        self.net.eval()

    def on_draw(self, dt):
        self.set_title(str(self.fps).encode("ascii"))
        tex = self.screen['tex']
        h,w = tex.shape[:2]

        # render with pytorch
        state = torch.zeros(*self.render_res, 4, device='cuda')

        coords = normalized_grid(*self.render_res)

        state[...,:2] = self.net(coords * 100)
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

if __name__=='__main__':
    app.use('glfw')
    window = InteractiveApp()
    window.init_state()
    app.run()

