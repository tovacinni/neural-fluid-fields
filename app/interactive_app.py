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
import torch.optim as optim

import pycuda.driver
from pycuda.gl import graphics_map_flags
from glumpy import app, gloo, gl

from torch.profiler import profile, record_function, ProfilerActivity

import neuralff
from neuralff.model import BasicNetwork
import neuralff.ops as nff_ops

from scipy import sparse
import scipy.sparse.linalg as linalg

import cv2
import skimage
import imageio

import tqdm

import argparse

def load_rgb(path):
    img = imageio.imread(path)
    img = skimage.img_as_float32(img)
    img = img[:,:,:3]
    #img -= 0.5
    #img *= 2.
    #img = img.transpose(2, 0, 1)
    img = img.transpose(1, 0, 2)
    return img

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

def load_rgb(path):
    img = imageio.imread(path)
    img = skimage.img_as_float32(img)
    img = img[:,:,:3]
    return img

def resize_rgb(img, height, width, interpolation=cv2.INTER_LINEAR):
    img = cv2.resize(img, dsize=(height, width), interpolation=interpolation)
    return img

class InteractiveApp(sys.modules[backend].Window):

    #def __init__(self, render_res=[720, 1024]):
    #def __init__(self, render_res=[100, 200]):
    def __init__(self, args):
        
        self.args = args
        self.rgb = torch.from_numpy(load_rgb(self.args.image_path)).cuda()
        render_res = self.rgb.shape[:2]
        self.render_res = render_res

        print("Controls:")
        print("h,l: switch optimization modes")
        print("j,k: switch display buffer")
        print("q  : quit simulation")
        print("n  : begin simulation")

        super().__init__(width=render_res[1], height=render_res[0], 
                         fullscreen=False, config=app.configuration.get_default())
        
        import pycuda.gl.autoinit
        import pycuda.gl
        assert torch.cuda.is_available()
        print('using GPU {}'.format(torch.cuda.current_device()))
        self.buffer = torch.zeros(*render_res, 4, device='cuda')

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

        #self.mode = "stable_fluids"
        self.mode = "neuralff"

        self.display_modes = ["rgb", "pressure", "velocity", "rho", "divergence", "euler"]
        self.display_mode_idx = 0
        self.display_mode = self.display_modes[self.display_mode_idx]
        
        self.optim_modes = ["euler", "divergence-free", "split"]
        self.optim_mode_idx = 0
        self.optim_mode = self.optim_modes[self.optim_mode_idx]

        self.max_euler_error = 0.0
        self.max_divergence_error = 0.0
        self.curr_error = 0.0
        self.optim_switch = False
        self.begin_switch = False

    def on_draw(self, dt):
        title = f"FPS: {self.fps:.3f}"
        title += f"  Buffer: {self.display_mode}"
        
        if self.display_mode == "divergence" or self.display_mode == "euler":
            title += f"  Error: {self.curr_error:.3e}"
        
        title += f"  Optimizing: {self.optim_mode}"

        self.set_title(title.encode("ascii"))
        tex = self.screen['tex']
        h,w = tex.shape[:2]

        # render with pytorch
        state = torch.zeros(*self.render_res, 4, device='cuda')

        coords = nff_ops.normalized_grid_coords(*self.render_res)

        out = self.render(coords)

        write_dim = out.shape[-1]

        state[...,:write_dim] = out
        state[...,3] = 1
        state = torch.flip(state, [0])

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
    
    def on_key_press(self, symbol, modifiers):
        if symbol == 75: # k
            self.display_mode_idx = (self.display_mode_idx + 1) % len(self.display_modes)
            self.display_mode = self.display_modes[self.display_mode_idx]
        elif symbol == 74: # j
            self.display_mode_idx = (self.display_mode_idx - 1) % len(self.display_modes)
            self.display_mode = self.display_modes[self.display_mode_idx]
        elif symbol == 81: # q
            self.close()
        elif symbol == 78: # n
            self.optim_switch = not self.optim_switch
        elif symbol == 76: # l
            self.optim_mode_idx = (self.optim_mode_idx + 1) % len(self.optim_modes)
            self.optim_mode = self.optim_modes[self.optim_mode_idx]
        elif symbol == 72: # h
            self.optim_mode_idx = (self.optim_mode_idx - 1) % len(self.optim_modes)
            self.optim_mode = self.optim_modes[self.optim_mode_idx]
        elif symbol == 66: # b
            self.begin_switch = not self.begin_switch

    def init_state(self):

        #self.gravity = 1e-4
        #self.timestep = 1e-1
        #self.timestep = 5e-2
        #self.timestep = 1e-5
        
        #self.timestep = 5e-3
        #self.timestep = 5e-2
        self.timestep = 1e-1

        self.image_coords = nff_ops.normalized_grid_coords(self.height, self.width, aspect=False, device="cuda")
        self.image_coords[...,1] *= -1

        if self.mode == "stable_fluids":
            self.grid_width = self.width // 8
            self.grid_height = self.height // 8

        elif self.mode == "neuralff":
            
            velocity_field_config = {
                "input_dim" : 2,    
                "output_dim" : 2,    
                "hidden_activation" : torch.sin,    
                "output_activation" : None,
                "bias" : True,    
                "num_layers" : 4,    
                "hidden_dim" : 128,    
            }
        
            self.velocity_field = neuralff.NeuralField(**velocity_field_config).cuda()

            pressure_field_config = {
                "input_dim" : 2,    
                "output_dim" : 1,    
                "hidden_activation" : torch.sin,    
                "output_activation" : None,
                "bias" : True,    
                "num_layers" : 4,    
                "hidden_dim" : 128,    
            }

            self.pressure_field = neuralff.NeuralField(**pressure_field_config).cuda()
            
            self.rho_field = neuralff.ImageDensityField(self.height, self.width)
            self.rho_field.update(self.rgb)

            self.pc_lr = self.args.pc_lr
            self.precondition_optimizer = optim.Adam([
                {"params": self.velocity_field.parameters(), "lr":self.pc_lr},
                {"params": self.pressure_field.parameters(), "lr":self.pc_lr},
            ])

            self.lr = self.args.lr
            self.optimizer = optim.Adam([
            #self.optimizer = optim.SGD([
                {"params": self.velocity_field.parameters(), "lr":self.lr},
                {"params": self.pressure_field.parameters(), "lr":self.lr},
            ])
            
            if self.args.precondition:
                self.precondition()

    def precondition(self):

        num_batch = self.args.pc_num_batch
        batch_size = self.args.pc_batch_size
        epochs = self.args.pc_epochs
        pts = torch.rand([batch_size*num_batch, 2], device='cuda') * 2.0 - 1.0
        
        initial_velocity = self.velocity_field.sample(pts).detach()
        print("Preconditioning body forces...")
        for i in tqdm.tqdm(range(epochs)):
            for j in range(num_batch):
                self.velocity_field.zero_grad()
                self.pressure_field.zero_grad()
                
                loss = nff_ops.body_forces_loss(
                        pts[j*batch_size:(j+1)*batch_size], 
                        self.velocity_field, self.timestep, 
                        initial_velocity=initial_velocity[j*batch_size:(j+1)*batch_size]) 
                loss = loss.mean()
                loss.backward()
                self.precondition_optimizer.step()

        print("Preconditioning divergence...")
        for i in tqdm.tqdm(range(epochs)):
            for j in range(num_batch):
                self.velocity_field.zero_grad()
                self.pressure_field.zero_grad()
                
                loss = nff_ops.divergence_free_loss(
                        pts[j*batch_size:(j+1)*batch_size], 
                        self.velocity_field) 
                loss = loss.mean()
                loss.backward()
                self.precondition_optimizer.step()
        
        initial_velocity = self.velocity_field.sample(pts).detach()
        
        print("Preconditioning Euler...")
        for i in tqdm.tqdm(range(epochs)):
            for j in range(num_batch):
                self.velocity_field.zero_grad()
                self.pressure_field.zero_grad()
                
                loss = nff_ops.euler_loss(
                        pts[j*batch_size:(j+1)*batch_size], 
                        self.velocity_field,
                        self.pressure_field, self.rho_field, self.timestep,
                        initial_velocity=initial_velocity[j*batch_size:(j+1)*batch_size]) 
                loss = loss.mean()
                loss.backward()
                self.precondition_optimizer.step()
    
    def render(self, coords):
        
        self.optimizer = optim.Adam([
        #self.optimizer = optim.SGD([
            {"params": self.velocity_field.parameters(), "lr":self.lr},
            {"params": self.pressure_field.parameters(), "lr":self.lr},
        ])
        
        if self.mode == "stable_fluids":
            # Add external forces
            self.velocity_field.vector_field[..., 1] += 9.8 * self.timestep * self.gravity
            
            # Remove divergence
            #self.velocities = remove_divergence(self.velocities, self.x_mapper, self.y_mapper)

        elif self.optim_switch:
            for i in range(6):
                self.pressure_field.zero_grad()
                self.velocity_field.zero_grad()
                pts = torch.rand([self.args.batch_size, 2], device=coords.device) * 2.0 - 1.0
                if self.optim_mode == "divergence-free":
                    loss = nff_ops.divergence_free_loss(pts, self.velocity_field)
                elif self.optim_mode == "split":
                    loss = nff_ops.body_forces_loss(pts, self.velocity_field, self.timestep) +\
                           nff_ops.incompressibility_loss(pts, self.velocity_field, self.pressure_field, 
                                   self.rho_field, self.timestep)
                elif self.optim_mode == "euler":
                    loss = nff_ops.euler_loss(pts, self.velocity_field, 
                            self.pressure_field, self.rho_field, self.timestep)
                loss.mean().backward()
                self.optimizer.step()
    

        with torch.no_grad():
            if self.display_mode == "rgb": 
                if self.begin_switch:
                    self.rho_field.update(nff_ops.semi_lagrangian_advection(
                        self.image_coords, self.rho_field.vector_field, self.velocity_field, self.timestep))    
                return self.rho_field.vector_field
            elif self.display_mode == "pressure":
                return (1.0 + self.pressure_field.sample(self.image_coords)) / 2.0
            elif self.display_mode == "velocity":
                return (1.0 + F.normalize(self.velocity_field.sample(self.image_coords), dim=-1)) / 2.0
            elif self.display_mode == "rho":
                rfsample = self.rho_field.sample(self.image_coords)
                return rfsample / rfsample.max()
            elif self.display_mode == "divergence":
                div = nff_ops.divergence(self.image_coords, self.velocity_field, method='finitediff')**2
                err = div.max()
                self.curr_error = err
                self.max_divergence_error = max(err, self.max_divergence_error)
                return div / self.max_divergence_error
                #return div / err
            elif self.display_mode == "euler":
                loss = nff_ops.euler_loss(self.image_coords, self.velocity_field, 
                        self.pressure_field, self.rho_field, self.timestep)
                err = loss.max()
                self.curr_error = err
                self.max_euler_error = max(err, self.max_euler_error)
                #return loss / self.max_euler_error
                return loss / err
            else:
                return torch.zeros_like(coords)

def parse_options():
    parser = argparse.ArgumentParser(description='Fluid simulation with neural networks.')
    
    # Global arguments
    global_group = parser.add_argument_group('global')
    global_group.add_argument('--lr', type=float, default=1e-6,
                              help='Learning rate for the simulation.')
    global_group.add_argument('--batch_size', type=int, default=4096,
                              help='Batch size for the simulation.')
    global_group.add_argument('--pc_lr', type=float, default=1e-6,
                              help='Learning rate for the preconditioner.')
    global_group.add_argument('--pc_batch_size', type=int, default=4096,
                              help='Batch size for the preconditioner.')
    global_group.add_argument('--pc_num_batch', type=int, default=10,
                              help='Number of batches to use for the preconditioner.')
    global_group.add_argument('--pc_epochs', type=int, default=100,
                              help='Number of epochs to train the preconditioner for.')
    global_group.add_argument('--precondition', action='store_true',
                              help='Use the preconditioner.')
    global_group.add_argument('--image_path', type=str, default="./data/test.png",
                              help='Path to the image to use for the simulation.')

    return parser.parse_args()

if __name__=='__main__':
    args = parse_options()
    app.use('glfw')
    window = InteractiveApp(args)
    window.init_state()
    app.run()


