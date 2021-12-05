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
import torch.nn as nn
import torch.nn.functional as F

class BasicNetwork(nn.Module):
    def __init__(self, 
        input_dim = 2, 
        output_dim = 2, 
        activation = torch.sin, 
        bias = True, 
        num_layers = 1, 
        hidden_dim = 32):
        
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
        out = torch.tanh(self.lout(h))
        return out

