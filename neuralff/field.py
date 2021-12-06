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

from neuralff.ops import sample_from_grid
from neuralff.model import BasicNetwork

class BaseField(nn.Module):
    def sample(self, coords):
        raise NotImplementedError
    def forward(self, coords):
        return self.sample(coords)


class RegularVectorField(BaseField):
    def __init__(self, height, width, fdim=2):
        super().__init__()
        self.vector_field = nn.Parameter(torch.randn([height, width, fdim]))

    def sample(self, coords):
        shape = coords.shape
        samples = sample_from_grid(coords.reshape(-1, shape[-1]), self.vector_field)
        sample_dim = samples.shape[-1]
        return samples.reshape(*shape[:-1], sample_dim)
    
    def update(self, vector_field):
        self.vector_field = nn.Parameter(vector_field)

class ImageDensityField(RegularVectorField):
    def sample(self, coords):
        alpha = 3.0 - super().sample(coords).sum(-1, keepdim=True)
        return (alpha + 1e-1) * 255

class NeuralField(BaseField):
    def __init__(self, **kwargs):
        super().__init__()
        self.vector_field = BasicNetwork(**kwargs)

    def sample(self, coords):
        vector = self.vector_field(coords*100)
        #vector[torch.abs(coords) >= 1.0] = 0
        return vector

class RegularNeuralField(BaseField):
    def __init__(self, height, width, fdim, **kwargs):
        super().__init__()
        self.vector_field = BasicNetwork(**kwargs)
        self.feature_field = nn.Parameter(torch.randn([height, width, fdim]))
    
    def sample(self, coords):
        shape = coords.shape
        features = sample_from_grid(coords.reshape(-1, shape[-1]), self.feature_field)
        feature_dim = features.shape[-1]
        features = features.reshape(*shape[:-1], feature_dim)
        return self.vector_field(features)


