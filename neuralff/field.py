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
    def __init__(self, height, width):
        super().__init__()
        self.vector_field = nn.Parameter(torch.zeros([height, width, 2]))

    def sample(self, coords):
        return sample_from_grid(coords, self.vector_field)

class NeuralField(BaseField):
    def __init__(self, **kwargs):
        super().__init__()
        self.vector_field = BasicNetwork(**kwargs)

    def sample(self, coords):
        vector = self.vector_field(coords*100)
        #vector[torch.abs(coords) >= 1.0] = 0
        return vector

