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
import torch.nn.functional as F

def normalized_grid_coords(height, width, aspect=True, device="cuda"):
    """Return the normalized [-1, 1] grid coordinates given height and width.

    Args:
        height (int) : height of the grid.
        width (int) : width of the grid.
        aspect (bool) : if True, use the aspect ratio to scale the coordinates, in which case the 
                        coords will not be normalzied to [-1, 1]. (Default: True)
        device : the device the tensors will be created on.
    """
    aspect_ratio = width/height if aspect else 1.0

    window_x = torch.linspace(-1, 1, steps=width, device=device) * aspect_ratio
    window_y = torch.linspace(1, -1, steps=height, device=device)
    coord = torch.stack(torch.meshgrid(window_x, window_y, indexing='ij')).permute(2,1,0)
    return coord

def sample_from_grid(coords, grid, padding_mode="zeros"):
    """Sample from a discrete grid at continuous coordinates.

    Args:
        coords (torch.FloatTensor): continuous coordinates in normalized coords [-1, 1] of size [N, 2]
        grid (torch.FloatTensor): grid of size [H, W, F].
    
    Returns:
        (torch.FloatTensor): interpolated values of [N, F]

    """
    N = coords.shape[0]
    sample_coords = coords.reshape(1, N, 1, 2)
    sample_coords = (sample_coords + 1.0) % 2.0 - 1.0
    samples = F.grid_sample(grid[None].permute(0,3,1,2), sample_coords, align_corners=True,
                            padding_mode=padding_mode)[0,:,:,0].transpose(0,1)
    return samples

