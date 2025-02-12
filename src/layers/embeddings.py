from abc import ABC, abstractmethod
import torch
from torch import nn

# Base class for embeddings
class Embedding(nn.Module, ABC):
    def __init__(self):
        super().__init__()
    
    @property
    @abstractmethod
    def out_channels(self):
        pass

# Utility function: creates a regular 2D grid given spatial dimensions.
def regular_grid_2d(spatial_dims, grid_boundaries=[[0, 1], [0, 1]]):
    """
    Creates a grid of x and y coordinates over a grid of size (height, width) 
    (here height = x and width = y). The output grid_x and grid_y will each have 
    shape (x, y).
    """
    height, width = spatial_dims
    # Create evenly spaced values in each dimension
    xt = torch.linspace(grid_boundaries[0][0], grid_boundaries[0][1], height)
    yt = torch.linspace(grid_boundaries[1][0], grid_boundaries[1][1], width)
    # Use torch.meshgrid to create the 2D coordinate grid (with indexing='ij')
    grid_x, grid_y = torch.meshgrid(xt, yt, indexing='ij')
    return grid_x, grid_y

# Modified GridEmbedding2D for inputs of shape (B, C, T, F, x, y)
class GridEmbedding2D(Embedding):
    """
    This grid embedding module simply concatenates two extra channels corresponding 
    to the x and y coordinates over the last two dimensions of the input.
    
    For an input of shape (batch, channel, time, fields, x, y), the output will have 
    shape (batch, channel+2, time, fields, x, y).
    
    The grid is defined over the x and y dimensions using the provided grid_boundaries.
    """
    def __init__(self, grid_boundaries=[[0, 1], [0, 1]]):
        super().__init__()
        self.grid_boundaries = grid_boundaries
        self._grid = None
        self._res = None  # Cache the grid for a given (x,y) resolution
    
    @property
    def out_channels(self):
        # The output will add 2 extra channels (x and y) regardless of the input channel count.
        return 2

    def grid(self, spatial_dims, device, dtype):
        """
        Given the spatial dimensions (x, y), create (or reuse) the grid embeddings.
        The produced grid_x and grid_y will have shape (1, 1, x, y).
        """
        if self._grid is None or self._res != spatial_dims:
            grid_x, grid_y = regular_grid_2d(spatial_dims, grid_boundaries=self.grid_boundaries)
            # Move to the proper device/dtype and add two singleton dims: one for batch and one for channel.
            grid_x = grid_x.to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
            grid_y = grid_y.to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
            self._grid = (grid_x, grid_y)
            self._res = spatial_dims
        return self._grid

    def forward(self, data):
        """
        Expects data of shape (B, C, T, F, x, y).
        It creates a grid of shape (1, 1, x, y), then unsqueezes extra dimensions 
        for time and fields, expands to (B, 1, T, F, x, y), and concatenates along 
        the channel dimension.
        """
        # Unpack the input dimensions
        B, C, T, F, X, Y = data.shape

        # Get the grid for the last two (spatial) dimensions.
        grid_x, grid_y = self.grid((X, Y), data.device, data.dtype)
        # The grid_x and grid_y are currently of shape (1, 1, X, Y).
        # Insert two singleton dimensions (for time and fields) so that they have shape (1, 1, 1, 1, X, Y)
        grid_x = grid_x.unsqueeze(2).unsqueeze(3)
        grid_y = grid_y.unsqueeze(2).unsqueeze(3)
        # Expand the grid along batch, time, and fields dimensions
        grid_x = grid_x.expand(B, -1, T, F, -1, -1)
        grid_y = grid_y.expand(B, -1, T, F, -1, -1)
        
        # Concatenate along the channel dimension
        return torch.cat((data, grid_x, grid_y), dim=1)