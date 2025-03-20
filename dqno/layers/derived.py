import torch
from typing import List
from torch import nn
from torch.func import vmap
from ..functions.hw import HasegawaWatakini
from .channel_mlp import ChannelMLP
import torch.nn as nn
import torch.nn.functional as F

class DerivedMLP(nn.Module):
    """
    Module for taking in input state computing derived quanities,
    using predicted next state computing derived quanities and then
    using linear layer (or in future conv) to inform/compute future
    derived quanities
    """
    def __init__(self, dx: float = None, _type: str = 'direct'):
        """
        layer: List[int], layers of the MLP
        """
        super().__init__()
        if _type == "direct":
            self.hw = HasegawaWatakini(dx=dx)
        elif _type == "mlp":
            self.mlp = ChannelMLP(
                in_channels=2,
                hidden_channels=4,
                out_channels=1
                )
        elif _type == "cnn":
            self.cnn = TimeDistributedCNN(channel=1, variables=3)
        else:
            raise ValueError(f"Type of derived quanity f{type(_type)} is not one of the possile \'none\', \'mlp\', or \'cnn\' ")
        self.input_derived: torch.tensor = None
        self._type = _type # Determine what type of method we are going to use to predict out new outputs
    
    def _compute_derived(self, x: torch.tensor) -> torch.tensor:
        '''
        Take in a state (b,c,t,v,x,y) output a derived state
        (b,t,)
        '''
        
        batch_gamma_n = torch.func.vmap(
        torch.func.vmap(self.hw.gamma_n, in_dims=0, out_dims=0),
            in_dims=0, out_dims=0
            )(x[:,0])
        return batch_gamma_n

    
    def store(self, x: torch.tensor):
        '''
        Store the derived quanities
        '''
        if self._type == 'derived':
            self.input_derived = x
    
    def forward(self, x: torch.tensor):
        '''
        if there is no input state compute and store
        '''
        # Compute predicted state derived 
        x_derived = self._compute_derived(x) if self._type != "cnn" else self.cnn(x).squeeze(-1)
        if self._type == 'derived':
            x = torch.cat((self.input_derived.unsqueeze(0),x_derived.unsqueeze(0)), dim=1) # Shape (B, 2, D)
            x = self.mlp(x).squeeze(1)
            x_derived = x
        return  x_derived # (B, D)


class SimpleFluxCNN(nn.Module):
    """
    A CNN that takes an input of shape (batch, channels, x, y)
    and produces a single scalar output per sample. This version reduces oversmoothing by:
      - Removing the second pooling layer
      - Adding a skip (residual) connection from the first convolution block
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int = 32,
                 hidden_channels: int = 16,
                 kernel_size: int = 3,
                 padding: int = 1,
                 ):
        super(SimpleFluxCNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = padding
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.hidden_channels,
                               kernel_size=self.kernel_size, padding=self.padding)
        self.conv2 = nn.Conv2d(in_channels=self.hidden_channels, out_channels=self.out_channels,
                               kernel_size=self.kernel_size, padding=self.padding)
        
        # Pooling layer (only one pooling after conv1 to reduce oversmoothing)
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        # Residual connection: adjusts channels from conv1 output to conv2's output size
        self.res_conv = nn.Conv2d(in_channels=self.hidden_channels, out_channels=self.out_channels, kernel_size=1)
        
        # Global Average Pool: outputs shape (batch_size, channels, 1, 1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final fully-connected layer to scalar
        self.fc = nn.Linear(in_features=self.out_channels, out_features=1)

    def forward(self, x):
        """
        Input x has shape: (batch_size, channels, x_dim, y_dim)
        """
        # First convolution + ReLU + pooling
        x1 = F.relu(self.conv1(x))
        x1 = self.pool(x1)
        
        # Second convolution without additional pooling
        x2 = F.relu(self.conv2(x1))
        
        # Residual connection: add skip connection from x1 (after adjusting channels)
        skip = self.res_conv(x1)
        x2 = F.relu(x2 + skip)
        
        # Global average pooling to reduce spatial dimensions
        x2 = self.gap(x2)
        
        # Flatten to (batch_size, out_channels)
        x2 = x2.view(x2.size(0), -1)
        
        # Final fully-connected layer to get (batch_size, 1)
        x2 = self.fc(x2)
        
        return x2

class TimeDistributedCNN(nn.Module):
    """
    This wraps the SimpleFluxCNN so that it can be applied
    across the time dimension of an input with shape:
    (batch, channel, time, variables, X, Y).

    The output will have shape (batch, time, 1).
    """
    def __init__(self, channel=3, variables=2):
        super(TimeDistributedCNN, self).__init__()
        
        # Combine 'channel' and 'variables' into a single dimension
        in_channels = channel * variables
        
        # Use the modified SimpleFluxCNN (with reduced pooling and skip connections)
        self.cnn = SimpleFluxCNN(in_channels=in_channels)

    def forward(self, x):
        """
        x: (batch, channel, time, variables, X, Y)
        returns: (batch, time, 1)
        """
        B, C, T, V, X, Y = x.shape
        
        # Step 1: Reshape to merge (B, T) into one dimension and (C, V) into 'in_channels'
        x = x.view(B * T, C * V, X, Y)  # (B*T, in_channels, X, Y)
        
        # Step 2: Apply the CNN
        out = self.cnn(x)  # (B*T, 1)
        
        # Step 3: Reshape back to (B, T, 1)
        out = out.view(B, T, 1)
        
        return out