from typing import List, Optional, Union

import torch
from torch import nn
import torch.nn.functional as F

from .channel_mlp import ChannelMLP
from .spectral_convolution import SpectralConv
from ..utils import validate_scaling_factor


Number = Union[int, float]


class FNOBlocks(nn.Module):
    """FNOBlocks implements a sequence of Fourier layers, the operations of which 
    are first described in [1]_. The exact implementation details of the Fourier 
    layer architecture are discussed in [2]_.

    Parameters
    ----------
    in_channels : int
        input channels to Fourier layers
    out_channels : int
        output channels after Fourier layers
    n_modes : List[Tuple[int,int]]
        number of modes to keep along each variable & dimension 
        in frequency space.     
    n_layers : int, optional
            number of Fourier layers to apply in sequence, by default 1
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        modes,
        n_layers=1,
        non_linearity=F.gelu,
        conv_module=SpectralConv,
        debug: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.modes = modes
        self.n_dim = len(modes)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.non_linearity = non_linearity
        self.debug = debug
        
        # Instatiate the Fourier Layers
        self.spectral_convs = []
        self.conv3ds  = [] 
        for _ in range(self.n_layers):
            self.spectral_convs.append(SpectralConv(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                vars=3,
                modes=8,
                )
            )
            self.conv3ds.append(nn.Conv3d(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                kernel_size=1, 
                padding=0
                )
            )
        self.spectral_convs = nn.ModuleList(self.spectral_convs)
        self.conv3ds = nn.ModuleList(self.conv3ds)

    # TODO - Implement variable wise instead and test if better?
    def forward(self, x, index=0, output_shape=None, conv_type: str = "spatio"):
        x_spec = self.spectral_convs[index](x)
        if conv_type == "spatio":
            b,c,t,v,h,w = x.shape
            x = x.reshape(b*v, c, t, h, w)
            x_conv = self.conv3ds[index](x)
            x_conv = x_conv.reshape(b,c,t,v,h,w)
        
        x = x_spec + x_conv
        x = self.non_linearity(x)
        return x
    
    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        for i in range(self.n_layers):
            self.convs[i].n_modes = n_modes
        self._n_modes = n_modes

    def get_block(self, indices):
        """Returns a sub-FNO Block layer from the jointly parametrized main block

        The parametrization of an FNOBlock layer is shared with the main one.
        """
        if self.n_layers == 1:
            raise ValueError(
                "A single layer is parametrized, directly use the main class."
            )

        return SubModule(self, indices)

    def __getitem__(self, indices):
        return self.get_block(indices)


class SubModule(nn.Module):
    """Class representing one of the sub_module from the mother joint module

    Notes
    -----
    This relies on the fact that nn.Parameters are not duplicated:
    if the same nn.Parameter is assigned to multiple modules,
    they all point to the same data, which is shared.
    """

    def __init__(self, main_module, indices):
        super().__init__()
        self.main_module = main_module
        self.indices = indices

    def forward(self, x):
        return self.main_module.forward(x, self.indices)