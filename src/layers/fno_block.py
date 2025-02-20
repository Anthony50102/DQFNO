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
    n_modes : int, List[int]
        number of modes to keep along each dimension 
        in frequency space. Can either be specified as
        an int (for all dimensions) or an iterable with one
        number per dimension
    n_layers : int, optional
        number of Fourier layers to apply in sequence, by default 1

    References
    -----------
    .. [1] Li, Z. et al. "Fourier Neural Operator for Parametric Partial Differential 
           Equations" (2021). ICLR 2021, https://arxiv.org/pdf/2010.08895.
    .. [2] Kossaifi, J., Kovachki, N., Azizzadenesheli, K., Anandkumar, A. "Multi-Grid
           Tensorized Fourier Neural Operator for High-Resolution PDEs" (2024). 
           TMLR 2024, https://openreview.net/pdf?id=AWiDlO63bH.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        modes,
        n_layers=1,
        non_linearity=F.gelu,
        conv_module=SpectralConv,
        **kwargs,
    ):
        super().__init__()
        self.modes = modes
        self.n_dim = len(modes)


        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        #
        self.non_linearity = non_linearity
        
        self.convs = nn.ModuleList([
                conv_module(
                self.in_channels,
                self.out_channels,
                self.n_dim,
                self.modes,
            ) 
            for i in range(n_layers)])

        self.channel_mlp = nn.ModuleList(
            [
                ChannelMLP(
                    in_channels=self.out_channels,
                    hidden_channels=round(self.out_channels),
                    n_dim=self.n_dim,
                )
                for _ in range(n_layers)
            ]
        )

    def set_ada_in_embeddings(self, *embeddings):
        """Sets the embeddings of each Ada-IN norm layers

        Parameters
        ----------
        embeddings : tensor or list of tensor
            if a single embedding is given, it will be used for each norm layer
            otherwise, each embedding will be used for the corresponding norm layer
        """
        if len(embeddings) == 1:
            for norm in self.norm:
                norm.set_embedding(embeddings[0])
        else:
            for norm, embedding in zip(self.norm, embeddings):
                norm.set_embedding(embedding)

    def forward(self, x, index=0, output_shape=None):
        return self.forward_with_postactivation(x, index, output_shape)
    
    def forward_with_postactivation(self, x, index=0, output_shape=None):
        # x = torch.tanh(x) #TODO - figure out what this shit is for

        x_fno = self.convs[index](x,)
        #self.convs(x, index, output_shape=output_shape)

        x = x_fno

        if (index < (self.n_layers - 1)):
            x = self.non_linearity(x)

        x = self.channel_mlp[index](x)

        if index < (self.n_layers - 1):
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