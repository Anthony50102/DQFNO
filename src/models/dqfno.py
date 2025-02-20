import torch
import torch.nn as nn
import torch.nn.functional as F
class SpectralConv(nn.Module):
    def __init__(self, in_channels, out_channels, vars, modes1, modes2, time):
        super(SpectralConv, self).__init__()
        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        Expects input shape: [batch, in_channels, x, y].
        """
        self.vars = vars
        self.in_channels = 4 
        self.out_channels = 4
        self.modes1 = modes1  # Number of Fourier modes to keep along x
        self.modes2 = modes2  # Number of Fourier modes to keep along y
        self.time = time

        # Initialize the learnable complex-valued weights
        self.scale = 1 / (self.in_channels * self.out_channels)
        self.weights1 = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, self.vars,
                                                      self.modes1, self.modes2,
                                                      dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, self.vars,
                                                      self.modes1,self.modes2,
                                                      dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        print(input.shape, weights.shape)
        # (batch, in_channel, vars, x, y), (in_channel, out_channel, vars, x, y) -> (batch, out_channel, vars, x, y)
        return torch.einsum("b i t v x y, i o v x y -> b o t v x y", input, weights)

    def forward(self, x):
        """
        x: [batch, in_channels, vars, x, y]
        Returns: [batch, out_channels, vars, x, y]
        """
        b, c, t, v, h, w = x.shape

        # Compute Fourier coefficients
        x_ft = torch.fft.rfft2(x)  # shape: [b, c, t, v, h, w//2+1]

        # Allocate output in Fourier space
        out_ft = torch.zeros(b, self.out_channels, t, v ,h, w//2 + 1, dtype=torch.cfloat, device=x.device)

        # Multiply the low-frequency part
        out_ft[:, :, :, :, :self.modes1, :self.modes2] = \
          self.compl_mul2d(x_ft[:, :, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, :, :, -self.modes1:, :self.modes2] = \
          self.compl_mul2d(x_ft[:, :, :, -self.modes1:, :self.modes2], self.weights2)

        # Inverse FFT to get back to physical space
        x = torch.fft.irfft2(out_ft, s=(h, w))
        return x

from functools import partialmethod
from typing import Tuple, List, Union, Literal

Number = Union[float, int]

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.embeddings import GridEmbedding2D
# from ..layers.spectral_convolution import SpectralConv
from ..layers.padding import DomainPadding
# from ..layers.fno_block import FNOBlocks
from ..layers.channel_mlp import ChannelMLP
from .base_model import BaseModel

class DQFNO(BaseModel, name='DQFNO'):
    """N-Dimensional Derived Quantities Fourier Neural Operator. The FNO learns a mapping between
    spaces of functions discretized over regular grids using Fourier convolutions, 
    as described in [1]_.
    
    The key component of an FNO is its SpectralConv layer (see 
    ``neuralop.layers.spectral_convolution``), which is similar to a standard CNN 
    conv layer but operates in the frequency domain.


    Parameters
    ----------
    n_modes : List[Tuple]
        number of modes to keep in Fourier Layer, along each dimension
        The dimensionality of the FNO is inferred from ``len(n_modes)``
    in_channels : int
        Number of channels in input function
    out_channels : int
        Number of channels in output function
    hidden_channels : int
        width of the FNO (i.e. number of channels), by default 256
    n_layers : int, optional
        Number of Fourier Layers, by default 4

    Documentation for more advanced parameters is below.

    Other parameters
    ------------------
    positional_embedding : Union[str, nn.Module], optional
        Positional embedding to apply to last channels of raw input
        before being passed through the FNO. Defaults to "grid"

        * If "grid", appends a grid positional embedding with default settings to 
        the last channels of raw input. Assumes the inputs are discretized
        over a grid with entry [0,0,...] at the origin and side lengths of 1.

        * If an initialized GridEmbedding module, uses this module directly
        See :mod:`neuralop.embeddings.GridEmbeddingND` for details.

        * If None, does nothing

    non_linearity : nn.Module, optional
        Non-Linear activation function module to use, by default F.gelu
    # TODO 
    Examples
    ---------
    
    >>> from neuralop.models import DQFNO
    >>> model = FNO(n_modes=(12,12), in_channels=1, out_channels=1, hidden_channels=64)
    >>> model
    FNO(
    (positional_embedding): GridEmbeddingND()
    (fno_blocks): FNOBlocks(
        (convs): SpectralConv(
        (weight): ModuleList(
            (0-3): 4 x DenseTensor(shape=torch.Size([64, 64, 12, 7]), rank=None)
        )
        )
            ... torch.nn.Module printout truncated ...

    References
    -----------
    .. [1] :

    Li, Z. et al. "Fourier Neural Operator for Parametric Partial Differential 
        Equations" (2021). ICLR 2021, https://arxiv.org/pdf/2010.08895.

    """

    def __init__(
        self,
        modes: Tuple[int],
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_layers: int=4,
        lifting_channel_ratio: int=2,
        projection_channel_ratio: int=2,
        positional_embedding: Union[str, nn.Module]="grid",
        non_linearity: nn.Module=F.gelu,
        conv_module: nn.Module=SpectralConv,
        **kwargs
    ):
        
        super().__init__()
        self.n_dim = len(modes)
        
        # n_modes is a special property - see the class' property for underlying mechanism
        # When updated, change should be reflected in fno blocks

        self.hidden_channels = hidden_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers

        # init lifting and projection channels using ratios w.r.t hidden channels
        self.lifting_channel_ratio = lifting_channel_ratio
        self.lifting_channels = lifting_channel_ratio * self.hidden_channels

        self.projection_channel_ratio = projection_channel_ratio
        self.projection_channels = projection_channel_ratio * self.hidden_channels

        self.non_linearity = non_linearity
        
        # TODO - understand why we are adding a 3rd grid embedding 
        if positional_embedding == "grid":
            spatial_grid_boundaries = [[0., 1.]] * self.n_dim
            self.positional_embedding = GridEmbedding2D()
        else:
            raise ValueError(f"Error: tried to instantiate FNO positional embedding with {positional_embedding},\
                              expected one of \'grid\', GridEmbeddingND")
        
        self._modes = modes
        # self.fno_blocks = FNOBlocks(
        #     in_channels=hidden_channels,
        #     out_channels=hidden_channels,
        #     modes=self.modes,
        #     non_linearity=non_linearity,
        #     conv_module=conv_module,
        #     n_layers=n_layers,
        #     **kwargs
        # )
        self.fourier_layers = nn.ModuleList()
        self.conv3d_layers  = nn.ModuleList()
        for _ in range(self.n_layers):
            # We'll do the 2D Fourier transform on shape [B, (V*C), X, Y].
            # So the in/out channels for SpectralConv2d = V*C => V*C.
            # But we do not fix V*C up front, we will flatten on the fly below.
            print(self.in_channels)
            fourier_block = SpectralConv(
                in_channels=self.lifting_channels,
                out_channels=self.lifting_channels,
                vars=3,
                modes1=8,
                modes2=8,
                time=10
            )
            self.fourier_layers.append(fourier_block)

            # 3D conv across the triple (variables, X, Y). The 'channels' dimension
            # is the "lifted_channels" in code.  So in shape [B, C=32, D=var_dim, H=X, W=Y].
            # We'll keep the output channels the same as input channels (32).
            conv3d_block = nn.Conv3d(
                in_channels=self.lifting_channels,
                out_channels=self.lifting_channels,
                kernel_size=1,  # you could try larger kernels in X/Y if desired
                padding=0
            )
            self.conv3d_layers.append(conv3d_block)
        
        # if adding a positional embedding, add those channels to lifting
        lifting_in_channels = self.in_channels
        if self.positional_embedding is not None:
            lifting_in_channels += 2

        # # with a hidden layer of size lifting_channels
        # if self.lifting_channels:
        #     print(f"Lifting channels MLP")
        #     self.lifting = ChannelMLP(
        #         in_channels=lifting_in_channels,
        #         out_channels=self.hidden_channels,
        #         hidden_channels=self.lifting_channels,
        #         n_layers=2,
        #         n_dim=self.n_dim,
        #         non_linearity=non_linearity
        #     )

        # otherwise, make it a linear layer
        # else:
        print(f"Lifting channels linear MLP")
        # self.lifting = ChannelMLP(
        #     in_channels=lifting_in_channels,
        #     hidden_channels=self.hidden_channels,
        #     out_channels=self.hidden_channels,
        #     n_layers=1,
        #     n_dim=self.n_dim,
        #     non_linearity=non_linearity
        # )
        self.lifting = nn.Linear(lifting_in_channels, self.hidden_channels)

        self.projection = ChannelMLP(
            in_channels=self.hidden_channels,
            out_channels=out_channels,
            hidden_channels=self.projection_channels,
            n_layers=2,
            n_dim=self.n_dim,
            non_linearity=non_linearity,
        )

    def forward(self, x, output_shape=None, **kwargs):
        """DQFNO's forward pass
        
        1. Applies optional positional encoding

        2. Sends inputs through a lifting layer to a high-dimensional latent space

        3. Applies optional domain padding to high-dimensional intermediate function representation

        4. Applies `n_layers` Fourier/FNO layers in sequence (SpectralConvolution + skip connections, nonlinearity) 

        5. If domain padding was applied, domain padding is removed

        6. Projection of intermediate function representation to the output channels

        Parameters
        ----------
        x : tensor
            input tensor : shape (B, F, X, Y, C)
        
        output_shape : {tuple, tuple list, None}, default is None
            Gives the option of specifying the exact output shape for odd shaped inputs.
            
            * If None, don't specify an output shape

            * If tuple, specifies the output-shape of the **last** FNO Block

            * If tuple list, specifies the exact output-shape of each FNO Block
        """

        if output_shape is None:
            output_shape = [None]*self.n_layers
        elif isinstance(output_shape, tuple):
            output_shape = [None]*(self.n_layers - 1) + [output_shape]

        # append spatial pos embedding if set
        if self.positional_embedding is not None:
            x = self.positional_embedding(x)
        
        
        # lift 
        b,c,t,v,h,w = x.shape
        x = x.permute(0,2,3,4,5,1)
        x = self.lifting(x)
        x = x.permute(0,5,1,2,3,4)
        

        # run through fnos
        # for layer_idx in range(self.n_layers):
        #     x = self.fno_blocks(x, layer_idx, output_shape=output_shape[layer_idx])

        for fourier_op, conv3d_op in zip(self.fourier_layers, self.conv3d_layers):

            # (a) SpectralConv2d across X,Y
            #     Flatten (V, C) => single "channels" dimension:
            b, c, t, v, h, w = x.shape
            x_ft = fourier_op(x)            # => [B, V*C, H, W]
            x_ft = x_ft.view(b, c, t, v, h, w)         # => [B, V, C, H, W]
            x_c3d_out = conv3d_op(x_ft)         # => [B, C, V, H, W]

            x = x_ft + x_c3d_out
            x = F.gelu(x)

        x = self.projection(x)

        return x

    @property
    def modes(self):
        return self._modes

    @modes.setter
    def modes(self, modes):
        self.fno_blocks.modes = modes
        self._modes = modes

