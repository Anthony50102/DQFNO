from typing import Tuple, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.embeddings import GridEmbedding2D
from ..layers.channel_mlp import ChannelMLP
from ..layers.spectral_convolution import SpectralConv
from ..layers.fno_block import FNOBlocks
from .base_model import BaseModel

class DQFNO(BaseModel, name='DQFNO'):
    # TODO - Implement more doc strings
    """3-Dimensional Derived Quanitity Fourier Neural Operator. The FNO learns a mapping between
    spaces of functions discretized over regular grids using Fourier convolutions, additionally 
    it learns the mappings of these functions to their physically based derived quanities.
    
    The key component of an DQFNO is its SpectralConv layer (see 
    ``neuralop.layers.spectral_convolution``), which is similar to a standard CNN 
    conv layer but operates in the frequency domain.

    Parameters
    ----------
    n_modes : List[Tuple]
        number of modes to keep in Fourier Layer, along each variable and 
        dimension. The dimensionality of the FNO is inferred from ``len(n_modes)``
    in_channels : int
        Number of channels in input function
    out_channels : int
        Number of channels in output function
    hidden_channels : int
        width of the DQFNO (i.e. number of channels), by default 256
    n_layers : int, optional
        Number of Fourier Layers, by default 4"""
    def __init__(
        self,
        modes: List[Tuple[int,int]],
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 256,
        n_layers: int = 4,
        lifting_channel_ratio: int = 2,
        projection_channel_ratio: int = 2,
        positional_embedding: Union[str, nn.Module] = "grid",
        non_linearity: nn.Module = F.gelu,
        conv_module: nn.Module = SpectralConv,
        debug: bool = False,
        **kwargs
    ) -> None:

        super().__init__()
        self.debug = debug
        self.n_dim = len(modes)
        self.hidden_channels = hidden_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.lifting_channel_ratio = lifting_channel_ratio
        self.lifting_channels = lifting_channel_ratio * hidden_channels
        self.projection_channel_ratio = projection_channel_ratio
        self.projection_channels = projection_channel_ratio * hidden_channels
        self.non_linearity = non_linearity
        self._modes = modes
        
        if positional_embedding == "grid":
            self.positional_embedding = GridEmbedding2D()
        else:
            raise ValueError(f"Invalid positional embedding: {positional_embedding}. Expected 'grid' or an nn.Module.")

        lifting_in_channels = self.in_channels + 2 if self.positional_embedding else self.in_channels
        self.lifting = ChannelMLP(lifting_in_channels, self.hidden_channels) 
        
        self.fno_blocks = FNOBlocks(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            modes=self.modes,
            non_linearity=non_linearity,
            conv_module=conv_module,
            n_layers=n_layers,
            debug = self.debug,
            **kwargs
        )
        
        self.projection = ChannelMLP(
            in_channels=self.hidden_channels,
            out_channels=out_channels,
            hidden_channels=self.projection_channels,
            n_layers=2,
            n_dim=self.n_dim,
            non_linearity=non_linearity,
        )
        
        if self.debug:
            self._debug_print_initialization()

    def forward(self, x: torch.Tensor, output_shape: Union[Tuple[int, ...], List[Tuple[int, ...]], None] = None, **kwargs) -> torch.Tensor:
        b, c, t, v, h, w = x.shape
        
        if self.debug:
            print(f"Input shape: (batch={b}, channels={c}, timesteps={t}, vars={v}, height={h}, width={w})")

        output_shape = [None] * self.n_layers if output_shape is None else output_shape if isinstance(output_shape, list) else [None] * (self.n_layers - 1) + [output_shape]
        
        if self.positional_embedding:
            x = self.positional_embedding(x)
        
        x = self.lifting(x)
        
        if self.debug:
            print(f"Shape after lifting: {x.shape}")
        
        for layer_idx in range(self.n_layers):
            x = self.fno_blocks(x, layer_idx)
            if self.debug:
                print(f"Shape after FNO layer {layer_idx + 1}: {x.shape}")
        
        x = self.projection(x)
        
        if self.debug:
            self._debug_print_model_parameters()
        
        return x
    
    @property
    def modes(self) -> Tuple[int, ...]:
        return self._modes

    @modes.setter
    def modes(self, modes: Tuple[int, ...]) -> None:
        self.fno_blocks.modes = modes
        self._modes = modes
    
    def _debug_print_initialization(self) -> None:
        print("[DEBUG] DQFNO Model Initialization")
        print(f"Modes: {self.modes}")
        print(f"In Channels: {self.in_channels}, Out Channels: {self.out_channels}")
        print(f"Hidden Channels: {self.hidden_channels}, Number of Layers: {self.n_layers}")
        print(f"Lifting Channels: {self.lifting_channels}, Projection Channels: {self.projection_channels}")
        print(f"Using Non-linearity: {self.non_linearity}")
        print(f"Positional Embedding: {self.positional_embedding}")
    
    def _debug_print_model_parameters(self) -> None:
        print("[DEBUG] Model Parameters:")
        for name, param in self.named_parameters():
            print(f"{name}: {param.shape}")
