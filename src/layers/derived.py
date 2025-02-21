import torch
from typing import List
from torch import nn
from torch.func import vmap
from ..functions.hw import HasegawaWatakini
from .channel_mlp import LinearChannelMLP

class DerivedMLP(nn.Module):
    """
    Module for taking in input state computing derived quanities,
    using predicted next state computing derived quanities and then
    using linear layer (or in future conv) to inform/compute future
    derived quanities
    """
    def __init__(self, layers: List[int], dx: float, **kwargs):
        """
        layer: List[int], layers of the MLP
        """
        super().__init__(**kwargs)
        self.hw = HasegawaWatakini(dx=dx)
        self.mlp = LinearChannelMLP(
            layers=layers
            )
        self.input_derived: torch.tensor = None
    
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
    
    def forward(self, x: torch.tensor):
        '''
        if there is no input state compute and store
        '''
        if self.input_derived == None:
            self.input_derived = self._compute_derived(x)
            return
        
        # Compute predicted state derived 
        x_derived = self._compute_derived(x)
        x = torch.cat((self.input_derived.flatten(),x_derived.flatten()))
        x = self.mlp(x)
        return  x

