import torch
from typing import List
from torch import nn
from torch.func import vmap
from ..functions.hw import HasegawaWatakini
from .channel_mlp import ChannelMLP

class DerivedMLP(nn.Module):
    """
    Module for taking in input state computing derived quanities,
    using predicted next state computing derived quanities and then
    using linear layer (or in future conv) to inform/compute future
    derived quanities
    """
    def __init__(self, dx: float = None, type:str = 'derived'):
        """
        layer: List[int], layers of the MLP
        """
        super().__init__()
        self.hw = HasegawaWatakini(dx=dx)
        self.mlp = ChannelMLP(
            in_channels=2,
            hidden_channels=4,
            out_channels=1
            )
        self.input_derived: torch.tensor = None
        self._type = type # Determine what type of method we are going to use to predict out new outputs
    
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
        x_derived = self._compute_derived(x)
        if self._type == 'derived':
            x = torch.cat((self.input_derived.unsqueeze(0),x_derived.unsqueeze(0)), dim=1) # Shape (B, 2, D)
            x = self.mlp(x).squeeze(1)
            x_derived = x
        return  x_derived # (B, D)

