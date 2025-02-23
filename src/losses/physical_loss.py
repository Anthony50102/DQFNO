"""
physical_losses.py contains code to compute physically derived objective
functions for training Neural Operators. 

By default, losses expect arguments y_pred (model predictions) and y (ground y.)
"""
import torch
from ..functions.hw import HasegawaWatakini
from .finite_diff import laplacian_2d

class PoissonResidual(object):
    """
    Computes the Poisson Residual of the states
    """
    # TODO - This aint't gonna work figure out a better way
    def __init__(self, hw: HasegawaWatakini, scale: float = 1.0):
        super().__init__()
        self.hw = hw
        self.scale = scale
    
    def abs(self, y_pred: torch.tensor, y: torch.tensor):
        # laplace_phi = self.hw.periodic_gradient(self.hw.periodic_gradient(y_pred[0,0,0,2,:,:], dx=self.hw.dx, axis=-2), dx=self.hw.dx, axis=-2)
        laplace_phi = laplacian_2d(y_pred[0,0,0,2,:,:], h=self.hw.dx) 
        abs_diff = torch.square(torch.abs(y_pred[0,0,0,1,:,:] - laplace_phi))
        return self.scale * abs_diff

    def rel(self, y_pred: torch.tensor, y: torch.tensor):
        pass

    def __call__(self, y_pred: torch.tensor, y: torch.tensor):
        return torch.mean(self.abs(y_pred, y))
