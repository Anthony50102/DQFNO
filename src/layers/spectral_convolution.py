from typing import List, Tuple
import torch
from torch import nn

class SpectralConv(nn.Module):
    r"""2D Fourier layer that performs FFT, applies a linear transform in Fourier space, and then
    applies an inverse FFT to return to the spatial domain.

    This layer expects an input tensor of shape:
    
        (batch, in_channels, t, vars, height, width)
    
    and produces an output tensor of shape:
    
        (batch, out_channels, t, vars, height, width)

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    vars : int
        Additional variable dimension (e.g., for multiple physical variables).
    modes : List[Tuple[int, int]]
        A list containing one tuple (modes1, modes2) that specifies the number of Fourier modes
        to keep along the height and width dimensions, respectively.
    """
    def __init__(self, in_channels: int, out_channels: int, vars: int, modes: List[Tuple[int, int]]) -> None:
        super().__init__()
        # TODO - Implement this
        modes1 = 8  
        modes2 = 8  
        self.vars: int = vars
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.modes1: int = modes1  
        self.modes2: int = modes2  

        # Initialize learnable complex-valued weights with a scaling factor.
        # Note: The weights do NOT depend on time, so they are shared across the t dimension.
        self.scale: float = 1 / (self.in_channels * self.out_channels)
        self.weights1: nn.Parameter = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, vars, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2: nn.Parameter = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, vars, modes1, modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        r"""Performs complex multiplication using Einstein summation.

        The multiplication is applied over the input channel dimension.
        (Because the weight tensor does not have a time index, it is automatically broadcast
         along the time dimension of the input.)

        Parameters
        ----------
        input : torch.Tensor
            Input tensor of Fourier coefficients with shape
            (batch, in_channels, t, vars, x, y).
        weights : torch.Tensor
            Weight tensor with shape (in_channels, out_channels, vars, x, y).

        Returns
        -------
        torch.Tensor
            Output tensor after multiplication with shape
            (batch, out_channels, t, vars, x, y).
        """
        # Notice that the weight tensor has no time index.
        # Its indices are broadcast over the time dimension of the input.
        return torch.einsum("b i t v x y, i o v x y -> b o t v x y", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Forward pass of the SpectralConv layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, in_channels, t, vars, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, out_channels, t, vars, height, width).
        """
        b, c, t, v, h, w = x.shape

        # Compute the Fourier coefficients along the spatial dimensions (height and width).
        # The FFT is applied on the last two dimensions.
        x_ft: torch.Tensor = torch.fft.rfft2(x, dim=(-2, -1))  # Shape: (b, c, t, v, h, w//2 + 1)

        # Allocate an output tensor in Fourier space.
        out_ft: torch.Tensor = torch.zeros(b, self.out_channels, t, v, h, w // 2 + 1,
                                             dtype=torch.cfloat, device=x.device)

        # Multiply the low-frequency parts with the corresponding weight tensors.
        # The slicing applies to the spatial dimensions only.
        out_ft[:, :, :, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :, :, :self.modes1, :self.modes2], self.weights1
        )
        out_ft[:, :, :, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :, :, -self.modes1:, :self.modes2], self.weights2
        )

        # Apply the inverse FFT to convert back to the spatial domain.
        x_out: torch.Tensor = torch.fft.irfft2(out_ft, s=(h, w))

        return x_out
