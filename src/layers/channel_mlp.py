import torch
from torch import nn
import torch.nn.functional as F


class ChannelMLP(nn.Module):
    """
    ChannelMLP applies an arbitrary number of 1D convolutional layers
    (with point‐wise kernels) and nonlinearity to the channel dimension
    of an input while being invariant to its spatial resolution.

    This module is designed for input tensors of shape:
        (batch, channels, time, fields, x, y)

    It works by flattening all dimensions beyond batch and channels,
    applying a series of Conv1d layers (with kernel size 1) that transform
    the channel vectors at each spatial location, and then reshaping the
    result back to (batch, out_channels, time, fields, x, y).

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int, optional
        Number of output channels; if None, defaults to in_channels.
    hidden_channels : int, optional
        Number of channels in the hidden layers; if None, defaults to in_channels.
    n_layers : int, default 2
        Number of layers in the MLP.
    n_dim : int, default 4
        Number of spatial dimensions (here: time, fields, x, y).
    non_linearity : callable, default F.gelu
        Activation function to apply between layers.
    dropout : float, default 0.0
        Dropout probability. If > 0, dropout layers are added.
    """
    def __init__(
        self,
        in_channels,
        out_channels=None,
        hidden_channels=None,
        n_layers=2,
        n_dim=4,  # For input shape (B, C, T, F, x, y), n_dim is 4.
        non_linearity=F.gelu,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.hidden_channels = in_channels if hidden_channels is None else hidden_channels
        self.non_linearity = non_linearity
        self.n_dim = n_dim  # number of extra dimensions (here: T, F, x, y)

        self.dropout = (
            nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)])
            if dropout > 0.0
            else None
        )
        
        # Use Conv1d with kernel_size=1 to act as a pointwise linear layer
        # applied to the channel dimension at each spatial location.
        self.fcs = nn.ModuleList()
        for i in range(n_layers):
            if i == 0 and i == (n_layers - 1):
                self.fcs.append(nn.Conv1d(self.in_channels, self.out_channels, kernel_size=1))
            elif i == 0:
                self.fcs.append(nn.Conv1d(self.in_channels, self.hidden_channels, kernel_size=1))
            elif i == (n_layers - 1):
                self.fcs.append(nn.Conv1d(self.hidden_channels, self.out_channels, kernel_size=1))
            else:
                self.fcs.append(nn.Conv1d(self.hidden_channels, self.hidden_channels, kernel_size=1))

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape either:
                (batch, channels, time, fields, x, y)  or
                (channels, time, fields, x, y)
        
        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, out_channels, time, fields, x, y)
        """
        reshaped = False
        size = list(x.shape)
        # If there are extra spatial dimensions (i.e. ndim > 2), flatten them.
        # (For an input (B, C, T, F, x, y), size = [B, C, T, F, x, y].)
        if x.ndim > 2:
            x = x.reshape(size[0], size[1], -1)
            reshaped = True

        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i < self.n_layers - 1:
                x = self.non_linearity(x)
            if self.dropout is not None:
                x = self.dropout[i](x)

        # If we flattened the spatial dimensions, reshape back to original.
        if reshaped:
            x = x.reshape((size[0], self.out_channels, *size[2:]))
        return x


class LinearChannelMLP(nn.Module):
    """
    LinearChannelMLP applies a sequence of linear (fully-connected) layers
    with nonlinearity and dropout to the channel dimension. It is intended for inputs of shape:
        (batch, channels, time, fields, x, y)

    In order to work on the channel vectors at each spatial location independently,
    the input is first permuted so that channels are the last dimension, the
    linear layers are applied to that last dimension, and then the tensor is permuted
    back to the original layout.

    Parameters
    ----------
    layers : list of int
        A list of layer sizes. For example, [in_channels, hidden, out_channels].
    non_linearity : callable, default F.gelu
        Activation function applied between linear layers.
    dropout : float, default 0.0
        Dropout probability.
    """
    def __init__(self, layers, non_linearity=F.gelu, dropout=0.0):
        super().__init__()
        self.n_layers = len(layers) - 1
        assert self.n_layers >= 1, "At least one linear layer is required."
        self.fcs = nn.ModuleList()
        self.non_linearity = non_linearity
        self.dropout = (
            nn.ModuleList([nn.Dropout(dropout) for _ in range(self.n_layers)])
            if dropout > 0.0
            else None
        )
        for j in range(self.n_layers):
            self.fcs.append(nn.Linear(layers[j], layers[j + 1]))

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape either:
                (batch, channels, time, fields, x, y)  or
                (channels, time, fields, x, y)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, out_channels, time, fields, x, y)
        """
        reshaped = False
        # If there's no batch dimension (i.e. input is (C, T, F, x, y)), add one.
        if x.ndim == 5:
            x = x.unsqueeze(0)
            reshaped = True

        # Permute so that channels are the last dimension:
        # from (B, C, T, F, x, y) to (B, T, F, x, y, C)
        x = x.permute(0, 2, 3, 4, 5, 1)
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i < self.n_layers - 1:
                x = self.non_linearity(x)
            if self.dropout is not None:
                x = self.dropout[i](x)
        # Permute back to (B, channels, T, F, x, y)
        x = x.permute(0, 5, 1, 2, 3, 4)
        if reshaped:
            x = x.squeeze(0)
        return x