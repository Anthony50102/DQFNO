import torch

class HasegawaWatakini():
    def __init__(self):
        pass

    
    def periodic_gradient(self, input_field: torch.Tensor, dx: float, axis: int = 0) -> torch.Tensor:
        """
        Compute the gradient of a 2D array using finite differences with periodic boundary conditions.

        Args:
            input_field (torch.Tensor): Input 2D array (torch tensor).
            dx (float): The spacing between grid points.
            axis (int): Axis along which the gradient is taken.

        Returns:
            torch.Tensor: Gradient along the specified axis with periodic boundary conditions.
        """
        # Handle periodic boundary conditions
        if axis == -2:  # y-direction
            shifted_forward = torch.roll(input_field, shifts=-1, dims=-2)
            shifted_backward = torch.roll(input_field, shifts=1, dims=-2)
        elif axis == -1:  # x-direction
            shifted_forward = torch.roll(input_field, shifts=-1, dims=-1)
            shifted_backward = torch.roll(input_field, shifts=1, dims=-1)
        else:
            raise ValueError("Axis must be -2 (y) or -1 (x) for 2D data.")

        # Compute the finite difference gradient
        gradient = (shifted_forward - shifted_backward) / (2 * dx)
        return gradient

    def get_gamma_n(self, n: torch.Tensor, p: torch.Tensor, dx: float, dy_p: torch.Tensor = None, dtype: str = 'torch') -> torch.Tensor:
        """
        Compute the average particle flux (Γₙ) using the formula:
        $$
            \\Gamma_n = - \\int{\\mathrm{d^2} x \; \\tilde{n} \\frac{\partial \\tilde{\\phi}}{\\partial y}}
        $$

        Args:
            n (torch.Tensor): Density (or similar field) of shape (..., y, x).
            p (torch.Tensor): Potential (or similar field) of shape (..., y, x).
            dx (float): Grid spacing.
            dy_p (torch.Tensor, optional): Gradient of potential in the y-direction. Computed from `p` if not provided.

        Returns:
            torch.Tensor: Computed average particle flux value.
        """
        if dtype == 'numpy':
            n, p, dx = torch.from_numpy(n), torch.from_numpy(p), torch.tensor(float(dx)).float()
            if dy_p != None:
                dy_p = torch.from_numpy(dy_p)

        if dy_p is None:
            dy_p = self.periodic_gradient(p, dx=dx, axis=-2)  # Gradient in the y-direction

        # Compute the product of n and dy_p, then take the mean
        gamma_n = -torch.mean(n * dy_p, dim=(-2, -1))  # Mean over y and x dimensions
        
        if dtype == 'numpy':
            return gamma_n.numpy()
        return gamma_n