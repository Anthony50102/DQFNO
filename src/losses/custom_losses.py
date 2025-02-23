import warnings
from typing import List, Callable, Tuple, Any
import torch

class MultiTaskLoss(object):
    """
    Computes a multi-task loss where:
      - The absolute loss is the weighted sum of individual losses.
      - The relative loss shows each loss's contribution as a percentage of the total.
    
    New Feature (multi_output):
      If your model produces multiple outputs (e.g. predicted state and derived value),
      you can pass a list of input selectors (one per loss function) that extract the 
      required inputs for each loss function from the full outputs.
      
      Each selector should be a callable that accepts (y_pred, y) and returns a tuple 
      of arguments to be passed to the loss function.
      
      For example, if your model outputs (state, derived) and you want a loss on the state:
          selector_state = lambda y_pred, y: (y_pred[0], y[0])
    
    Args:
      loss_functions: List of loss functions.
      scales: List of floats indicating the relative scale (weight) for each loss.
      multi_output: If True, indicates that each loss function uses a selector to pick its inputs.
      input_selectors: List of callables (one per loss) that extract the desired inputs.
                       If None and multi_output is True, a default selector is used that returns (y_pred, y).
    """
    def __init__(self, 
                 loss_functions: List[Callable], 
                 scales: List[float], 
                 multi_output: bool = False,
                 input_selectors: List[Callable[[Any, Any], Tuple]] = None):
        
        if len(loss_functions) != len(scales):
            raise ValueError("The number of loss functions must match the number of scales.")
        
        self.loss_functions = loss_functions
        
        total_scale = sum(scales)
        if abs(total_scale - 1.0) > 1e-6:
            warnings.warn("Scales do not sum to 1. Normalizing scales.")
            self.scales = [s / total_scale for s in scales]
        else:
            self.scales = scales

        self.multi_output = multi_output
        if self.multi_output:
            if input_selectors is not None:
                if len(input_selectors) != len(loss_functions):
                    raise ValueError("The number of input selectors must match the number of loss functions.")
                self.input_selectors = input_selectors
            else:
                # Default: no selection; pass the full y_pred and y
                self.input_selectors = [lambda y_pred, y: (y_pred, y)] * len(loss_functions)
        else:
            # When not using multi_output, selectors are not used.
            self.input_selectors = None

    def abs(self, y_pred, y):
        """Compute the weighted sum (absolute loss) of all individual losses."""
        total_loss = 0
        for idx, (loss_fn, scale) in enumerate(zip(self.loss_functions, self.scales)):
            if self.multi_output:
                # Use the provided selector to extract relevant inputs.
                args = self.input_selectors[idx](y_pred, y)
                loss_value = loss_fn(*args)
            else:
                loss_value = loss_fn(y_pred, y)
            total_loss += scale * loss_value
        return total_loss

    def rel(self, y_pred, y):
        """
        Compute the relative contribution of each loss to the total loss.
        
        Returns a tensor where each element is in [0, 1] and the sum is 1.
        If the total loss is zero, a tensor of zeros is returned.
        """
        loss_values = []
        for idx, (loss_fn, scale) in enumerate(zip(self.loss_functions, self.scales)):
            if self.multi_output:
                args = self.input_selectors[idx](y_pred, y)
                loss_val = scale * loss_fn(*args)
            else:
                loss_val = scale * loss_fn(y_pred, y)
            loss_values.append(loss_val)
        
        losses = torch.stack(loss_values)
        total_loss = losses.sum()
        if total_loss == 0:
            return torch.zeros_like(losses)
        return losses / total_loss

    def __call__(self, y_pred, y):
        """Return the absolute weighted sum loss."""
        return self.abs(y_pred, y)
