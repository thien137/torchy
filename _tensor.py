import numpy as np
import cupy as cp

from typing import Optional

class Tensor():
    def __init__(self, data, device="cpu"):
        if device == "cpu":
            self._array = np.array(data)
        elif device == "cuda":
            self._array = cp.array(data)
        else:
            raise NotImplementedError(f"Device '{device}' is not implemented.")

        self.device = device 
        self.requires_grad = False 
        self.grad_fn = None 
        self.grad = None
        
    def __getattr__(self, name):
        """Delegate attribute access to the underlying array.

        Args:
            name (_type_): _description_
        """
        if hasattr(self, name):
            return getattr(self, name)
        raise AttributeError(f"'{type(self).__name__} object has no attribute '{name}")

    def __repr__(self):
        return repr(self._array)
    
    def __getitem__(self, idx):
        return self._array[idx]
    
    def __setitem__(self, idx, val):
        self._array[idx] = val
    
    def to(self, device):
        # TODO
        pass

    def backward(self, dY=None):
        # TODO:
        pass
