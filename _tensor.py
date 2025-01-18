import numpy as np
import cupy as cp

import torchy.autograd.functional as F

from typing import Optional

class Tensor:

    VALID_DEVICES = {"cpu", "cuda"}

    def __init__(self, 
                 data, 
                 requires_grad: bool = False, 
                 device: str = "cpu", 
                 dtype = None):
        
        if device not in Tensor.VALID_DEVICES:
            raise ValueError(f"Device must be one of {Tensor.VALID_DEVICES}")

        self.device = device
        self.dtype = dtype
        self._array = self._to_backend_array(data)

        self.requires_grad = requires_grad # Required to run backward 
        self.grad_fn = None 
        self.grad = None # Set when calling 

    def _to_backend_array(self, data):
         match self.device:
            case "cpu":
                return np.array(data)
            case "cuda":
                return cp.array(data)
 
    def to_device(self, device: str):
        if device not in Tensor.VALID_DEVICES:
            raise ValueError(f"Device must be one of {Tensor.VALID_DEVICES}") 
        if self.device != device:
            self.device = device 
            self._array = self._to_backend_array(self._array)

    def __getattr__(self, name: str):
        if hasattr(self._array, name):
            attr = getattr(self._array, name)
            if callable(attr):
                def wrapper(*args, **kwargs):
                    result = attr(*args, **kwargs)
                    if isinstance(result, (np.ndarray, cp.ndarray)):
                        return Tensor(result, self.device)
                    return result 
                return wrapper
            return attr 
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __repr__(self):
        return f"tensor({repr(self._array)}, device={self.device}, requires_grad={self.requires_grad})"
    
    def __getitem__(self):
        return repr(self._array)
    
    def __setitem__(self, index: int, value: str):
        self._array[index] = value 
    
    def _binary_check_device_and_convert_input_to_tensor(method):
        def wrapper(self, other):
            if isinstance(other, Tensor):
                if self.device != other.device:
                    raise ValueError(f"Cannot operate on tensors with different devices: {self.device} vs {other.device}")
            else:
                other = Tensor(other, device=self.device)
            return method(self, other)
        return wrapper

    @_binary_check_device_and_convert_input_to_tensor
    def __add__(self, other):
        return F.Add.apply(self, other)
    
    @_binary_check_device_and_convert_input_to_tensor
    def __mul__(self, other):
        return F.Mul.apply(self, other)
    
    def backward(
            self, grad_output: Optional['Tensor'] = None
    ):
        if not self.requires_grad:
            raise RuntimeError("The tensor does not require gradients. Set requires_grad=True to track gradients.")

        if grad_output is None:
            # TODO: adjust this to cuda
            grad_output = Tensor(np.ones_like(self._array), device=self.device) # Gradient of scalar w.r.t itself

        if self.grad_fn is not None:
            self.grad_fn.backward(grad_output)

        self.grad = grad_output

        

        