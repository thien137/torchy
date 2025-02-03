import numpy as np
import cupy as cp
import torchy
from torchy.autograd import AccumulateGradient

from typing import Optional, Tuple

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

        self.requires_grad = requires_grad
        self.grad_fn = None 
        self.grad = None

        if self.requires_grad:
            self = AccumulateGradient.apply(self)

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
    
    def __add__(self, other): 
        return torchy.add(self, other)

    def __sub__(self, other):
        return torchy.subtract(self, other)

    def __mul__(self, other):
        return torchy.multiply(self, other)
    
    def __truediv__(self, other):
        return torchy.divide(self, other)
    
    def backward(
            self, grad_output: Optional['Tensor'] = None
    ):
        if not self.requires_grad:
            raise RuntimeError("Cannot call backward on a tensor that does not require gradients.")
        
        if grad_output is None:
            if self.shape == ():
                grad_output = Tensor(1.0, device=self.device)
            else:
                raise RuntimeError("grad_output must be specified for non-scalar tensors.")
        
        if grad_output.shape != self.shape:
            raise RuntimeError(f"grad_output shape {grad_output.shape} does not match tensor shape {self.shape}.")
        
        if self.grad_fn is None:
            raise RuntimeError("Cannot call backward on a tensor that does not have a grad_fn.")
        
        self.grad_fn.backward(grad_output)
        
        if not self.grad:
            self.grad = grad_output
        else:
            self.grad += grad_output
    
    @property 
    def shape(self) -> Tuple[int]:
        return self._array.shape

    def uniform_(self, a=0.0, b=1.0) -> None:
        # TODO: Implement this later
        self._array = np.random.uniform(low=a, high=b, size=self._array.shape)
    
    def normal_(self, mean=0.0, std=1.0) -> None:
        # TODO: Implement this later
        self._array = np.random.normal(loc=mean, scale=std, size=self._array.shape)
    
    def dim(self):
        return len(self._array.shape)
    
    def size(self, dim: int):
        return self._array.shape[dim]