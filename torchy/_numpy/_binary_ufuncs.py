from torchy import Tensor
from ._binary_ufuncs_impl import * 

# Functional interface for _binary_ufuncs_impl
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
def add(x: Tensor, y: Tensor):
    return Add.apply(x, y)

@_binary_check_device_and_convert_input_to_tensor
def subtract(x: Tensor, y: Tensor):
    return Sub.apply(x, y)

@_binary_check_device_and_convert_input_to_tensor
def multiply(x: Tensor, y: Tensor):
    return Mul.apply(x, y)

@_binary_check_device_and_convert_input_to_tensor
def divide(x: Tensor, y: Tensor):
    return Div.apply(x, y)

@_binary_check_device_and_convert_input_to_tensor
def pow(x: Tensor, y: Tensor):
    return Pow.apply(x, y)

@_binary_check_device_and_convert_input_to_tensor
def maximum(x: Tensor, y: Tensor):
    return Maximum.apply(x, y)

@_binary_check_device_and_convert_input_to_tensor
def minimum(x: Tensor, y: Tensor):
    return Minimum.apply(x, y)

@_binary_check_device_and_convert_input_to_tensor
def equal(x: Tensor, y: Tensor):
    return Equal.apply(x, y)

@_binary_check_device_and_convert_input_to_tensor
def dot(x: Tensor, y: Tensor):
    return Dot.apply(x, y)
