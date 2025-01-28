# from .parameter import Parameter
from torchy import Tensor

__all__ = [
    "Parameter",
    "backward"
]

def backward(tensor: Tensor) -> None:
    """Backpropagate gradients through the computational graph."""
    if tensor.grad_fn is None:
        raise RuntimeError("Cannot call backward on a tensor that does not require gradients.")
    grad = tensor.grad_fn.backward(tensor.grad)
    tensor.grad = grad
    if tensor.grad_fn is not None:
        tensor.grad_fn.backward(grad)

