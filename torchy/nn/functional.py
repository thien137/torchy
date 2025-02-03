import torchy 
from torchy import Tensor 
from torchy.autograd import Function, FunctionCtx

from _functional_impl import *

def relu(x: Tensor) -> Tensor:
    return Relu.apply(x)

def sigmoid(x: Tensor) -> Tensor:
    return Sigmoid.apply(x)

def mse_loss(x: Tensor, y: Tensor, reduction: str = 'mean') -> Tensor:
    return MSE_Loss.apply(x, y, reduction)

def linear(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    return Linear.apply(x, weight, bias)