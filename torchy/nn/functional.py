import torchy 
from torchy import Tensor 
from torchy.autograd import Function, FunctionCtx

from _functional_impl import *

def relu(x: Tensor) -> Tensor:
    return Relu.apply(x)

def sigmoid(x: Tensor) -> Tensor:
    return Sigmoid.apply(x)