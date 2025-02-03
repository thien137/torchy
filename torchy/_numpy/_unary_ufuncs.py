from torchy import Tensor
from ._unary_ufuncs_impl import * 

def log(x: Tensor):
    return Log.apply(x)

def exp(x: Tensor):
    return Exp.apply(x)

def sin(x: Tensor): 
    return Sin.apply(x)

def cos(x: Tensor):
    return Cos.apply(x)

def tan(x: Tensor):
    return Tan.apply(x)

def arcsin(x: Tensor):
    return Arcsin.apply(x)

def arccos(x: Tensor):
    return Arccos.apply(x)

def arctan(x: Tensor):
    return Arctan.apply(x)

def sqrt(x: Tensor):
    return Sqrt.apply(x)

def abs(x: Tensor):
    return Abs.apply(x)

def sign(x: Tensor):
    return Sign.apply(x)

def mean(x: Tensor):
    return Mean.apply(x)

def sum(x: Tensor, axis=None):
    return Sum.apply(x, axis=axis)