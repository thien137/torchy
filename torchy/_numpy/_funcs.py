from torchy import Tensor
import torchy._engine_wrapper as engine

from ._funcs_impl import *

from typing import Tuple

def empty(size: Tuple[int, ...], requires_grad=False, device=None):
    return Tensor(engine.empty(size), requires_grad=requires_grad, device=device)

def zeros(size, requires_grad=False, device=None):
    return Tensor(engine.zeros(size), requires_grad=requires_grad, device=device)

def ones(size: Tuple[int, ...], requires_grad=False, device=None):
    return Tensor(engine.ones(size), requires_grad=requires_grad, device=device)

def clip(a, min, max):
    return Clip.apply(a, min, max)