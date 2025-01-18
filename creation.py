import torchy 
import numpy as np

from typing import Tuple

def empty(size: Tuple[int, ...], requires_grad=False, device=None):
    if not device:
        device = torchy.get_default_device()
    return torchy.Tensor(np.empty(size), requires_grad=requires_grad, device=device)

def zeros(size, requires_grad=False, device=None):
    if not device:
        device = torchy.get_default_device()
    return torchy.Tensor(np.zeros(size), requires_grad=requires_grad, device=device)

def ones(size: Tuple[int, ...], requires_grad=False, device=None):
    if not device:
        device = torchy.get_default_device()
    return torchy.Tensor(np.ones(size), requires_grad=requires_grad, device=device)