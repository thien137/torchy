from ._utils import infer_engine

from typing import Tuple

def empty(size: Tuple[int, ...], requires_grad=False, device=None):
    engine = infer_engine()
    return engine.empty(size, requires_grad=requires_grad, device=device)

def zeros(size, requires_grad=False, device=None):
    engine = infer_engine()
    return engine.zeros(size, requires_grad=requires_grad, device=device)

def ones(size: Tuple[int, ...], requires_grad=False, device=None):
    engine = infer_engine()
    return engine.ones(size, requires_grad=requires_grad, device=device)

def clip(a, min, max):
    engine = infer_engine()
    return engine.clip(a, min, max)