from ._utils import infer_engine

from typing import Tuple

def empty(size: Tuple[int, ...], device=None):
    engine = infer_engine()
    return engine.empty(size, device=device)

def zeros(size: Tuple[int, ...], device=None):
    engine = infer_engine()
    return engine.zeros(size, device=device)

def ones(size: Tuple[int, ...], device=None):
    engine = infer_engine()
    return engine.ones(size, device=device)

def zeros_like(a):
    engine = infer_engine()
    return engine.zeros_like(a)

def flatten(a):
    engine = infer_engine()
    return engine.flatten(a)

def ones_like(a):
    engine = infer_engine()
    return engine.ones_like(a)

def clip(a, min, max):
    engine = infer_engine()
    return engine.clip(a, min, max)