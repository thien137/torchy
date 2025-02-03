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

def clip(a, min, max):
    engine = infer_engine()
    return engine.clip(a, min, max)