import numpy as np
import cupy as cp 

import torchy

from types import ModuleType
from typing import Union, List

ArrayType = Union[np.ndarray, cp.ndarray]

def infer_engine_from_arrays(*a: List[ArrayType]) -> Union[ModuleType, ModuleType]:
    for arr in a:
        if isinstance(arr, np.ndarray):
            return np
        elif isinstance(arr, cp.ndarray):
            return cp
    raise ValueError("Could not infer engine from arrays")
    
def infer_engine() -> ModuleType:
    match torchy.get_default_device():
        case "cpu":
            return np
        case "cuda":
            return cp