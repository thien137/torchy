import numpy as np
import cupy as cp 

import torchy

from types import ModuleType
from typing import Union

ArrayType = Union[np.ndarray, cp.ndarray]

def infer_engine_from_array(a: ArrayType) -> Union[ModuleType, ModuleType]:
    if isinstance(a, np.ndarray):
        return np 
    elif isinstance(a, cp.ndarray):
        return cp
    else:
        raise RuntimeError(f"Did not expect underlying datatype {type(a)}")
    
def infer_engine() -> ModuleType:
    match torchy.get_default_device():
        case "cpu":
            return np
        case "cuda":
            return cp