"""This file contains utilities for initializing neural network parameters."""

import math
import torchy 
from torchy import Tensor

from typing import Optional 

def calculate_gain(activation, param=None):
    match activation:
        case "linear":
            return 1
        case "tanh":
            return 5.0 / 3 
        case "relu":
            return math.sqrt(2.0)
        case "selu":
            return (
                3.0 / 4
            )
        case _:
            raise ValueError(f"Unsupported activation {activation}")

def uniform(
        tensor: Tensor,
        a: float = 0.0,
        b: float = 1.0
) -> Tensor:
    return

def normal(
        tensor: Tensor,
        mean: float = 0.0,
        std: float = 1.0,
        generator: Optional[torchy.Generator] = None
) -> Tensor:
    return tensor.normal_(mean, std, generator=generator)

def kaiming_uniform(
        tensor: Tensor
) -> Tensor:
    # TODO
    return tensor

def _calculate_fan_in_and_fan_out(tensor: Tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError(
            "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
        )
     
    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    receptive_field_size = 1
    if tensor.dim() > 2:
        for s in tensor.shape[2:]:
            receptive_field_size 