"""This file contains utilities for initializing neural network parameters."""

import math
from torchy import Tensor

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

def _calculate_correct_fan(tensor: Tensor, mode: str):
    mode = mode.lower()
    valid_modes = ["fan_in", "fan_out"]
    if mode not in valid_modes:
        raise ValueError(f"Mode must be one of {valid_modes}")
    
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == "fan_in" else fan_out

def uniform_(
        tensor: Tensor,
        a: float = 0.0,
        b: float = 1.0
) -> Tensor:
    tensor.uniform_(a, b)

def normal_(
        tensor: Tensor,
        mean: float = 0.0,
        std: float = 1.0,
) -> Tensor:
    tensor.normal_(mean, std)

def kaiming_uniform(
        tensor: Tensor,
        a=0,
        mode: str = "fan_in",
        nonlinearity: str = "relu"
) -> Tensor:
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    
    tensor.uniform_(-bound, bound)
    
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
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out