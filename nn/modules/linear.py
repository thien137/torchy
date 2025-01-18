import math
import torchy 
from torchy import Tensor
from torchy.nn import functional as F, Parameter
from torchy.nn import init

from .module import Module

class Identity(Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__() 

    def forward(self, X: torchy.Tensor) -> torchy.Tensor:
        return X 
    
class Linear(Module):

    in_features: int 
    out_features: int 
    weight: torchy.Tensor

    def __init__(
        self,
        in_features: int, 
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features 
        self.out_features = out_features 
        self.weight = Parameter(
            torchy.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = Parameter(torchy.empty(out_features), **factory_kwargs)
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
    
    def reset_parameters(self) -> None: 
        init.kaiming_uniform(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _  = init._calculate_fan_in_and_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform(self.bias, -bound, bound)
    
    def forward(self, X: Tensor) -> Tensor:
        return F.linear(X, self.weight, self.bias)
    