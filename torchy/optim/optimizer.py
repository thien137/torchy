import torchy
from torchy import Tensor

from typing import Any, Union, Iterable, TypeAlias

ParamsT: TypeAlias = Union[Iterable[Tensor], Iterable[dict[str, Any]]]

class Optimizer:
    
    def __init__(self, params: ParamsT, defaults) -> None:
        if isinstance(params, Tensor):
            raise TypeError(
                "params argument must be an iterable of Tensors"
            )
        
        self.param_groups: list[dict[str, Any]] = []
        
        param_groups = list(params)
        
        for param_group in param_groups:
            self.add_param_group(cast(dic, param_group))