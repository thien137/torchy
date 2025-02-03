import torchy
from torchy import Tensor

from typing import Any, Union, Iterable, TypeAlias

ParamsT: TypeAlias = Union[
    Iterable[dict[str, Any]],
    Iterable[tuple[str, Tensor]]
]

class Optimizer:
    
    def __init__(self, params: ParamsT, defaults: dict[str, Any]) -> None:
        if isinstance(params, Tensor):
            raise TypeError(
                "params argument must be an iterable of Tensors"
            )
        
        self.params = list(params)
        self.defaults = defaults

    def step(self):
        raise NotImplementedError("step() must be implemented by subclass")