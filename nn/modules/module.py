import torchy
from torchy.nn import Parameter 

from typing import Optional

class Module:
    """Base class for all neural network modules."""

    _version: int = 1

    training: bool 
    _parameters: dict[str, Optional[Parameter]]
    _buffers: dict[str, Optional[torchy.Tensor]]

    def __init__(self):
        training: bool = False 
        _parameters: dict[str, Optional[Parameter]] = {}
        _buffers: dict[str, Optional[torchy.Tensor]] = {}
        _modules: dict[str, Optional[Module]] = {}

    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
        if param is None:
            self._parameters[name] = None 
        elif not isinstance(param, Parameter):
            raise TypeError(
                f"cannot assign '{torch.typename(param)}' object to parameter '{name}' "
                "(torch.nn.Parameter or None required)"
            )
        elif param.grad_fn:
            raise ValueError(
                f"Cannot assign non-leaf Tensor to parameter '{name}'. Model "
                f"parameters must be created explicitly. To express '{name}' "
                "as a function of anotehr Tensor, computer the value in "
                "the forward() method."
            )
        else:
            self._parameters[name] = param

    def add_module(self, name: str, module: Optional['Module']) -> None:
        if not isinstance(module, Module) and module is not None:
            raise TypeError(f"{module} is not a Module subclass")
        elif not isinstance(name, str):
            raise TypeError(
                f"module name should be a string. Got {name}"
            )
        elif hasattr(self, name) and name not in self._modules:
            raise KeyError(f"attribute {name} already exists")
        else:
            self._modules[name] = module

    def get_parameter(self, target: str) -> Parameter:
        pass

    def get_buffer(self, target: str) -> torchy.Tensor:
        pass 

    