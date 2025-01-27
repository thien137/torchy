import torchy
from torchy import Tensor
from torchy.nn import Parameter 

from typing import Optional, Union, Callable, Any

class Module:
    """Base class for all neural network modules."""

    _version: int = 1

    training: bool 
    _parameters: dict[str, Optional[Parameter]]
    _buffers: dict[str, Optional[torchy.Tensor]]

    def __init__(self):
        training: bool = super().__setattr__("training", True)
        _parameters: dict[str, Optional[Parameter]] = super().__setattr__("parameters", {})
        _modules: dict[str, Optional[Module]] = super().__setattr__("modules", {})

    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
        if param is None:
            self._parameters[name] = None 
        elif not isinstance(param, Parameter):
            raise TypeError(
                f"cannot assign '{torchy.typename(param)}' object to parameter '{name}' "
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

    def _forward_unimplemented(self, *input: Any) -> None:
        raise NotImplementedError(
            f'Module {type(self).__name__} does not implement forward()'
        )
        
    forward: Callable[..., Any] = _forward_unimplemented
    
    def __call___(self, *input: Any, **kwargs: Any) -> Any:
        return self.forward(*input, **kwargs)

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
        module_path, _, param_name = target.rpartition(".")

        mod: torchy.nn.Module = self.get_submodule(module_path)

        if not hasattr(mod, param_name):
            raise AttributeError(
                mod._get_name() + " has no attribute `" + param_name + "`"
            )

        param: torchy.nn.Parameter = getattr(mod, param_name)

        if not isinstance(param, torchy.nn.Parameter):
            raise AttributeError("`" + param_name + "` is not an " "nn.Parameter")

        return param

    def get_submodule(self, target: str) -> torchy.Tensor:
        module_path, _, buffer_name = target.rpartition(".")

        mod: torchy.nn.Module = self.get_submodule(module_path)

        if not hasattr(mod, buffer_name):
            raise AttributeError(
                mod._get_name() + " has no attribute `" + buffer_name + "`"
            )

        buffer: torchy.Tensor = getattr(mod, buffer_name)

        if buffer_name not in mod._buffers:
            raise AttributeError("`" + buffer_name + "` is not a buffer")

        return buffer
    
    def _get_name(self):
        return self.__class__.__name__

    def __setattr__(self, name: str, value: object) -> None:
        def remove_from(*dicts_or_sets):
            for d in dicts_or_sets:
                if name in d:
                    if isinstance(d, dict):
                        del d[name]
                    else:
                        d.remove(name)
        
        params = self.__dict__.get('_parameters')
        if isinstance(value, Parameter):
            if params is None:
                raise AttributeError(
                    "cannot assign parameters before Module.__init__() call"
                )
            remove_from(self.__dict__, self._parameters)
            self.register_parameter(name, value)
        elif params is not None and name in params:
            if value is not None:
                raise TypeError("cannot assign to parameter")
            self.register_parameter(name, value)
        else:
            modules = self.__dict__.get('_modules')
            if isinstance(value, Module):
                if modules is None:
                    raise AttributeError(
                        "cannot assign module before Module.__init__() call"
                    )
                remove_from(self.__dict__, self._modules)
                self.add_module(name, value)
            elif modules is not None and name in modules:
                if value is not None:
                    raise TypeError("cannot assign to module")
                self.add_module(name, value)
            else:
                object.__setattr__(self, name, value)
                
    def __getattr__(self, name: str) -> Union[Tensor, "Module"]:
        if "_parameters" in self.__dict__:
            _parameters = self.__dict__["_parameters"]
            if name in _parameters:
                return _parameters[name]
        if "_modules" in self.__dict__:
            modules = self.__dict__["_modules"]
            if name in modules:
                return modules[name]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )