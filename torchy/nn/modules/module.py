import torchy
from torchy import Tensor
from torchy.nn import Parameter 

from typing import Optional, Union, Callable, Any, Iterator

class Module:
    """Base class for all neural network modules."""

    _version: int = 1

    training: bool 
    _parameters: dict[str, Optional[Parameter]]
    _buffers: dict[str, Optional[torchy.Tensor]]

    def __init__(self):
        super().__setattr__("training", True)
        super().__setattr__("_parameters", {})
        super().__setattr__("_modules", {})
        
        super().__init__()

    def _forward_unimplemented(self, *input: Any) -> None:
        raise NotImplementedError(
            f'Module {type(self).__name__} does not implement forward()'
        )
        
    forward: Callable[..., Any] = _forward_unimplemented
    
    def __call__(self, *input: Any, **kwargs: Any) -> Any:
        return self.forward(*input, **kwargs)

    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
        self._parameters[name] = param

    def add_module(self, name: str, module: Optional['Module']) -> None:
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
    
    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        for _name, param in self.named_parameters(recurse=recurse):
            yield param

    def modules(self) -> Iterator['Module']:
        for _, module in self.named_modules():
            yield module

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[tuple[str, Parameter]]:
        gen = self.named_members(
            lambda module: module._parameters.items(),
            prefix=prefix,
            recurse=recurse,
        )
        for elem in gen:
            yield elem

    def named_members(
        self, get_members_fn: Callable[['Module'], Iterator], prefix: str = "", recurse: bool = True
    ) -> Iterator[tuple[str, Any]]:
        memo = set()
        modules = (
            self.named_modules(prefix=prefix) if recurse else [(prefix, self)]
        )

        for module_prefix, module in modules:
            members = get_members_fn(module)
            for k, v in members:
                if v is None or v in memo:
                    continue
                memo.add(v)
                name = module_prefix + ("." if module_prefix else "") + k
                yield name, v
    
    def named_modules(
        self, memo: Optional[set['Module']] = None, prefix: str = "", remove_duplicate: bool = True
    ) -> Iterator[tuple[str, 'Module']]:
        if memo is None:
            memo = set() 
        if self not in memo:
            if remove_duplicate:
                memo.add(self)
            yield prefix, self
            for name, module in self._modules.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ("." if prefix else "") + name
                yield from module.named_modules(memo, submodule_prefix, remove_duplicate)

    def get_submodule(self, target: str) -> torchy.Tensor:
        if target == "":
            return self
        
        atoms: list[str] = target.split(".")
        mod: torchy.nn.Module = self

        for atom in atoms:
            if not hasattr(mod, atom):
                raise AttributeError(
                    mod._get_name() + " has no attribute `" + atom + "`"
                )
            mod = getattr(mod, atom)

            if not isinstance(mod, torchy.nn.Module):
                raise AttributeError("`" + atom + "` is not an " "nn.Module")
        
        return mod
    
    def _get_name(self):
        return self.__class__.__name__

    def __setattr__(self, name: str, value: object) -> None:
        if isinstance(value, Parameter):
            self.register_parameter(name, value)
        elif isinstance(value, Module):
            self.add_module(name, value)
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name: str) -> Union[Parameter, "Module"]:
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