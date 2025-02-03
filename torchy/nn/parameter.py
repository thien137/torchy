import torchy
from torchy import Tensor

class _ParameterMeta(type):
    def __instancecheck__(self, instance):
        if self is Parameter:
            return isinstance(instance, Tensor) and hasattr(instance, "_is_param")
        return False

class Parameter(Tensor, metaclass=_ParameterMeta):
    """A kind of Tensor that is to be considered a module parameter.
    
    Parameters are Tensor subclasses, that have a special property
    that when used with Module classses, are automatically added to
    the list of its parameters.
    """

    def __new__(self, data=None, requires_grad=True, device="cpu"):
        if data is None:
            data = torchy.empty(0, requires_grad=requires_grad, device=device)
            data._is_param = True
            return data

        t = torchy.Tensor(data, requires_grad=requires_grad, device=device)
        t._is_param = True 
        return t 
    
    def __repr__(self):
        return "Parameter contain:\n" + super().__repr__()
