import torchy

class Parameter(torchy.Tensor):
    """A kind of Tensor that is to be considered a module parameter.
    
    Parameters are Tensor subclasses, that have a special property
    that when used with Module classses, are automatically added to
    the list of its parameters.
    """

    def __new__(self, data=None, requires_grad=True):
        if data is None:
            data = torchy.empty(0)
        if type(data) is torchy.Tensor or type(data) is Parameter:
            data._is_param = True
            return data

        t = torchy.Tensor(data, requires_grad=True)
        t._is_param = True 
        return t 
    
    def __repr__(self):
        return "Parameter contain:\n" + super().__repr__()
