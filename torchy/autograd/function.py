from typing import Any, Tuple, List, Union

# Built to resemble:
# https://github.com/pytorch/pytorch/blob/main/torch/autograd/function.py#L320

class FunctionCtx:
    def __init__(self):
        self.saved_inputs = ()

    def save_for_backward(self, *args):
        self.saved_inputs = (*args,)
        
class BackwardFunction:
    def __init__(self):
        self.ctx = FunctionCtx()

    def backward(self, grad_output):
        # Traverse gradient tree backwards
        grad_inputs = self._forward_cls.backward(self.ctx, grad_output)
        if not isinstance(grad_inputs, list):
            grad_inputs = [grad_inputs]
        for i in range(len(self.ctx.saved_inputs)):
            if self.ctx.saved_inputs[i].requires_grad:
                self.ctx.saved_inputs[i].backward(grad_inputs[i])

class FunctionMeta(type):
    """Function metaclass -> Function class factory.
    This metaclass sets up the following properties:
        _backward_cls: The Function class corresponding to the differentiated
            version of this function (which is generated on the fly by this
            metaclass)."""
    def __init__(cls, name, bases, attrs):
        backward_fn = type(
            name + "Backward", (BackwardFunction,), {"_forward_cls": cls}
        )

        cls._backward_cls = backward_fn

        super().__init__(name, bases, attrs)

class Function(metaclass=FunctionMeta):

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            f"{self.__class__} should not be instantiated."
        ) 
    
    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            "Please use 'apply()' function to call autograd."
        )

    @staticmethod
    def forward(*args, **kwargs):
        raise NotImplementedError(
            "You must implement the forward function for custom autograd.Function"
        )

    @staticmethod 
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise NotImplementedError(
            """You must implement the backward function for custom autograd.Function"""
        )

    @classmethod
    def apply(cls, *inputs) -> Tuple:
        # Create a context object to store information for backward
        backward_fn = cls._backward_cls()

        outputs = cls.forward(backward_fn.ctx, *inputs)

        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        for output in outputs:
            output.requires_grad = any(x.requires_grad for x in inputs)

            if output.requires_grad:
                output.grad_fn = backward_fn

        return outputs[0] if len(outputs) == 1 else outputs