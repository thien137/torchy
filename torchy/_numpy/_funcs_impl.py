from torchy import Tensor 
from torchy.autograd import Function, FunctionCtx

import torchy._engine_wrapper as engine
from torchy._engine_wrapper import ArrayType

class Clip(Function):
    @staticmethod 
    def forward(ctx: FunctionCtx, a: Tensor, min_val: float, max_val: float) -> Tensor:
        ctx.save_for_backward(a, min_val, max_val)
        
        def _forward(a: ArrayType, min_val: float, max_val: float) -> ArrayType:
            return engine.clip(a, min_val, max_val)
        
        return Tensor(_forward(a._array, min_val, max_val))
    
    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Tensor) -> Tensor:
        a, min_val, max_val = ctx.saved_inputs
        
        def _backward(grad_output: ArrayType, a: ArrayType) -> ArrayType:
            return grad_output * (a >= min_val) * (a <= max_val)
        
        return list(map(Tensor, _backward(grad_output._array, a._array)))