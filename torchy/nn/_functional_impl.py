import torchy 
from torchy import Tensor 
from torchy.autograd import Function, FunctionCtx

import _engine_wrapper as engine
from _engine_wrapper import ArrayType

class Relu(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, a: Tensor) -> Tensor:
        ctx.save_for_backward(a)
        
        def _forward(a: ArrayType) -> ArrayType:
            return ctx.engine.maximum(0, a)
            
        return Tensor(_forward(a._array))
    
    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Tensor) -> Tensor:
        a, = ctx.saved_inputs

        def _backward(grad_output: ArrayType, a: ArrayType) -> ArrayType:
            grad_a = torchy.ones(a.shape)
            grad_a[a <= 0] = 0
            return grad_output * grad_a
        
        return Tensor(_backward(grad_output._array, a._array))
    
class Sigmoid(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, a: Tensor) -> Tensor:
        ctx.save_for_backward(a)
        
        def _forward(a: ArrayType) -> ArrayType:
            a = engine.clip(a, -500, 500)
            return 1 / (1 + engine.exp(-a))
        
        return Tensor(_forward(a._array))
    
    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Tensor) -> Tensor:
        a, = ctx.saved_inputs
        
        def _backward(grad_output: ArrayType, a: ArrayType) -> ArrayType:
            a = engine.clip(a, -500, 500)
            sig = 1 / (1 + engine.exp(-a))
            return grad_output * sig * (1 - sig)
        
        return Tensor(_backward(grad_output._array, a._array))