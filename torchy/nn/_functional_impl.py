import torchy 
from torchy import Tensor 
from torchy.autograd import Function, FunctionCtx

import torchy._engine_wrapper as engine
from torchy._engine_wrapper import ArrayType

from typing import List

class Relu(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, a: Tensor) -> Tensor:
        ctx.save_for_backward(a)
        
        def _forward(a: ArrayType) -> ArrayType:
            return engine.maximum(0, a)
            
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
    
class MSE_Loss(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, input: Tensor, target: Tensor, reduction: str) -> Tensor:
        ctx.save_for_backward(input, target)
        
        def _forward(input: ArrayType, target: ArrayType) -> ArrayType:
            return engine.mean((input - target) ** 2)
        
        return Tensor(_forward(input._array, target._array))

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Tensor) -> Tensor:
        input, target = ctx.saved_inputs
        
        def _backward(grad_output: ArrayType, input: ArrayType, target: ArrayType) -> ArrayType:
            return 2 * (input - target) * grad_output
        
        return Tensor(_backward(grad_output._array, input._array, target._array))
    
class Linear(Function):
    @staticmethod 
    def forward(ctx: FunctionCtx, input: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
        ctx.save_for_backward(input, weight, bias)
        
        def _forward(input: ArrayType, weight: ArrayType, bias: ArrayType) -> ArrayType:
            return engine.dot(input, weight) + bias
        
        return Tensor(_forward(input._array, weight._array, bias._array))
    
    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Tensor) -> Tensor:
        input, weight, bias = ctx.saved_inputs
        
        def _backward(grad_output: ArrayType, input: ArrayType, weight: ArrayType, bias: ArrayType) -> List[ArrayType]:
            grad_input = engine.dot(grad_output, weight.T)
            grad_weight = engine.dot(input.T, grad_output)
            grad_bias = engine.dot(engine.ones((1, grad_output.shape[0])), grad_output).T
            
            return grad_input, grad_weight, grad_bias
        
        return list(map(Tensor, _backward(grad_output._array, input._array, weight._array, bias._array)))
    