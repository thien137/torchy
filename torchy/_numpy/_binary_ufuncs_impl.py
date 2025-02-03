from torchy import Tensor 
from torchy.autograd import Function, FunctionCtx

import torchy._engine_wrapper as engine
from torchy._engine_wrapper import ArrayType

from typing import List
    
class Add(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)
        
        def _forward(a: ArrayType, b: ArrayType) -> ArrayType:
            return a + b
        
        return Tensor(_forward(a._array, b._array))

    @staticmethod 
    def backward(ctx: FunctionCtx, grad_output: Tensor) -> List[Tensor]:
        
        def _backward(grad_output: ArrayType) -> ArrayType:
            return grad_output, grad_output
        
        return list(map(Tensor, _backward(grad_output._array)))

class Mul(Function):
    @staticmethod 
    def forward(ctx: FunctionCtx, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)
        
        def _forward(a: ArrayType, b: ArrayType) -> ArrayType:
            return a * b
        
        return Tensor(_forward(a._array, b._array))
    
    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Tensor) -> Tensor:
        a, b = ctx.saved_inputs
        
        def _backward(grad_output: ArrayType, a: ArrayType, b: ArrayType) -> List[ArrayType]:
            return grad_output * b, grad_output * a

        return list(map(Tensor, _backward(grad_output._array, a._array, b._array)))

class Sub(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)
        
        def _forward(a: ArrayType, b: ArrayType) -> ArrayType:
            return a - b

        return Tensor(_forward(a._array, b._array))

    @staticmethod 
    def backward(ctx: FunctionCtx, grad_output: Tensor) -> Tensor:
        
        def _backward(grad_output: ArrayType) -> ArrayType:
            return grad_output, -grad_output
        
        return list(map(Tensor, _backward(grad_output._array)))
    
class Div(Function):
    @staticmethod 
    def forward(ctx: FunctionCtx, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)
        
        def _forward(a: ArrayType, b: ArrayType) -> ArrayType:
            return a / b
        
        return Tensor(_forward(a._array, b._array))
        
    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Tensor) -> Tensor:
        a, b = ctx.saved_inputs
        
        def _backward(grad_output: ArrayType, a: ArrayType, b: ArrayType) -> ArrayType:
            grad_a = grad_output / b
            grad_b = -grad_output * (a / b**2)
            
            return grad_a, grad_b

        return list(map(Tensor, _backward(grad_output._array, a._array, b._array)))

class Pow(Function):
    @staticmethod 
    def forward(ctx: FunctionCtx, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)
        
        def _backward(a: ArrayType, b: ArrayType) -> ArrayType:
            return a ** b
        
        return Tensor(_backward(a._array, b._array))
    
    @staticmethod 
    def backward(ctx: FunctionCtx, grad_output: Tensor) -> Tensor:
        a, b = ctx.saved_inputs 
        
        def _backward(grad_output: ArrayType, a: ArrayType, b: ArrayType) -> List[ArrayType]:
            grad_a = grad_output * b * a ** (b - 1)
            grad_b = grad_output * a ** b * engine.log(a)
            
            return grad_a, grad_b
        
        return list(map(Tensor, _backward(grad_output._array, a._array, b._array)))
    
class Maximum(Function):
    @staticmethod 
    def forward(ctx: FunctionCtx, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)
        
        def _forward(a: ArrayType, b: ArrayType) -> ArrayType:
            return engine.maximum(a, b)
        
        return Tensor(_forward(a._array, b._array))
    
    @staticmethod 
    def backward(ctx: FunctionCtx, grad_output: Tensor) -> Tensor:
        a, b = ctx.saved_inputs 
        
        def _backward(grad_output: ArrayType, a: ArrayType, b: ArrayType) -> List[ArrayType]:
            grad_a = grad_output * (a > b)
            grad_a += grad_a * (a == b) * 0.5
            grad_b = grad_output * (b > a)
            grad_b += grad_b * (a == b) * 0.5
            
            return grad_a, grad_b
        
        return list(map(Tensor, _backward(grad_output._array, a._array, b._array)))
    
class Minimum(Function):
    @staticmethod 
    def forward(ctx: FunctionCtx, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)
        
        def _forward(a: ArrayType, b: ArrayType) -> ArrayType:
            return engine.minimum(a, b)
        
        return Tensor(_forward(a._array, b._array))
    
    @staticmethod 
    def backward(ctx: FunctionCtx, grad_output: Tensor) -> Tensor:
        a, b = ctx.saved_inputs 
        
        def _backward(grad_output: ArrayType, a: ArrayType, b: ArrayType) -> List[ArrayType]:
            grad_a = grad_output * (a < b)
            grad_a += grad_a * (a == b) * 0.5
            grad_b = grad_output * (b < a)
            grad_b += grad_b * (a == b) * 0.5
            
            return grad_a, grad_b
        
        return list(map(Tensor, _backward(grad_output._array, a._array, b._array)))
    
class Equal(Function):
    @staticmethod 
    def forward(ctx: FunctionCtx, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)
        
        def _forward(a: ArrayType, b: ArrayType) -> ArrayType:
            return engine.equal(a, b)
        
        return Tensor(_forward(a._array, b._array))
    
    @staticmethod 
    def backward(ctx: FunctionCtx, grad_output: Tensor) -> Tensor:
        a, b = ctx.saved_inputs 
        
        def _backward(grad_output: ArrayType, a: ArrayType, b: ArrayType) -> List[ArrayType]:
            return grad_output, grad_output
        
        return list(map(Tensor, _backward(grad_output._array, a._array, b._array)))
    
class Dot(Function):
    @staticmethod 
    def forward(ctx: FunctionCtx, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)
        
        def _forward(a: ArrayType, b: ArrayType) -> ArrayType:
            return engine.dot(a, b)
        
        return Tensor(_forward(a._array, b._array))
    
    @staticmethod 
    def backward(ctx: FunctionCtx, grad_output: Tensor) -> Tensor:
        a, b = ctx.saved_inputs 
        
        def _backward(grad_output: ArrayType, a: ArrayType, b: ArrayType) -> List[ArrayType]:
            return engine.dot(grad_output, b.T), engine.dot(a.T, grad_output)
        
        return list(map(Tensor, _backward(grad_output._array, a._array, b._array)))