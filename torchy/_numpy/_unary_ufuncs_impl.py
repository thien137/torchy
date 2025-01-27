import torchy
from torchy import Tensor 
from torchy.autograd import Function, FunctionCtx

import torchy._engine_wrapper as engine
from torchy._engine_wrapper import ArrayType

from typing import List

class Log(Function):
    @staticmethod 
    def forward(ctx: FunctionCtx, a: Tensor) -> Tensor:
        ctx.save_for_backward(a)
        
        def _forward(a: ArrayType) -> ArrayType:
            return engine.log(a)
        
        return Tensor(_forward(a._array))
    
    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Tensor) -> Tensor:
        a, = ctx.saved_inputs
        
        def _backward(grad_output: ArrayType, a: ArrayType) -> List[ArrayType]:
            return grad_output / a
        
        return list(map(Tensor, _backward(grad_output._array, a._array)))
    
class Exp(Function):
    @staticmethod 
    def forward(ctx: FunctionCtx, a: Tensor) -> Tensor:
        ctx.save_for_backward(a)
        
        def _forward(a: ArrayType) -> ArrayType:
            return engine.exp(a)
        
        return Tensor(_forward(a._array))
    
    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Tensor) -> Tensor:
        a, = ctx.saved_inputs
        
        def _backward(grad_output: ArrayType, a: ArrayType) -> List[ArrayType]:
            return grad_output * engine.exp(a)
        
        return list(map(Tensor, _backward(grad_output._array, a._array)))
    
class Sin(Function):
    @staticmethod 
    def forward(ctx: FunctionCtx, a: Tensor) -> Tensor:
        ctx.save_for_backward(a)
        
        def _forward(a: ArrayType) -> ArrayType:
            return engine.sin(a)
        
        return Tensor(_forward(a._array))
    
    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Tensor) -> Tensor:
        a, = ctx.saved_inputs
        
        def _backward(grad_output: ArrayType, a: ArrayType) -> List[ArrayType]:
            return grad_output * engine.cos(a)
        
        return list(map(Tensor, _backward(grad_output._array, a._array)))
    
class Cos(Function):
    @staticmethod 
    def forward(ctx: FunctionCtx, a: Tensor) -> Tensor:
        ctx.save_for_backward(a)
        
        def _forward(a: ArrayType) -> ArrayType:
            return engine.cos(a)
        
        return Tensor(_forward(a._array))
    
    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Tensor) -> Tensor:
        a, = ctx.saved_inputs
        
        def _backward(grad_output: ArrayType, a: ArrayType) -> List[ArrayType]:
            return -grad_output * engine.sin(a)
        
        return list(map(Tensor, _backward(grad_output._array, a._array)))

class Tan(Function):
    @staticmethod 
    def forward(ctx: FunctionCtx, a: Tensor) -> Tensor:
        ctx.save_for_backward(a)
        
        def _forward(a: ArrayType) -> ArrayType:
            return engine.tan(a)
        
        return Tensor(_forward(a._array))
    
    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Tensor) -> Tensor:
        a, = ctx.saved_inputs
        
        def _backward(grad_output: ArrayType, a: ArrayType) -> List[ArrayType]:
            return grad_output / engine.cos(a) ** 2
        
        return list(map(Tensor, _backward(grad_output._array, a._array)))
    
class Arcsin(Function):
    @staticmethod 
    def forward(ctx: FunctionCtx, a: Tensor) -> Tensor:
        ctx.save_for_backward(a)
        
        def _forward(a: ArrayType) -> ArrayType:
            return engine.arcsin(a)
        
        return Tensor(_forward(a._array))
    
    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Tensor) -> Tensor:
        a, = ctx.saved_inputs
        
        def _backward(grad_output: ArrayType, a: ArrayType) -> List[ArrayType]:
            return grad_output / engine.sqrt(1 - a ** 2)
        
        return list(map(Tensor, _backward(grad_output._array, a._array)))
    
class Arccos(Function):
    @staticmethod 
    def forward(ctx: FunctionCtx, a: Tensor) -> Tensor:
        ctx.save_for_backward(a)
        
        def _forward(a: ArrayType) -> ArrayType:
            return engine.arccos(a)
        
        return Tensor(_forward(a._array))
    
    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Tensor) -> Tensor:
        a, = ctx.saved_inputs
        
        def _backward(grad_output: ArrayType, a: ArrayType) -> List[ArrayType]:
            return -grad_output / engine.sqrt(1 - a ** 2)
        
        return list(map(Tensor, _backward(grad_output._array, a._array)))
    
class Arctan(Function):
    @staticmethod 
    def forward(ctx: FunctionCtx, a: Tensor) -> Tensor:
        ctx.save_for_backward(a)
        
        def _forward(a: ArrayType) -> ArrayType:
            return engine.arctan(a)
        
        return Tensor(_forward(a._array))
    
    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Tensor) -> Tensor:
        a, = ctx.saved_inputs
        
        def _backward(grad_output: ArrayType, a: ArrayType) -> List[ArrayType]:
            return grad_output / (1 + a ** 2)
        
        return list(map(Tensor, _backward(grad_output._array, a._array)))
    
class Sqrt(Function):
    @staticmethod 
    def forward(ctx: FunctionCtx, a: Tensor) -> Tensor:
        ctx.save_for_backward(a)
        
        def _forward(a: ArrayType) -> ArrayType:
            return engine.sqrt(a)
        
        return Tensor(_forward(a._array))
    
    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Tensor) -> Tensor:
        a, = ctx.saved_inputs
        
        def _backward(grad_output: ArrayType, a: ArrayType) -> List[ArrayType]:
            return grad_output / (2 * engine.sqrt(a))
        
        return list(map(Tensor, _backward(grad_output._array, a._array)))

class Abs(Function):
    @staticmethod 
    def forward(ctx: FunctionCtx, a: Tensor) -> Tensor:
        ctx.save_for_backward(a)
        
        def _forward(a: ArrayType) -> ArrayType:
            return engine.abs(a)
        
        return Tensor(_forward(a._array))
    
    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Tensor) -> Tensor:
        a, = ctx.saved_inputs
        
        def _backward(grad_output: ArrayType, a: ArrayType) -> List[ArrayType]:
            return grad_output * engine.sign(a)
        
        return list(map(Tensor, _backward(grad_output._array, a._array)))
    
class Sign(Function):
    @staticmethod 
    def forward(ctx: FunctionCtx, a: Tensor) -> Tensor:
        ctx.save_for_backward(a)
        
        def _forward(a: ArrayType) -> ArrayType:
            return engine.sign(a)
        
        return Tensor(_forward(a._array))
    
    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Tensor) -> Tensor:
        a, = ctx.saved_inputs
        
        def _backward(grad_output: ArrayType, a: ArrayType) -> List[ArrayType]:
            return torchy.zeros(a.shape)
        
        return list(map(Tensor, _backward(grad_output._array, a._array)))