from .function import Function, FunctionCtx

__all__ = [
    "Function",
    "FunctionCtx",
    "AccumulateGradient"
]

    
class AccumulateGradient(Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output