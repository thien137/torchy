import torchy 
from torchy import Tensor 
from torchy.autograd import Function, FunctionCtx

class ReLU(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, a: Tensor):
        ctx.save_for_backward(a)
        return ctx.engine.maximum(0, a._array)
    
    @staticmethod
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        ret = torchy.ones(a.shape)
        ret[a > 0] = 1
        ret[a <= 0] = 0
        return grad_output * ret

class Sigmoid(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return 1 / (1 + ctx.engine.exp(-a._array))

    @staticmethod 
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors 
        sig = torchy.Tensor(1 / (1 + ctx.engine.exp(-a._array)))
        return grad_output * sig * (1 - sig)