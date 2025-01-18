import torchy 
from torchy import Tensor 

from torchy.autograd import Function, FunctionCtx

class Maximum(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return
    
    @staticmethod 
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors 
        return 

class Minimum(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return 

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors 
        
        return  