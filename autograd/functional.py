import torchy
from torchy import Tensor 
from torchy.autograd import Function 

class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return torchy.Tensor(a._array + b._array) 

    @staticmethod 
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors

        grad_a = grad_output 
        grad_b = grad_output 

        return grad_a, grad_b

class Mul(Function):
    @staticmethod 
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return torchy.Tensor(a._array * b._array)
    
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors

        grad_a = torchy.Tensor(grad_output._array * b._array) 
        grad_b = torchy.Tensor(grad_output._array * a._array)

        return grad_a, grad_b 

class Sub(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Save the input tensors for backward pass
        ctx.save_for_backward(a, b)
        # Perform element-wise subtraction and return as a new tensor
        return torchy.Tensor(a._array - b._array) 

    @staticmethod 
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        a, b = ctx.saved_tensors

        # Gradients of the inputs with respect to the subtraction
        grad_a = grad_output
        grad_b = -grad_output  # Gradient of b is negative because of subtraction

        return grad_a, grad_b
    
class Div(Function):
    @staticmethod 
    def forward(ctx, a, b):
        # Save the input tensors for backward pass
        ctx.save_for_backward(a, b)
        # Perform element-wise division and return as a new tensor
        return torchy.Tensor(a._array / b._array)
    
    @staticmethod
    def backward(ctx, grad_output: Tensor):
        # Retrieve saved tensors
        a, b = ctx.saved_tensors

        # Gradients of the inputs with respect to division
        grad_a = grad_output / b._array  # Derivative of a / b with respect to a is 1 / b
        grad_b = -grad_output * (a._array / b._array**2)  # Derivative of a / b with respect to b is -a / b^2

        return grad_a, grad_b