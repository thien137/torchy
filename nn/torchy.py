import numpy as np

class Tensor(np.ndarray):
    def __new__(cls, arr, grad_fn=None):
        tensor = np.asarray(arr).view(cls)
        
        tensor.grad_fn = grad_fn
        tensor.grad = None
        
        return tensor
    
    def __array__finalize__(self, tensor):
        if tensor is None: return 

        self.grad_fn = getattr(tensor, 'grad_fn', None)
        self.grad = getattr(tensor, 'grad', None)
    
    def zero_grad(self):
        self.grad = np.zeros(self.grad.shape)

    def backward(self, dY=None):
        if self.grad_fn:
            if dY is not None:
                self.grad_fn.backward(dY)
            else:
                self.grad_fn.backward()
