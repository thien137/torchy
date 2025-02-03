import torchy 
from torchy import Tensor 

from torchy.nn import functional as F

from .module import Module 

class _Loss(Module):
    reduction: str
    
    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__()
        self.reduction = reduction
    
class MSELoss(_Loss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.mse_loss(input, target, reduction=self.reduction)