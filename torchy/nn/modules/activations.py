import torchy 
from torchy import Tensor 
from torchy.nn import functional as F

from .module import Module

class ReLU(Module):
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace 
    
    def forward(self, input: Tensor) -> Tensor: 
        return F.relu(input, inplace=self.inplace)
    
class Sigmoid(Module):
    def forward(self, input: Tensor) -> Tensor:
        return torchy.sigmoid(input)

