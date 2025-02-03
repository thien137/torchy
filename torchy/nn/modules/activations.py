import torchy 
from torchy import Tensor 
from torchy.nn import functional as F

from .module import Module

class ReLU(Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor: 
        return F.relu(input)
    
class Sigmoid(Module):
    def forward(self, input: Tensor) -> Tensor:
        return torchy.sigmoid(input)

