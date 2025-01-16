import numpy as np
from abc import ABC, abstractmethod
from layers import Layer

class Network(ABC):
    def __init__(self):
        super().__init__()
        
    def __call__(self, X):
        return self.forward(X)
    
    @abstractmethod
    def forward(self, X):
        pass
    
    def state_dict(self):
        return {layer_name: layer.parameters for layer_name, layer in vars(self).items()
                    if isinstance(layer, Layer)}
    
    def parameters(self):
        p = {}
        for layer_name, layer_params in self.state_dict().items():
            for param in layer_params:
                p[f"{layer_name}: {param}"] = layer_params[param]
        return p

    def zero_grad(self):
        for _, param in self.parameters().items():
            param.zero_grad()