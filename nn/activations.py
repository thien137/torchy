import numpy as np
from abc import ABC, abstractmethod 

from torchy import Tensor

class Activation(ABC):
    """Abstract class defining the common interface for all activation."""
    
    def __init__(self):
        self.cache = {}
    
    def __call__(self, X: np.ndarray):
        return self.forward(X)
    
    @abstractmethod
    def forward(self, X: np.ndarray):
        pass 

class Sigmoid(Activation):
    
    def __init__(self):
        super().__init__()
        
        self.cache = {}
        
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass for sigmoid function:
        f(X) = 1 / (1 + exp(-X))

        Args:
            X (np.ndarray): input pre-activations (any shape)

        Returns:
            np.ndarray: f(X) applied elementwise
        """
        self.cache["X"] = X
        
        Y = 1 / (1 + np.exp(np.clip(-X, -500, 500)))
        
        return Tensor(Y, grad_fn=self)

    def backward(self, dY: np.ndarray) -> np.ndarray:
        """Backward pass for sigmoid function.

        Args:
            X (np.ndarray): input to `forward` method
            dY (np.ndarray): gradient of loss w.r.t the output of this layer

        Returns:
            np.ndarray: gradient of loss w.r.t input of this layer
        """
        X = self.cache["X"]
        
        Y = 1 / (1 + np.exp(np.clip(-X, -500, 500)))
        
        dYdX = Y * (1 - Y)
        
        X.backward(Tensor(dY * dYdX))

class ReLU(Activation):
    def __init__(self):
        super().__init__()
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass for relu activation:
        f(X) = X if X >= 0
                0 otherwise

        Args:
            X (np.ndarray): input pre-activation (any shape)

        Returns:
            np.ndarray: f(X) applied elementwise
        """
        self.cache["X"] = X
        
        return Tensor(np.maximum(0, X), grad_fn=self)
    
    def backward(self, dY: np.ndarray) -> np.ndarray:
        """Backward pass for relu activation.

        Args:
            X (np.ndarray): input to `forward` method
            dY (np.ndarray): gradient of loss w.r.t the output of this layer

        Returns:
            np.ndarray: gradient of loss w.r.t input of this layer
        """
        X = self.cache["X"]
        
        X.backward(Tensor(dY * (self.cache["X"] > 0)))