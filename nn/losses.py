import numpy as np

from torchy import Tensor
from abc import ABC, abstractmethod 

class Loss(ABC):
    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self):
        pass

class MSE(Loss):
    """Mean squared error loss function."""

    def __init__(self, name: str) -> None:
        self.name = name 
        self.cache = {}
    
    def __call__(self, Y: Tensor, Y_hat: Tensor) -> Tensor:
        return self.forward(Y, Y_hat)
    
    def forward(self, Y: Tensor, Y_hat: Tensor) -> Tensor:
        """Computes the MSE loss for predictions 'Y' given ground truth 'Y_hat'

        Args:
            Y (Tensor): Predictions
            Y_hat (np.ndarray): Ground truth

        Returns:
            float: a single float representing the loss
        """
        self.cache["Y"] = Y
        self.cache["Y_hat"] = Y_hat
        
        return Tensor(np.sum((Y - Y_hat)**2)/Y_hat.shape[0], grad_fn=self)
    
    def backward(self) -> None:
        """Backwards pass of mean squared error.
        """
        Y = self.cache["Y"]
        Y_hat = self.cache["Y_hat"]
        
        dY = 2 * Tensor((Y - Y_hat)/np.prod(Y_hat.shape[1:]))
        Y.backward(dY)
        
class BinaryCrossEntropy(Loss):
    """Binary cross entropy loss function."""
    
    def __init__(self, name: str) -> None:
        self.name = name
        self.cache = {}
    
    def __call__(self, Y: Tensor, Y_hat: Tensor) -> float:
        return self.forward(Y, Y_hat)

    def forward(self, Y: Tensor, Y_hat: Tensor) -> float:
        """Computes the loss for predictions 'Y' given binary labels 'Y_hat'

        Args:
            Y (Tensor): Prediction labels
            Y_hat (Tensor): Ground truth labels

        Returns:
            float: binary cross entropy loss
        """
        self.cache["Y"] = Tensor(Y+10e-8, grad_fn=Y.grad_fn)
        self.cache["Y_hat"] = Tensor(Y_hat+10e-8)
        
        left_log = np.maximum(np.log(Y), -100)
        right_log = np.maximum(np.log(1 - Y), -100)
        
        return Tensor(-np.mean(Y_hat*left_log + (1 - Y_hat)*right_log), grad_fn=self)
    
    def backward(self) -> float:
        """Computes the backward pass for binary cross entropy."""
        Y = self.cache["Y"]
        Y_hat = self.cache["Y_hat"]
        
        dY = -(Y_hat/Y - (1-Y_hat)/(1-Y))
        dY = dY / np.prod(Y.shape[1:])
        
        Y.backward(Tensor(dY)) 
        
class CrossEntropy(Loss):
    """Cross entropy loss function."""

    def __init__(self, name: str) -> None:
        self.name = name 
    
    def __call__(self, Y: np.ndarray, Y_hat: np.ndarray) -> float:
        return self.forward(Y, Y_hat)

    def forward(self, Y: np.ndarray, Y_hat: np.ndarray) -> float:
        """Computes the loss for predictions `Y_hat` given one-hot encoded labels
        `Y`.

        Args:
            Y (np.ndarray): one-hot encoded labels of shape (batch_size, num_classes)
            Y_hat (np.ndarray): model predictions in range (0, 1) of shape (batch_size, num_classes)

        Returns:
            float: a single float representing the loss
        """
        return ...

    def backward(self, Y: np.ndarray, Y_hat: np.ndarray) -> np.ndarray:
        """Backward pass of cross-entropy loss.

        Args:
            Y (np.ndarray): one-hot encoded labels of shape (batch_size, num_classes)
            Y_hat (np.ndarray): model predictions in range (0, 1) of shape (batch_size, num_classes)

        Returns:
            np.ndarray: a single float representing the loss
        """
        return ...