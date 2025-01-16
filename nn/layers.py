import numpy as np
from abc import ABC, abstractmethod 

from torchy import Tensor
from weights import initialize_weights

from typing import Callable, List, Literal, Tuple, Union

class Layer(ABC):
    """Abstract class defining the `Layer` interface."""
    
    def __init__(self):
        super().__init__()
        
        self.n_in = None 
        self.n_out = None 
        
        self.parameters = {}
        self.cache = {}
        
    @abstractmethod
    def __call__(self, z: np.ndarray) -> np.ndarray:
        pass 

    def zero_grad(self) -> None:
        self.cache = {a: [] for a, b in self.cache.items()}
        for _, param in self.parameters.items():
            param.zero_grad()
    
    def get_parameters(self) -> List[np.ndarray]:
        return [b for _, b in self.parameters.items()]
 
class FullyConnected(Layer):
    """A fully-connected layer multiplies by a weight matrix and
    adds a bias
    """
    def __init__(
        self, 
        n_in: int,
        n_out: int, 
        weight_init="normal",
        activation=None
    ) -> None:
        
        super().__init__()
        
        self.n_in = n_in 
        self.n_out = n_out
        self.init_weights = initialize_weights(weight_init, activation)
    
        self._init_parameters()
        
    def _init_parameters(self) -> None:
        """Initialize all layer parameters(weights, biases)."""

        W = Tensor(arr=self.init_weights((self.n_in, self.n_out)),
                   grad_fn=self)
        b = Tensor(arr=np.zeros((1, self.n_out)),
                   grad_fn=self)
        
        self.parameters = {"W": W, "b": b}
        self.cache = {}
        
    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: multiply by a weight matrix, add a bias, apply activation.
        Also, store all necessary intermediate results in the `cache` dictionary
        to be able to compute the backward pass.

        Args:
            X (np.ndarray): input matrix of shape (batch_size, input_dim)

        Returns:
            np.ndarray: matrix of shape (batch_size, output_dim)
        """

        self.cache["X"] = X
        return Tensor(X @ self.parameters["W"] + np.ones((X.shape[0], 1)) @ self.parameters["b"], 
                      grad_fn=self)

    def backward(self, dLdY: Tensor) -> None:
        """Backward pass for fully connected layer.
        Compute the gradients of the loss with respect to:
            1. the weights of this layer (mutate the `gradients` dictionary)
            2. the bias of this layer (mutate the `gradients` dictionary)
            3. the input of this layer (return this)

        Args:
            dLdY (np.ndarray): gradient of the loss with respect to the output
            of this layer shape (batch_size, output_dim)

        Returns:
            np.ndarray: gradient of the loss with respect to the input of this
            layer shape (batch_size, input_dim)
        """

        if self.cache:
            self.parameters["W"].grad = self.cache["X"].T @ dLdY 
            self.parameters["b"].grad = dLdY.sum(axis=0, keepdims=True)
            self.cache["X"].backward(dLdY @ self.parameters["W"].T)        

class Conv2D(Layer):
    """Convolutional layer for inputs with 2 spatial dimensions."""
    def __init__(
        
    ) -> None:
        return