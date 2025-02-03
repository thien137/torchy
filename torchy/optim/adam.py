import torchy 
from torchy import Tensor

from .optimizer import Optimizer 

from typing import Any, Union, Tuple

class Adam(Optimizer):
    def __init__(
        self, 
        params: Any, #TODO
        lr: Union[float, Tensor] = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad
        )

        super().__init__(params, defaults)
    
    def step(self):
        loss = None 
        
        for group in self.param_groups:
            