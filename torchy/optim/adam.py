import torchy 
from torchy import Tensor

from .optimizer import Optimizer 

from typing import Any, Union, Tuple, Iterator

eps = 10e-12

class Adam(Optimizer):
    def __init__(
        self, 
        params: Iterator[Tuple[str, Tensor]],
        lr: Union[float, Tensor] = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
        *,
        maximize: bool = False
    ):
        self.params = params
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize
        )

        super().__init__(params, defaults)
        
        self._initalize_state()
        
    def _initalize_state(self):
        self.params_state : dict[str, dict] = {}

    def step(self):
        for name, param in self.params:
            if name not in self.params_state:
                self.params_state[name] = {
                    "lr": self.defaults["lr"],
                    "beta1": self.defaults["betas"][0],
                    "beta2": self.defaults["betas"][1],
                    "eps": self.defaults["eps"],
                    "weight_decay": self.defaults["weight_decay"],
                    "amsgrad": self.defaults["amsgrad"],
                    "maximize": self.defaults["maximize"],
                    "m": torchy.zeros_like(param),
                    "v": torchy.zeros_like(param),
                    "t": 0,
                    "vt_max": 0,
                }
            self._step(param, self.params_state[name])

    def _step(self, param: Tensor, state: dict[str, Union[Tensor, int, float]]):
        state["t"] += 1
        
        if state["maximize"]:
            gt = -param.grad
        else:
            gt = param.grad
        if state["weight_decay"]:
            gt += state["weight_decay"]*param
        
        state["m"] = state["beta1"]*state["m"] + (1-state["beta1"])*gt
        state["v"] = state["beta2"]*state["v"] + (1-state["beta2"])*gt**2
        mt = state["m"] / (1 - state["beta1"] ** state["t"])
        vt = state["v"] / (1 - state["beta2"] ** state["t"])

        if state["amsgrad"]:
            state["vt_max"] = max(state["vt_max"], vt)
            param[:] = param - state["lr"]*mt / (torchy.sqrt(state["vt_max"]) + state["eps"])
        else:
            param[:] = param - state["lr"]*mt / (torchy.sqrt(vt) + state["eps"])
            