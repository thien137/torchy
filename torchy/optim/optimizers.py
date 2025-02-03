import numpy as np
from abc import ABC, abstractmethod 
from torchy import Tensor

from typing import List

eps = 10e-12

class Adam(Optimizer):
    def __init__(
        self,
        params: List[Tensor],
        lr=0.001,
        betas=(0.9, 0.999),
        weight_decay=0,
        amsgrad=False,
        maximize=False
    ):
        self.params = params
        self.lr = lr
        self.betas = betas 
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.maximize = maximize

        self._initalize_state()
        
    def _initalize_state(self):
        self.m = {param: 0 for param in self.params}
        self.v = {param: 0 for param in self.params}
        self.v_max = 0
        self.t = 0 

    def zero_grad(self):
        for _, param in self.params:
            param.zero_grad()
    
    def step(self):
        self.t += 1

        for param_name, param in self.params.items(): 
            if self.maximize:
                gt = -param.grad
            else:
                gt = param.grad 

            if self.weight_decay:
                gt += self.weight_decay*param
            
            self.m[param_name] = self.betas[0]*self.m[param_name] + (1-self.betas[0])*gt 
            self.v[param_name] = self.betas[1]*self.v[param_name] + (1-self.betas[1])*gt**2
            mt = self.m[param_name] / (1 - self.betas[0] ** self.t)
            vt = self.v[param_name] / (1 - self.betas[1] ** self.t)
            
            if self.amsgrad:
                self.vt_max = max(self.vt_max, vt)
                param[:] = param - self.lr*mt / (np.sqrt(self.vt_max) + eps)
            else:
                param[:] = param - self.lr*mt / (np.sqrt(vt) + eps)