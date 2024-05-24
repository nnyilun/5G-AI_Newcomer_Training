from typing import Any
import numpy as np
from .tensor import Tensor

class Module:
    def __init__(self) -> None:
        self.training = True
        
    def __call__(self, *input:Any) -> Tensor:
        return self.forward(*input)
    
    def forward(self, *input:Any) -> None:
        raise NotImplementedError
    
    def backward(self, grad:Tensor) -> None:
        raise NotImplementedError
    
    def parameter(self) -> list:
        return []
    
    def train(self) -> None:
        if 'training' in vars(self):
            self.training = True
        for attr in vars(self).values():
            if isinstance(attr, Module):
                attr.train()

    def eval(self) -> None:
        if 'training' in vars(self):
            self.training = False
        for attr in vars(self).values():
            if isinstance(attr, Module):
                attr.eval()


class Linear(Module):
    def __init__(self, input_num:int, output_num:int) -> None:
        super().__init__()
        self.weight = Tensor((output_num, input_num))
        self.bias = Tensor((output_num, ))

    def forward(self, x:Tensor) -> Tensor:
        self.last_input = x
        return x @ self.weight.T + self.bias
    
    def backward(self, dy:Tensor) -> Tensor:
        self.weight.grad = dy.T @ self.last_input
        self.bias.grad = np.sum(dy, axis=0, keepdims=True)
        return dy @ self.weight
    

if __name__ == "__main__":
    a = Tensor([[1, 2], [3, 4]])
    linear = Linear(2, 2)
    out = linear(a)
    dy = Tensor([[1, 0], [0, 1]])
    grad_input = linear.backward(dy)
    print("Gradient w.r.t input:\n", grad_input)
    print("Gradient w.r.t weights:\n", linear.weight)
    print("Gradient w.r.t bias:\n", linear.bias)