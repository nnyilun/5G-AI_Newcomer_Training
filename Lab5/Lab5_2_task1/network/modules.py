from typing import Any, Optional, Union, Dict, TYPE_CHECKING
import numpy as np
from .tensor import Tensor

if TYPE_CHECKING:
    from typing import Self
ParameterType = Dict[str, Union[Tensor, 'ParameterType', None]]

class Module:
    def __init__(self) -> None:
        self.training = True
        
    def __call__(self, *input:Any) -> Tensor:
        return self.forward(*input)
    
    def forward(self) -> None:
        raise NotImplementedError
    
    def backward(self) -> None:
        raise NotImplementedError
    
    def parameters(self) -> ParameterType:
        return {}
    
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
        # TODO Initialize weights and biases
        # weight: [output_num, input_num]
        # bias: [output_num, ]
        ...

    def forward(self, x:Tensor) -> Tensor:
        # TODO Implement forward pass
        ...
    
    def backward(self, dy:Tensor) -> Tensor:
        # TODO Implement backward pass
        ...


class Sigmoid(Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x: Tensor) -> Tensor:
        # TODO Implement forward pass
        ...

    def backward(self, dy: Tensor) -> Tensor:
        # TODO Implement backward pass
        ...
    

class CrossEntropyLoss(Module):
    def __init__(self, eps: float = 1.e-16) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # TODO Implement forward pass
        ...

    def backward(self) -> Tensor:
        # TODO Implement backward pass
        ...