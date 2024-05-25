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
        self.weight = Tensor((output_num, input_num))
        self.bias = Tensor((output_num, ))

    def forward(self, x:Tensor) -> Tensor:
        self.last_input = x
        return x @ self.weight.T + self.bias
    
    def backward(self, dy:Tensor) -> Tensor:
        self.weight.grad = dy.T @ self.last_input
        self.bias.grad = np.sum(dy, axis=0)
        return dy @ self.weight
    
    def parameters(self) -> ParameterType:
        return {
            'weight': self.weight,
            'bias': self.bias,
        }


class Sigmoid(Module):
    def __init__(self) -> None:
        super().__init__()
    
    def sigmoid(self, x: Tensor) -> Tensor:
        indices_pos = np.nonzero(x >= 0)
        indices_neg = np.nonzero(x < 0)
 
        y = np.zeros_like(x)
        y[indices_pos] = 1 / (1 + np.exp(-x[indices_pos]))
        y[indices_neg] = np.exp(x[indices_neg]) / (1 + np.exp(x[indices_neg]))
        return y
        
    # def sigmoid(self, x: Tensor) -> Tensor:
    #     return 1 / (1 + np.exp(-x))
    
    def forward(self, x: Tensor) -> Tensor:
        self.last_output = self.sigmoid(x)
        return Tensor(self.last_output)

    def backward(self, dy: Tensor) -> Tensor:
        sig_grad = self.last_output * (1 - self.last_output)
        return Tensor(dy * sig_grad)
    

class ReLU(Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x: Tensor) -> Tensor:
        self.last_input = x
        return np.maximum(x, 0)
    
    def backward(self, dy: Tensor) -> Tensor:
        return dy * (self.last_input > 0)
    

class CrossEntropyLoss(Module):
    def __init__(self, eps: float = 1.e-16) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        exps = np.exp(input - np.max(input, axis=1, keepdims=True))
        softmax_output = exps / np.sum(exps, axis=1, keepdims=True)

        softmax_clamped = np.clip(softmax_output, self.eps, 1 - self.eps)
        
        self.last_input, self.last_target = input, target
        correct_log_probs = -np.log(softmax_clamped[np.arange(len(target)), target.astype(int)])
        
        loss = np.mean(correct_log_probs)
        return Tensor(loss)

    def backward(self) -> Tensor:
        exps = np.exp(self.last_input - np.max(self.last_input, axis=1, keepdims=True))
        softmax_output = exps / np.sum(exps, axis=1, keepdims=True)

        softmax_output[np.arange(len(self.last_target)), self.last_target.astype(int)] -= 1
        softmax_output /= len(self.last_target)
        return Tensor(softmax_output)


if __name__ == "__main__":
    a = Tensor([[1, 2], [3, 4]])
    linear = Linear(2, 2)
    out = linear(a)
    dy = Tensor([[1, 0], [0, 1]])
    grad_input = linear.backward(dy)
    print("Gradient w.r.t input:\n", grad_input)
    print("Gradient w.r.t weights:\n", linear.weight)
    print("Gradient w.r.t bias:\n", linear.bias)