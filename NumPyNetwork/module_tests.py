import random
import functools
import torch
import torch.nn.functional as F
import numpy as np
import network.modules as nnm
from network.tensor import Tensor

isclose = functools.partial(np.isclose, rtol=1.e-12, atol=1.e-12)

class TestBase:
    def __init__(self, module_name: str, input_shape: str, model_params: str = None, test_num: int = 3) -> None:
        self.module_name = module_name
        self.test_num = test_num

        input_shape = [s.strip() for s in input_shape.split(',')]
        model_params = [s.strip() for s in model_params.split(',')] if model_params else []
        keys = set(input_shape + model_params)
        args = [{k: v for k, v in zip(keys, [random.randint(8, 16) for _ in range(len(keys))])} for _ in range(test_num)]

        self.inputs = [Tensor(tuple(args[i][k] for k in input_shape)) for i in range(test_num)]
        self.nets = [getattr(nnm, self.module_name)(*tuple(args[i][k] for k in model_params)) for i in range(test_num)]
        
        self.net_grad = None
        self.torch_out = torch.Tensor()
        self.torch_grad = torch.Tensor()

    def __call__(self) -> None:
        self.run()

    def forward_test(self, test_id: int) -> bool:
        self.net_out = self.nets[test_id](self.inputs[test_id])
        if self.net_out is None:
            return False
        return isclose(self.net_out, self.torch_out.detach().numpy()).all().item()

    def backward_test(self, test_id: int) -> bool:
        if self.net_out is None:
            return False
        self.net_grad = self.nets[test_id].backward(Tensor.ones(self.net_out.shape))
        if self.net_grad is None:
            return False
        self.torch_out.retain_grad()
        self.torch_out.sum().backward()
        self.torch_grad = self.torch_input.grad
        return isclose(self.net_grad, self.torch_grad.detach().numpy()).all().item()

    def run(self) -> None:
        str_len = 40
        print(f'[\033[35m{self.module_name}\033[0m]{"." * (str_len - len(self.module_name))}')
        result_str = lambda result: '[\033[32mpass\033[0m]' if result else '[\033[31mfail\033[0m]'

        _forward_str = '[Forward][pass]'
        _backward_str = '[Backward][pass]'

        for test_id in range(self.test_num):
            print(f'[\033[34mForward\033[0m]' + '.' * (str_len - len(_forward_str)) + \
                    result_str(self.forward_test(test_id)))
            print(f'[\033[34mBackward\033[0m]' + '.' * (str_len - len(_backward_str)) + \
                    result_str(self.backward_test(test_id)))


class TestLinear(TestBase):
    def __init__(self) -> None:
        super().__init__('Linear', 'Batch, Input_num', 'Input_num, Output_num')

    def forward_test(self, test_id: int) -> bool:
        self.torch_input = torch.tensor(self.inputs[test_id], requires_grad=True)
        self.torch_weight = torch.tensor(self.nets[test_id].weight, requires_grad=True)
        self.torch_bias = torch.tensor(self.nets[test_id].bias, requires_grad=True)
        self.torch_out = F.linear(input=self.torch_input, weight=self.torch_weight, bias=self.torch_bias)
        return super().forward_test(test_id)
    
    def backward_test(self, test_id: int) -> bool:
        ret = super().backward_test(test_id)
        ret &= isclose(self.nets[test_id].weight.grad, self.torch_weight.grad).all().item()
        ret &= isclose(self.nets[test_id].bias.grad, self.torch_bias.grad).all().item()
        return ret


class TestConv2d(TestBase):
    def __init__(self) -> None:
        super().__init__('Conv2d', 'Batch, Channel_In, Height, Width', 'Channel_Out, Channel_In')

    def forward_test(self, test_id: int) -> bool:
        raise NotImplementedError
    
    def backward_test(self, test_id: int) -> bool:
        raise NotImplementedError


class TestSigmoid(TestBase):
    def __init__(self) -> None:
        super().__init__("Sigmoid", "Batch, Length")

    def forward_test(self, test_id: int) -> bool:
        self.torch_input = torch.tensor(self.inputs[test_id], requires_grad=True)
        self.torch_out = torch.sigmoid(self.torch_input)
        return super().forward_test(test_id)
    
    def backward_test(self, test_id: int) -> bool:
        return super().backward_test(test_id)


class TestReLU(TestBase):
    def __init__(self) -> None:
        super().__init__("ReLU", "Batch, Length")

    def forward_test(self, test_id: int) -> bool:
        self.torch_input = torch.tensor(self.inputs[test_id], requires_grad=True)
        self.torch_out = torch.relu(self.torch_input)
        return super().forward_test(test_id)
    
    def backward_test(self, test_id: int) -> bool:
        return super().backward_test(test_id)


class TestLossBase(TestBase):
    def __init__(self, module_name: str, input_shape: str = "Batch, Label") -> None:
        super().__init__(module_name, input_shape)
        self.type = type

    def forward_test(self, test_id: int) -> bool:
        batch, label = self.inputs[test_id].shape
        self.target = Tensor(np.random.randint(0, label, size=(batch, )))

        self.torch_input = torch.tensor(self.inputs[test_id], dtype=torch.float64, requires_grad=True)
        self.torch_target = torch.tensor(self.target, dtype=torch.long)

        loss_func = torch.nn.CrossEntropyLoss()
        self.torch_out = loss_func(self.torch_input, self.torch_target)
        
        self.net_out = self.nets[test_id](self.inputs[test_id], self.target)
        if self.net_out is None:
            return False
        return isclose(self.net_out, self.torch_out.detach().numpy()).all().item()
    
    def backward_test(self, test_id: int) -> bool:
        if self.net_out is None:
            return False
        self.net_grad = self.nets[test_id].backward()
        if self.net_grad is None:
            return False
        self.torch_out.retain_grad()
        self.torch_out.sum().backward()
        self.torch_grad = self.torch_input.grad
        return isclose(self.net_grad, self.torch_grad.detach().numpy()).all().item()


class TestCrossEntropyLoss(TestLossBase):
    def __init__(self) -> None:
        super().__init__("CrossEntropyLoss")

    def forward_test(self, test_id: int) -> bool:
        return super().forward_test(test_id)
    
    def backward_test(self, test_id: int) -> bool:
        return super().backward_test(test_id)


if __name__ == '__main__':
    TestLinear()()
    TestSigmoid()()
    TestReLU()()
    TestCrossEntropyLoss()()
