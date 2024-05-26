import numpy as np
from .tensor import Tensor
from .modules import ParameterType


class Optimizer(object):
    def __init__(self, parameters: ParameterType, lr: float) -> None:
        self.parameters = parameters
        self.learning_rate = lr

    def step(self) -> None:
        self._step_layer(self.parameters)

    def _step_layer(self, parameters: ParameterType) -> None:
        # parameters is a Dict, containing the parameters of the model
        # key: layer name; value: layer parameters, new Dict or Tensor or None
        
        # TODO Iterate over the parameters, 
        # if the value is a tensor, update the weight of the tensor
        # if the value is a dict, call _step_layer recursively
        ...

    def _update_weight(self, tensor: Tensor) -> None:
        # Update the weight of the tensor
        ...


class SGD(Optimizer):
    def __init__(self, parameters: ParameterType, lr: float, momentum: float = 0.0):
        super().__init__(parameters, lr)
        # TODO
        ...

    def _update_weight(self, tensor: Tensor, key: str, parent_velocity: dict):
        # TODO
        ...

    def _step_layer(self, layer: ParameterType, parent_velocity: dict = None):
        # TODO
        ...