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
        for key, value in parameters.items():
            if isinstance(value, Tensor):
                self._update_weight(value)
            elif isinstance(value, dict):
                self._step_layer(value)

    def _update_weight(self, tensor: Tensor) -> None:
        tensor -= self.learning_rate * tensor.grad


class SGD(Optimizer):
    def __init__(self, parameters: ParameterType, lr: float, momentum: float = 0.0):
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.velocity = self._initialize_velocity(parameters)

    def _initialize_velocity(self, module: ParameterType):
        velocity = {}
        for key, value in module.items():
            if isinstance(value, Tensor):
                velocity[key] = 0
            elif isinstance(value, dict):
                velocity[key] = self._initialize_velocity(value)
        return velocity

    def _update_weight(self, tensor: Tensor, key: str, parent_velocity: dict):
        if tensor.grad is not None:
            if key not in parent_velocity:
                parent_velocity[key] = np.zeros_like(tensor)
            parent_velocity[key] = self.momentum * parent_velocity[key] - self.learning_rate * tensor.grad
            tensor += parent_velocity[key]

    def _step_layer(self, layer: ParameterType, parent_velocity: dict = None):
        if parent_velocity is None:
            parent_velocity = self.velocity

        for key, value in layer.items():
            if isinstance(value, Tensor):
                self._update_weight(value, key, parent_velocity)
            elif isinstance(value, dict):
                if key not in parent_velocity:
                    parent_velocity[key] = {}
                self._step_layer(value, parent_velocity[key])


class Adam(Optimizer):
    def __init__(self, parameters: ParameterType, lr: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        super().__init__(parameters, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.moment = self._initialize_moment(parameters)
        self.velocity = self._initialize_velocity(parameters)
        self.t = 0

    def _initialize_moment(self, module: ParameterType):
        moment = {}
        for key, value in module.items():
            if isinstance(value, Tensor):
                moment[key] = 0
            elif isinstance(value, dict):
                moment[key] = self._initialize_moment(value)
        return moment

    def _initialize_velocity(self, module: ParameterType):
        velocity = {}
        for key, value in module.items():
            if isinstance(value, Tensor):
                velocity[key] = 0
            elif isinstance(value, dict):
                velocity[key] = self._initialize_velocity(value)
        return velocity

    def _update_weight(self, tensor: Tensor, key: str, parent_moment: dict, parent_velocity: dict):
        if tensor.grad is not None:
            if key not in parent_moment:
                parent_moment[key] = np.zeros_like(tensor)
                parent_velocity[key] = np.zeros_like(tensor)
            self.t += 1
            parent_moment[key] = self.beta1 * parent_moment[key] + (1 - self.beta1) * tensor.grad
            parent_velocity[key] = self.beta2 * parent_velocity[key] + (1 - self.beta2) * (tensor.grad ** 2)
            moment_hat = parent_moment[key] / (1 - self.beta1 ** self.t)
            velocity_hat = parent_velocity[key] / (1 - self.beta2 ** self.t)
            tensor -= self.learning_rate * moment_hat / (np.sqrt(velocity_hat) + self.eps)

    def _step_layer(self, layer: ParameterType, parent_moment: dict = None, parent_velocity: dict = None):
        if parent_moment is None:
            parent_moment = self.moment
        if parent_velocity is None:
            parent_velocity = self.velocity

        for key, value in layer.items():
            if isinstance(value, Tensor):
                self._update_weight(value, key, parent_moment, parent_velocity)
            elif isinstance(value, dict):
                if key not in parent_moment:
                    parent_moment[key] = {}
                    parent_velocity[key] = {}
                self._step_layer(value, parent_moment[key], parent_velocity[key])