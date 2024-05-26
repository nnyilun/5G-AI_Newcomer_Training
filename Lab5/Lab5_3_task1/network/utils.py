import os
import struct
from typing import Callable, Tuple, List 
import matplotlib.pyplot as plt
import numpy as np
from .tensor import Tensor


def load_mnist(root:str, train:bool=True) -> list[Tensor, Tensor]:
    # TODO load MNIST dataset
    ...


class DataLoader:
    # TODO Implement DataLoader
    ...