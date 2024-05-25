import numpy as np
from typing import Union, Tuple

class Tensor(np.ndarray):
    def __new__(cls, input_array: Union[np.ndarray, Tuple[int, ...]]) -> 'Tensor':
        if isinstance(input_array, tuple):
            obj = np.random.normal(loc=0, scale=1, size=input_array).astype(np.float64).view(cls)
        else:
            obj = np.array(input_array, dtype=np.float64).view(cls)
        
        obj._grad = None
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._grad = getattr(obj, 'grad', None)

    def __str__(self) -> str:
        return f"{super().__str__()}"
    
    def __repr__(self) -> str:
        return f"Tensor(data={super().__repr__()}, grad={self._grad})"

    @property
    def grad(self) -> np.ndarray:
        return self._grad

    @grad.setter
    def grad(self, value:'Tensor') -> None:
        if isinstance(value, Tensor):
            self._grad = value.view(np.ndarray)
        else:
            self._grad = np.array(value, dtype=np.float64)

    @classmethod
    def zeros(cls, shape: Tuple[int, ...]) -> 'Tensor':
        obj = np.zeros(shape, dtype=np.float64).view(cls)
        obj._grad = None
        return obj

    @classmethod
    def ones(cls, shape: Tuple[int, ...]) -> 'Tensor':
        obj = np.ones(shape, dtype=np.float64).view(cls)
        obj._grad = None
        return obj

    @classmethod
    def random(cls, shape: Tuple[int, ...], loc: float = 0.0, scale: float = 1.0) -> 'Tensor':
        obj = np.random.normal(loc=loc, scale=scale, size=shape).view(cls)
        obj._grad = None
        return obj


if __name__ == "__main__":
    a = Tensor((2, 2))
    print(a)
    
    b = Tensor(np.array([[1, 2], [3, 4]]))
    print(b)
    
    c = a + b
    print(c)

    d = a @ b
    print(d)


