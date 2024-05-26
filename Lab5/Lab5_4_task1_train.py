from tqdm import tqdm
import network.modules as nnm
import network.optim as optim
from network.utils import load_mnist, DataLoader, normalize
from network.tensor import Tensor

n_features = 28 * 28
n_classes = 10
n_epoches = 10
batch_size = 128
lr = 0.5
net_params = (n_features, 512, n_classes)


class MLP(nnm.Module):
    def __init__(self) -> None:
        super().__init__()
        # TODO: create the layers, init params
        ...
    
    def forward(self, x: Tensor) -> Tensor:
        # TODO: forward pass
        ...
    
    def backward(self, loss: Tensor) -> Tensor:
        # TODO: backward pass
        ...

    def parameters(self) -> nnm.ParameterType:
        # TODO: return the parameters to optimize
        ...


def main():
    # First: load the MNIST data
    # TODO

    # Second: create the network, optimizer and criterion
    # TODO

    # Third: train the network
    for epoch in range(n_epoches):
        # TODO
        # forward
        # caclulate loss
        # backward
        # update weights
        ...


if __name__ == '__main__':
    main()