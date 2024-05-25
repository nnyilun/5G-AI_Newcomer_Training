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
        self.layers = {
            'linear1': nnm.Linear(net_params[0], net_params[1]),
            'act1': nnm.Sigmoid(),
            'linear2': nnm.Linear(net_params[1], net_params[2]),
        }
    
    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers.values():
            x = layer(x)
        return x
    
    def backward(self, loss: Tensor) -> Tensor:
        for layer in reversed(list(self.layers.values())):
            loss = layer.backward(loss)
        return loss

    def parameters(self) -> nnm.ParameterType:
        return {
            'linear1': self.layers['linear1'].parameters(),
            'act1': None,
            'linear2': self.layers['linear2'].parameters(),
        }


def main():
    train_data, train_label = load_mnist('./mnist', train=True)
    test_data, test_label = load_mnist('./mnist', train=False)
    train_iter = DataLoader(train_data, train_label, batch_size=batch_size, preprocess=[normalize])
    test_iter = DataLoader(test_data, test_label, batch_size=batch_size, preprocess=[normalize])

    net = MLP()
    optimizer = optim.SGD(net.parameters(), lr=lr)
    criterion = nnm.CrossEntropyLoss()

    for epoch in range(n_epoches):
        bar = tqdm(train_iter, desc='Epochs')
        for X, y in bar:
            y_hat = net(X)
            loss = criterion(y_hat, y)
            loss_value = loss.item()
            net.backward(criterion.backward())
            optimizer.step()
            bar.set_postfix({'Loss': loss_value})
        
        correct = 0
        total = 0
        for X, y in test_iter:
            y_hat = net(X)
            correct += (y_hat.argmax(1) == y).sum()
            total += y.size
        print(f"Epoch: {epoch + 1}, Test Accuracy: {correct / total * 100}%")


if __name__ == '__main__':
    main()