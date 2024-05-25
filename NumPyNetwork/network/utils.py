import os
import struct
from typing import Callable, Tuple, List 
import matplotlib.pyplot as plt
import numpy as np
from .tensor import Tensor


def load_mnist(root:str, train:bool=True) -> list[Tensor, Tensor]:
    if train:
        images_path = os.path.join(root, 'train-images-idx3-ubyte')
        labels_path = os.path.join(root, 'train-labels-idx1-ubyte')
    else:
        images_path = os.path.join(root, 't10k-images-idx3-ubyte')
        labels_path = os.path.join(root, 't10k-labels-idx1-ubyte')

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = Tensor(np.fromfile(lbpath, dtype=np.uint8))

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = Tensor(np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784))

    return images, labels


def resize_image(image: Tensor, new_size: int):
    height, width = image.shape
    new_height, new_width = new_size
    resized_image = Tensor.zeros((new_height, new_width))
    row_ratio, col_ratio = new_height / height, new_width / width

    for r in range(new_height):
        for c in range(new_width):
            old_r, old_c = int(r / row_ratio), int(c / col_ratio)
            resized_image[r, c] = image[old_r, old_c]
    return resized_image


def center_crop(image: Tensor, crop_size: int):
    height, width = image.shape
    crop_height, crop_width = crop_size
    start_row = (height - crop_height) // 2
    start_col = (width - crop_width) // 2
    cropped_image = image[start_row:start_row + crop_height, start_col:start_col + crop_width]
    return cropped_image


def preprocess_images(images: Tensor):
    processed_images = []
    for image in images:
        image = image.reshape(28, 28)
        image = resize_image(image, (16, 16))
        image = center_crop(image, (8, 8))
        processed_images.append(image.flatten())
    return np.array(processed_images)


def display_predictions(images: Tensor, labels: Tensor, predictions: Tensor, num_images: int=20, save_root: str=None):
    rows = (num_images + 7) // 8
    fig, axes = plt.subplots(rows, 8, figsize=(15, 2 * rows))
    axes = axes.ravel()
    for i in range(num_images):
        axes[i].imshow(images[i].reshape(28, 28), cmap='gray')
        axes[i].set_title(f'True: {labels[i]}\nPred: {predictions[i]}')
        axes[i].axis('off')
    for i in range(num_images, rows * 8):
        axes[i].axis('off')
    plt.tight_layout()
    if save_root is None:
        plt.show()
    else:
        plt.savefig(save_root)


def normalize(data: np.ndarray) -> Tuple[np.ndarray]:
    min_val = np.min(data)
    max_val = np.max(data)
    data = (data - min_val) / (max_val - min_val)
    return data


class DataLoader:
    def __init__(self, data, labels, batch_size, shuffle=True, 
                 preprocess: List[Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]] = None) -> None:
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.preprocess = preprocess
        self.indices = np.arange(len(data))
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.current_index = 0

    def __iter__(self) -> 'DataLoader':
        return self

    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.current_index >= len(self.data):
            if self.shuffle:
                np.random.shuffle(self.indices)
            self.current_index = 0
            raise StopIteration
        batch_indices = self.indices[self.current_index:self.current_index + self.batch_size]
        batch_data = self.data[batch_indices]
        batch_labels = self.labels[batch_indices]
        if self.preprocess:
            for func in self.preprocess:
                batch_data = func(batch_data)
        self.current_index += self.batch_size
        return batch_data, batch_labels
    
    def __len__(self) -> int:
        return -(- len(self.data) // self.batch_size)
    

if __name__ == '__main__':
    train, label = load_mnist('../mnist', train=True)
    print(train.shape, label.shape)
