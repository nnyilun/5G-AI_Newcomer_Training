import random
import numpy as np
from tqdm import tqdm
from knn import KNN
from utils import load_mnist, display_predictions, preprocess_images, DataLoader


def main():
    X_train, y_train = load_mnist('mnist', train=True)
    X_test, y_test = load_mnist('mnist', train=False)

    print("preprocessing...")
    X_train = preprocess_images(X_train)

    # train_loader = DataLoader(X_train, y_train, batch_size=128)
    # test_loader = DataLoader(X_test, y_test, batch_size=64)

    knn = KNN(k=3)
    knn.fit(X_train, y_train)


    print("predicting...")
    num_display_images = 20
    indices = random.sample(range(len(X_test)), num_display_images)
    images = X_test[indices]
    processed_images = preprocess_images(images)
    labels = y_test[indices]

    predictions = knn.predict(processed_images)
    correct = np.sum(predictions == labels)
    total = len(labels)
    print(f'Accuracy: {correct / total * 100:.2f}%')
    display_predictions(images, labels, predictions, num_images=num_display_images, save_root="knn_result.png")


if __name__ == "__main__":
    main()
    # Accuracy: 95.00%
