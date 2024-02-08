import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import trange


def one_hot(data: np.ndarray) -> np.ndarray:
    y_train = np.zeros((data.size, data.max() + 1))
    rows = np.arange(data.size)
    y_train[rows, data] = 1
    return y_train


def plot(loss_history: list, accuracy_history: list, filename='plot'):
    n_epochs = len(loss_history)
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history)
    plt.xlabel('Epoch number')
    plt.ylabel('Loss')
    plt.xticks(np.arange(0, n_epochs, 4))
    plt.title('Loss on train dataframe from epoch')
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(accuracy_history)
    plt.xlabel('Epoch number')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(0, n_epochs, 4))
    plt.title('Accuracy on test dataframe from epoch')
    plt.grid()
    plt.savefig(f'{filename}.png')


def scale(data: np.ndarray) -> np.ndarray:
    return data / data.max()


def xavier(n_in: int, n_out: int) -> np.ndarray:
    limit = np.sqrt(6) / np.sqrt(n_in + n_out)
    return np.random.uniform(-limit, limit, (n_in, n_out))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x: np.ndarray) -> np.ndarray:
    return x * (1 - x)


def mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return np.mean((y_pred - y_true) ** 2)


def d_mse(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    return 2 * (y_pred - y_true)


def train_epoch(estimator, X: np.ndarray, y: np.ndarray, alpha: float, batch: int = 100) -> float:
    n = X.shape[0]
    for i in range(0, n, batch):
        estimator.forward(X[i:i + batch])
        estimator.backprop(X[i:i + batch], y[i:i + batch], alpha)
    estimator.forward(X)
    return mse(estimator.activations[-1], y)


def accuracy(estimator, X: np.ndarray, y: np.ndarray) -> float:
    estimator.forward(X)
    true_pred = np.argmax(estimator.activations[-1], axis=1) == np.argmax(y, axis=1)
    return np.sum(true_pred) / y.shape[0]


class OneLayerNeural:
    def __init__(self, n_features, n_classes):
        self.weights = xavier(n_features, n_classes)
        self.bias = xavier(1, n_classes)
        self.prediction = None

    def forward(self, X: np.ndarray) -> None:
        self.prediction = sigmoid(np.dot(X, self.weights) + self.bias)

    def backprop(self, X: np.ndarray, y: np.ndarray, alpha: float = 0.1) -> None:
        grad_weights = np.dot(X.T, d_sigmoid(self.prediction) * d_mse(self.prediction, y) / X.shape[0])
        grad_bias = np.mean(d_sigmoid(self.prediction) * d_mse(self.prediction, y), axis=0)
        self.weights -= alpha * grad_weights
        self.bias -= alpha * grad_bias


class NLayerNeural:
    def __init__(self, n_features, n_classes, hidden=(64,)):
        self.neurons = [n_features, *hidden, n_classes]
        self.n_layers = len(self.neurons) - 1
        self.weights = [xavier(self.neurons[i], self.neurons[i+1]) for i in range(self.n_layers)]
        self.biases = [xavier(1, self.neurons[i+1]) for i in range(self.n_layers)]
        self.activations = None

    def forward(self, X: np.ndarray) -> None:
        self.activations = [X,]
        for i in range(self.n_layers):
            self.activations.append(sigmoid(np.dot(self.activations[i], self.weights[i]) + self.biases[i]))

    def backprop(self, X: np.ndarray, y: np.ndarray, alpha: float = 0.1) -> None:
        upstream = d_sigmoid(self.activations[-1]) * d_mse(self.activations[-1], y) / X.shape[0]
        grad_w, grad_b = [], []
        for i in range(self.n_layers):  # calculate gradients
            grad_w.append(np.dot(self.activations[-2-i].T, upstream))
            grad_b.append(np.sum(upstream, axis=0))
            upstream = d_sigmoid(self.activations[-2-i]) * np.dot(upstream, self.weights[-1-i].T)
        for i in range(self.n_layers):  # update weights and biases
            self.weights[i] -= alpha * grad_w[::-1][i]
            self.biases[i] -= alpha * grad_b[::-1][i]


def main():
    # Read train, test data.
    raw_train = pd.read_csv('../Data/fashion-mnist_train.csv')
    raw_test = pd.read_csv('../Data/fashion-mnist_test.csv')
    X_train = raw_train[raw_train.columns[1:]].values
    X_test = raw_test[raw_test.columns[1:]].values
    y_train = one_hot(raw_train['label'].values)
    y_test = one_hot(raw_test['label'].values)
    # rescale the data
    X_train_s, X_test_s = scale(X_train), scale(X_test)

    # train network
    network = NLayerNeural(X_train_s.shape[1], y_train.shape[1], hidden=(64,))
    mse_logg, acc_logg = [], []
    for _ in trange(20):
        mse_logg.append(train_epoch(network, X_train_s, y_train, alpha=0.5))
        acc_logg.append(accuracy(network, X_test_s, y_test))

    # print accuracy change, make plots
    print(acc_logg)
    plot(mse_logg, acc_logg)


if __name__ == '__main__':
    main()
