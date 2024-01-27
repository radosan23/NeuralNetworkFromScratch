import numpy as np
import pandas as pd
import os
import requests
from matplotlib import pyplot as plt


def one_hot(data: np.ndarray) -> np.ndarray:
    y_train = np.zeros((data.size, data.max() + 1))
    rows = np.arange(data.size)
    y_train[rows, data] = 1
    return y_train


def plot(loss_history: list, accuracy_history: list, filename='plot'):
    # function to visualize learning process at stage 4
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


def download_data():
    if not os.path.exists('../Data'):
        os.mkdir('../Data')
    # Download data if it is unavailable.
    if ('fashion-mnist_train.csv' not in os.listdir('../Data') and
            'fashion-mnist_test.csv' not in os.listdir('../Data')):
        print('Train dataset loading.')
        url = "https://www.dropbox.com/s/5vg67ndkth17mvc/fashion-mnist_train.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/fashion-mnist_train.csv', 'wb').write(r.content)
        print('Loaded.')
        print('Test dataset loading.')
        url = "https://www.dropbox.com/s/9bj5a14unl5os6a/fashion-mnist_test.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/fashion-mnist_test.csv', 'wb').write(r.content)
        print('Loaded.')


def scale(data: np.ndarray) -> np.ndarray:
    return data / data.max()


def xavier(n_in: int, n_out: int) -> np.ndarray:
    limit = np.sqrt(6) / np.sqrt(n_in + n_out)
    return np.random.uniform(-limit, limit, (n_in, n_out))


def sigmoid(x: float) -> float:
    return 1 / (1 + np.e ** -x)


def main():
    download_data()
    # Read train, test data.
    raw_train = pd.read_csv('../Data/fashion-mnist_train.csv')
    raw_test = pd.read_csv('../Data/fashion-mnist_test.csv')

    X_train = raw_train[raw_train.columns[1:]].values
    X_test = raw_test[raw_test.columns[1:]].values

    y_train = one_hot(raw_train['label'].values)
    y_test = one_hot(raw_test['label'].values)

    X_train_s, X_test_s = scale(X_train), scale(X_test)
    answer_scale = [X_train_s[2, 778], X_test_s[0, 774]]
    answer_xavier = xavier(2, 3).flatten().tolist()
    answer_sigmoid = [sigmoid(x) for x in (-1, 0, 1, 2)]
    print(answer_scale, answer_xavier, answer_sigmoid)


if __name__ == '__main__':
    main()
