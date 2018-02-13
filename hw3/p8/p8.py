import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_data(filename):
    x = []
    y = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            xx = []
            for num in line.split():
                xx.append(float(num))
            x.append([1.0] + xx[:-1])
            y.append(xx[-1])
    return x, y

def sigmoid(x):
    return np.divide(1, 1 + np.exp(-x))

def single_grad(w, x, y):
    return -y * x * sigmoid(-y * np.dot(w, x))

def grad(w, x, y):
    gradient = np.zeros(21)
    for i in range(len(x)):
        gradient += -y[i] * x[i] * sigmoid(-y[i] * np.dot(w, x[i]))
    return np.divide(gradient, len(x))

def train(data, label, lr, sgd):
    x = np.array(data)
    y = np.array(label)
    w = np.zeros(21)
    Ein_list = []
    for i in range(2000):
        if sgd:
            idx = i % len(x)
            gradient = single_grad(w, x[idx], y[idx])
        else:
            gradient = grad(w, x, y)
        w = w - lr*gradient
        Ein = zero_one_error(w, x, y)
        Ein_list.append(Ein)
    return w, Ein_list

def zero_one_error(w, data, label):
    x = np.array(data)
    y = np.array(label)
    result = sigmoid(np.dot(x, w))
    ans = []
    for i in range(len(result)):
        if result[i] >= 0.5:
            ans.append(1)
        else:
            ans.append(-1)
    return np.mean(ans != y)

def main():
    train_x, train_y = get_data('./hw3_train.dat')
    test_x, test_y = get_data('./hw3_test.dat')
    w_GD, Ein_GD = train(train_x, train_y, 0.01, False)
    w_SGD, Ein_SGD = train(train_x, train_y, 0.01, True)

    line_GD, = plt.plot(range(2000), Ein_GD, color='b', label='GD')
    line_SGD, = plt.plot(range(2000), Ein_SGD, color='r', label='SGD')
    plt.xlabel('Round')
    plt.ylabel('Ein')
    plt.title('Round vs. Ein')
    plt.legend(handles=[line_GD, line_SGD], prop={'size': 15})
    plt.savefig('lr=0.01.pdf')

if __name__ == '__main__':
    main()
