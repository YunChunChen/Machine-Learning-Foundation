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

def w_reg(x, y, l):
    tmp = np.linalg.inv(np.matmul(np.transpose(x), x) + l * np.eye(len(x[0])))
    z = np.matmul(tmp, np.transpose(x))
    return np.matmul(z, y)

def main():
    train_x, train_y = get_data('./hw4_train.dat')
    test_x, test_y = get_data('./hw4_test.dat')

    Ein_list = []
    Eout_list = []

    lambda_list = range(2, -11, -1)
    for l in lambda_list:
        w = w_reg(np.array(train_x), np.array(train_y), 10**l)
        Ein_list.append(zero_one_error(w, train_x, train_y))
        Eout_list.append(zero_one_error(w, test_x, test_y))

    pos = Ein_list.index(min(Ein_list))
    print "lambda with min Ein:", 10**(lambda_list[pos])
    print "Corresponding Ein:", Ein_list[pos]
    print "Corresponding Eout:", Eout_list[pos]
    
    pos = Eout_list.index(min(Eout_list))
    print "lambda with min Eout:", 10**(lambda_list[pos])
    print "Corresponding Ein:", Ein_list[pos]
    print "Corresponding Eout:", Eout_list[pos]

    Ein_plot, = plt.plot(lambda_list, Ein_list, label='Ein')
    Eout_plot, = plt.plot(lambda_list, Eout_list, label='Eout')

    plt.xlabel('Lambda')
    plt.ylabel('Loss')
    plt.title('Lambda vs. Loss')
    plt.legend(handles=[Ein_plot, Eout_plot], prop={'size': 15})
    plt.savefig('p7.pdf')

if __name__ == '__main__':
    main()
