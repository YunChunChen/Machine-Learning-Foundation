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
    train_data, train_label = get_data('./hw4_train.dat')
    train_x = train_data[:120]
    val_x = train_data[120:]
    train_y = train_label[:120]
    val_y = train_label[120:]
    test_x, test_y = get_data('./hw4_test.dat')

    Etrain_list = []
    Eval_list = []

    lambda_list = range(2, -11, -1)
    for l in lambda_list:
        w = w_reg(np.array(train_x), np.array(train_y), 10**l)
        Etrain_list.append(zero_one_error(w, train_x, train_y))
        Eval_list.append(zero_one_error(w, val_x, val_y))

    pos = Etrain_list.index(min(Etrain_list))
    print "lambda with min Etrain:", 10**(lambda_list[pos])
    print "Corresponding Etrain:", Etrain_list[pos]
    print "Corresponding Eval:", Eval_list[pos]
    
    pos = Eval_list.index(min(Eval_list))
    print "lambda with min Eval:", 10**(lambda_list[pos])
    print "Corresponding Etrain:", Etrain_list[pos]
    print "Corresponding Eval:", Eval_list[pos]

    Etrain_plot, = plt.plot(lambda_list, Etrain_list, label='Etrain')
    Eval_plot, = plt.plot(lambda_list, Eval_list, label='Eval')

    plt.xlabel('Lambda')
    plt.ylabel('Loss')
    plt.title('Lambda vs. Loss')
    plt.legend(handles=[Etrain_plot, Eval_plot], prop={'size': 15})
    plt.savefig('p8.pdf')

if __name__ == '__main__':
    main()
