import random
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def sign(num):
    if num > 0:
        return 1
    else:
        return -1

def flip(num):
    p = random.random()
    if p < 0.1:
        return -1 * num
    else:
        return num

def get_data():
    return random.uniform(-1, 1)

def main():
    size = 1000
    iters = 1000
    eout_list = []
    Eout_list = []
    counter_list = []
    for i in range(iters):
        x = []
        y = []
        for it in range(size):
            x1 = get_data()
            x2 = get_data()
            x.append([1, x1, x2, x1*x2, x1*x1, x2*x2])
            y.append(flip(sign(x1*x1 + x2*x2 - 0.6)))
        X = np.array(x)
        Y = np.array(y)
        XT = np.transpose(X)
        XP = np.dot(np.linalg.inv(np.dot(XT, X)), XT)
        W = np.dot(XP, np.transpose(Y))
        coeff = np.array([-1, -0.05, 0.08, 0.13, 1.5, 1.5])
        hit = 0.0
        for idx in range(size):
            if sign(np.dot(W, np.transpose(X[idx]))) == sign(np.dot(coeff, np.transpose(X[idx]))):
                hit += 1
        test_x = []
        test_y = []
        for it in range(size):
            x1 = get_data()
            x2 = get_data()
            test_x.append([1, x1, x2, x1*x2, x1*x1, x2*x2])
            test_y.append(flip(sign(x1*x1 + x2*x2 - 0.6)))
        test_X = np.array(test_x)
        test_Y = np.array(test_y)
        miss = 0.0
        for idx in range(size):
            if sign(np.dot(W, np.transpose(test_X[idx]))) != sign(test_Y[idx]):
                miss += 1
        Eout = miss/size
        print "Eout:", Eout
        eout_list.append(Eout)
        if Eout not in Eout_list:
            Eout_list.append(Eout)
            counter_list.append(1)
        else:
            counter_list[Eout_list.index(Eout)] += 1
    print "Average Eout:", sum(eout_list)/size
    plt.hist(eout_list)
    plt.title('Histogram of Eout')
    plt.xlabel('Eout')
    plt.ylabel('Frequency')
    plt.savefig('p7.pdf')

if __name__ == '__main__':
    main()
