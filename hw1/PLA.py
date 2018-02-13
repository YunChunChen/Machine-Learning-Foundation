import sys
import random

def read_file(filename):
    X = []
    Y = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            y = int(line.split('\t')[-1])
            Y.append(y)
            token = line.split('\t')[0]
            xx = [1] + [float(num) for num in token.split(' ')]
            X.append(xx)
    return X, Y

def sign(x):
    if x > 0:
        return 1
    else:
        return -1

def multiply(w, x):
    result = 0
    for i in range(len(w)):
        result += w[i] * x[i]
    return result

if __name__ == '__main__':
    filename = sys.argv[1]
    X, Y = read_file(filename)
    update_list = []
    for times in range(2000):
        W = [0] * 5
        updates = 0
        random.seed(times)
        sequence = random.sample(range(len(Y)), len(Y))
        while True:
            counter = 0
            for i in sequence:
                if Y[i] != sign(multiply(W, X[i])):
                    updates += 1
                    counter += 1
                    for idx in range(len(W)):
                        W[idx] = W[idx] + Y[i]*X[i][idx]
            if counter == 0:
                update_list.append(updates)
                break
    print "Number of average updates: {}".format(sum(update_list)/float(len(update_list)))