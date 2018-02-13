import random
import plot

def sign(num):
    if num > 0:
        return 1
    else:
        return -1

def h(s, x, theta):
    return s * sign(x - theta)

def flip(num):
    p = random.random()
    if p < 0.2:
        return -1 * num
    else:
        return num

def get_data():
    return random.uniform(-1, 1)

def main():
    size = 20
    iters = 1000
    Ein_list = []
    Eout_list = []
    for i in range(iters):
        random.seed(i)
        Ein_best = 1.0
        theta_best = 0
        s_best = 0
        print("Iter:", i)
        x = []
        y = []
        while True:
            tmp = get_data()
            if tmp not in x:
                x.append(tmp)
                y.append(flip(sign(tmp)))
                if len(x) == size:
                    break
        for idx in range(size+1):
            if idx == 0:
                theta = (x[idx] - 1)/2
            elif idx == size:
                theta = (x[idx-1] + 1)/2
            else:
                theta = (x[idx-1] + x[idx])/2
            for s in [-1, 1]:
                error = 0
                for k in range(size):
                    if h(s, x[k], theta) != y[k]:
                        error += 1
                Ein = error / size
                if Ein < Ein_best:
                    Ein_best = Ein
                    theta_best = theta
                    s_best = s
        Eout = 0.5 + 0.3 * s_best * (abs(theta_best) - 1)
        print("Best Ein:", Ein_best, " Best Eout:", Eout)
        Ein_list.append(Ein_best)
        Eout_list.append(Eout)
    print("Mean Ein:", sum(Ein_list)/len(Ein_list))
    print("Mean Eout:", sum(Eout_list)/len(Eout_list))
    plot.scatter_plot(Ein_list, Eout_list)

if __name__ == '__main__':
    main()
