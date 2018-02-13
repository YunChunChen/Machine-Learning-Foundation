import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def scatter_plot(x, y):
    plt.title("Ein vs. Eout")
    plt.xlabel('Ein')
    plt.ylabel('Eout')
    plt.plot(x, y, 'o')
    plt.savefig('result.pdf')
