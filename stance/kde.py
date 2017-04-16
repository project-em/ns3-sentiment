from sklearn.neighbors.kde import KernelDensity
import numpy as np
import matplotlib.pyplot as plt

def plotKDE(kde, filename):
    X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]

    log_dens = kde.score_samples(X_plot)
    plt.fill(X_plot[:, 0], np.exp(log_dens), fc='#AAAAFF')
    plt.text(-3.5, 0.31, "Gaussian Kernel Density")
    plt.savefig(filename)

def main():

    libX = np.loadtxt("data/lib_kde.txt")
    lib_kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(libX)
    plotKDE(lib_kde, "data/lib_kde_graph.png")


if __name__ == '__main__':
    main()
