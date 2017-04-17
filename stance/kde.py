from sklearn.neighbors.kde import KernelDensity
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

def plotKDE(infile, title, outfile):
    X = np.loadtxt(infile)
    X = X[np.isfinite(X)]
    X = X.reshape(-1, 1)
    kde = KernelDensity(kernel='gaussian', bandwidth=4).fit(X)
    X_plot = np.linspace(-200, 200, 1000)[:, np.newaxis]
    fig, ax = plt.subplots()
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))

    log_dens = kde.score_samples(X_plot)
    ax.fill(X_plot[:, 0], np.exp(log_dens), fc='g')
    ax.set_xticks(np.arange(-200, 200, 50))
    ax.axvspan(-200, -40, facecolor='b', alpha=0.5)
    ax.axvspan(30, 200, facecolor='r', alpha=0.5)
    ax.set_xlim([-200,200])
    ax.set_ylim([0,.018])
    ax.set_title(title)
    ax.set_xlabel('Score')
    ax.set_ylabel('Density')
    plt.savefig(outfile)

def main():
    plotKDE("data/lib_kde.txt", "Liberal Data", "data/lib_kde_graph.png")
    plotKDE("data/cons_kde.txt", "Conservative Data", "data/cons_kde_graph.png")
    plotKDE("data/neutral_kde.txt", "Neutral Data", "data/neutral_kde_graph.png")

if __name__ == '__main__':
    main()
