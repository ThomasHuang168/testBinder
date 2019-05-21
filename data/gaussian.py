import numpy as np
from scipy.integrate import quad, dblquad
from scipy.special import xlogy


class Gaussian():
    def __init__(self, sample_size=400, rho=0.9, mean=[0, 0], varValue=0):
        self.sample_size = sample_size
        self.mean = mean
        self.rho = rho

    @property
    def data(self):
        """[summary]
        Returns:
            [np array] -- [N by 2 matrix]
        """
        dim = len(self.mean)
        cov = (np.identity(dim)+self.rho*np.identity(dim)[::-1]).tolist()
        return np.random.multivariate_normal(
            mean=self.mean,
            cov=cov,
            size=self.sample_size)

    @property
    def ground_truth(self):
        dim = len(self.mean)
        if dim % 2 == 1:
            raise ValueError("dimension of gaussian is assummed to be even")
        len_X = dim//2
        return -0.5*np.log(1-self.rho**2)*len_X

    def plot_i(self, ax, xs, ys):
        if len(self.mean) != 2:
            raise ValueError("Only 2-dimension gaussian can be plotted")
        i_ = [self.I(xs[i, j], ys[i, j]) for j in range(ys.shape[1])
              for i in range(xs.shape[0])]
        i_ = np.array(i_).reshape(xs.shape[0], ys.shape[1])
        i_ = i_[:-1, :-1]
        i_min, i_max = -np.abs(i_).max(), np.abs(i_).max()
        c = ax.pcolormesh(xs, ys, i_, cmap='RdBu', vmin=i_min, vmax=i_max)
        # set the limits of the plot to the limits of the data
        ax.axis([xs.min(), xs.max(), ys.min(), ys.max()])
        return ax, c

    def I(self, x, y):
        dim = len(self.mean)
        if dim % 2 == 1:
            raise ValueError("dimension of gaussian is assummed to be even")
        covMat, mu = (np.identity(dim)+self.rho*np.identity(dim)
                      [::-1]), np.array(self.mean)

        def fxy(x, y):
            X = np.array([x, y])
            temp1 = np.matmul(
                np.matmul(X-mu, np.linalg.inv(covMat)), (X-mu).transpose())
            return np.exp(-.5*temp1) / (2*np.pi * np.sqrt(np.linalg.det(covMat)))

        def fx(x):
            return np.exp(-(x-mu[0])**2/(2*covMat[0, 0])) / np.sqrt(2*np.pi*covMat[0, 0])

        def fy(y):
            return np.exp(-(y-mu[1])**2/(2*covMat[1, 1])) / np.sqrt(2*np.pi*covMat[1, 1])

        return np.log(fxy(x, y)/(fx(x)*fy(y)))


if __name__ == '__main__':
    import os
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    eq = []
    integral = []
    rhos = []
    log_rhos = []
    for i in range(12):
        rho = 1 - 1/(10**i)
        rhos.append(rho)
        log_rhos.append(-i)
        gaus = Gaussian(200, [[1, rho], [rho, 1]], [0, 0], rho)
        integral.append(gaus.ground_truth)
        eq.append(-0.5*np.log(1-rho*rho))
    
    # Plot Ground Truth MI
    fig, axs = plt.subplots(1, 2, figsize=(16*2, 9))
    ax = axs[0]
    ax.plot(rhos, eq, color='r', marker='o', label='equation')
    ax.plot(rhos, integral, color='g', marker='x', label='integral')
    ax.legend()
    ax.set_xlabel('rho')
    ax.set_ylabel('mutual information')
    ax.set_title("mi ground truth comparison")

    ax = axs[1]
    ax.plot(log_rhos, eq, color='r', marker='o', label='equation')
    ax.plot(log_rhos, integral, color='g', marker='x', label='integral')
    ax.legend()
    ax.set_xlabel('log(1-rho)')
    ax.set_ylabel('mutual information')
    ax.set_title("mi ground truth comparison")

    figName = os.path.join("minee/experiments",
                           "mi_ground_truth_compare_Test.png")
    fig.savefig(figName, bbox_inches='tight')
    plt.close()
