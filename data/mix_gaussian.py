import numpy as np
from scipy.integrate import quad, dblquad
from scipy.special import xlogy


class MixedGaussian():
    # Mixture of two bivariate gaussians
    #
    # data(mix,Mode,Rho,N) generates N samples with
    # mix: mixing ration between 0 and 1
    # Rho[0] correlation for the first bivariate gaussian and Rho[1] for the second
    # Mode[0] separation between the two bivariate gaussians along the x-axis and Mode[1] is the separation along the y-axis

    def __init__(self, sample_size=400, mean1=0, mean2=0, rho1=0.9, rho2=-0.9, mix=0.5, theta=0):
        self.sample_size = sample_size
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        covMat1 = np.array([[1, rho1], [rho1, 1]])
        covMat2 = np.array([[1, rho2], [rho2, 1]])
        self.sample_size = sample_size
        self.mix = mix
        self.covMat1 = np.matmul(np.matmul(R, covMat1), R.transpose())
        self.covMat2 = np.matmul(np.matmul(R, covMat2), R.transpose())
        self.mu = np.array([mean1, mean2])
        self.name = 'bimodal'

    @property
    def data(self):
        """[summary]

        Returns:
            [np.array] -- [N by 2 matrix]
        """
        N1 = int(self.mix*self.sample_size)
        N2 = self.sample_size-N1
        temp1 = np.random.multivariate_normal(mean=self.mu,
                                              cov=self.covMat1,
                                              size=N1)
        temp2 = np.random.multivariate_normal(mean=-self.mu,
                                              cov=self.covMat2,
                                              size=N2)
        X = np.append(temp1, temp2, axis=0)
        np.random.shuffle(X)
        return X

    @property
    def ground_truth(self):
        # MI(mix,Rho,Mode) computes mutual information
        # for mixture of two bivariate gaussians with
        # mix, Rho, and Mode as above.
        # Numerical integration may cause issues for Mode values above 10 and / or correlation above .99
        # (performance suffers for choices that results in large Mode[i]*Rho[i])
        # can possibly be resolved by avoiding exponentials or by using other integration methods
        mix, covMat1, covMat2, mu = self.mix, self.covMat1, self.covMat2, self.mu

        def fxy(x, y):
            X = np.array([x, y])
            temp1 = np.matmul(
                np.matmul(X-mu, np.linalg.inv(covMat1)), (X-mu).transpose())
            temp2 = np.matmul(
                np.matmul(X+mu, np.linalg.inv(covMat2)), (X+mu).transpose())
            return mix*np.exp(-.5*temp1) / (2*np.pi * np.sqrt(np.linalg.det(covMat1))) \
                + (1-mix)*np.exp(-.5*temp2) / \
                (2*np.pi * np.sqrt(np.linalg.det(covMat2)))

        def fx(x):
            return mix*np.exp(-(x-mu[0])**2/(2*covMat1[0, 0])) / np.sqrt(2*np.pi*covMat1[0, 0]) \
                + (1-mix)*np.exp(-(x+mu[0])**2/(2*covMat2[0, 0])
                                 ) / np.sqrt(2*np.pi*covMat2[0, 0])

        def fy(y):
            return mix*np.exp(-(y-mu[1])**2/(2*covMat1[1, 1])) / np.sqrt(2*np.pi*covMat1[1, 1]) \
                + (1-mix)*np.exp(-(y+mu[1])**2/(2*covMat2[1, 1])
                                 ) / np.sqrt(2*np.pi*covMat2[1, 1])

        lim = np.inf
        hx = quad(lambda x: -xlogy(fx(x), fx(x)), -lim, lim)
        isReliable = hx[1]
        hy = quad(lambda y: -xlogy(fy(y), fy(y)), -lim, lim)
        isReliable = np.maximum(isReliable, hy[1])
        hxy = dblquad(lambda x, y: -xlogy(fxy(x, y), fxy(x, y)), -
                      lim, lim, lambda x: -lim, lambda x: lim)
        isReliable = np.maximum(isReliable, hxy[1])
        return hx[0] + hy[0] - hxy[0]

    def plot_i(self, ax, xs, ys):
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
        mix, covMat1, covMat2, mu = self.mix, self.covMat1, self.covMat2, self.mu

        def fxy(x, y):
            X = np.array([x, y])
            temp1 = np.matmul(
                np.matmul(X-mu, np.linalg.inv(covMat1)), (X-mu).transpose())
            temp2 = np.matmul(
                np.matmul(X+mu, np.linalg.inv(covMat2)), (X+mu).transpose())
            return mix*np.exp(-.5*temp1) / (2*np.pi * np.sqrt(np.linalg.det(covMat1))) \
                + (1-mix)*np.exp(-.5*temp2) / \
                (2*np.pi * np.sqrt(np.linalg.det(covMat2)))

        def fx(x):
            return mix*np.exp(-(x-mu[0])**2/(2*covMat1[0, 0])) / np.sqrt(2*np.pi*covMat1[0, 0]) \
                + (1-mix)*np.exp(-(x+mu[0])**2/(2*covMat2[0, 0])
                                 ) / np.sqrt(2*np.pi*covMat2[0, 0])

        def fy(y):
            return mix*np.exp(-(y-mu[1])**2/(2*covMat1[1, 1])) / np.sqrt(2*np.pi*covMat1[1, 1]) \
                + (1-mix)*np.exp(-(y+mu[1])**2/(2*covMat2[1, 1])
                                 ) / np.sqrt(2*np.pi*covMat2[1, 1])

        return np.log(fxy(x, y)/(fx(x)*fy(y)))


if __name__ == '__main__':
    bimodel = MixedGaussian()
    data = bimodel.data
    import os
    import matplotlib
    import matplotlib.pyplot as plt

    # Plot Ground Truth MI
    fig, axs = plt.subplots(1, 2, figsize=(45, 15))
    ax = axs[0]
    ax.scatter(data[:, 0], data[:, 1], color='r', marker='o')

    ax = axs[1]
    Xmax = max(data[:, 0])+1
    Xmin = min(data[:, 0])-1
    Ymax = max(data[:, 1])+1
    Ymin = min(data[:, 1])-1
    x = np.linspace(Xmin, Xmax, 300)
    y = np.linspace(Ymin, Ymax, 300)
    xs, ys = np.meshgrid(x, y)
    ax, c = bimodel.plot_i(ax, xs, ys)
    fig.colorbar(c, ax=ax)
    ax.set_title("i(X;Y)")
    figName = os.path.join("experiments", "bimodel_rho=0.9_i_XY.png")
    fig.savefig(figName, bbox_inches='tight')
    plt.close()

    print(bimodel.ground_truth)