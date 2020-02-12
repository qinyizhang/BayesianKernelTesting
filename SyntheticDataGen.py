'''Generate Synthetic Datasets'''

import numpy as np
from numpy import sin, cos
from scipy import stats


class SyntheticDataGen(object):
    def __init__(self):
        pass

    # Simple 1 Dimensional Gaussian Distributions
    @staticmethod
    def simple_gaussian_1d(Xmean, Ymean, Xstd, Ystd, nsim= 100):
        X = np.random.normal(Xmean, Xstd, nsim)
        Y = np.random.normal(Ymean, Ystd, nsim)
        X = X.reshape((nsim,1))
        Y = Y.reshape((nsim,1))
        return X, Y

    # Simple 2 Dimensional Gaussian Distributions
    @staticmethod
    def simple_gaussian_2d(Xmean_loc, Ymean_loc, epsilon, alpha = 45, nsim=100):
        Xcov = [[1,0],[0,1]]
        Q = np.array([[cos(alpha), sin(alpha)],[-sin(alpha), cos(alpha)]])
        S = np.array([[epsilon, 0],[0, 1]])
        Ycov = Q.dot(S.dot(Q.T))
        X = np.random.multivariate_normal(Xmean_loc, Xcov, nsim)
        Y = np.random.multivariate_normal(Ymean_loc, Ycov, nsim)
        return X, Y

    # 2 by 2 Gaussian Blobs
    @staticmethod
    def gaussian_blobs_2d(mean_loc, epsilon, alpha = 45, nsimpb=100):
        # mean_loc: 1d array containing the x location
        # epsilon: the ratio (real number) between the largest to the smallest eval
        # rs = check_random_state(seed)
        nloc = np.shape(mean_loc)[0]
        X = np.zeros((nsimpb*nloc**2,2))
        Y = np.zeros((nsimpb*nloc**2,2))
        Xcov = [[1,0],[0,1]]
        Q = np.array([[cos(alpha), sin(alpha)],[-sin(alpha), cos(alpha)]])
        S = np.array([[epsilon, 0],[0, 1]])
        Ycov = Q.dot(S.dot(Q.T))
        count = 0
        for ii in range(nloc):
            for jj in range(nloc):
                Xmean = [mean_loc[ii], mean_loc[jj]]
                X[range(count*nsimpb, (count+1)*nsimpb), 0], X[range(count*nsimpb, (count+1)*nsimpb), 1] = \
                    np.random.multivariate_normal(Xmean, Xcov, nsimpb).T
                Y[range(count * nsimpb, (count + 1) * nsimpb), 0], Y[range(count * nsimpb, (count + 1) * nsimpb), 1] = \
                    np.random.multivariate_normal(Xmean, Ycov, nsimpb).T
                count = count + 1
        return X, Y

    # 2 by 2 Gaussian Blobs with noise in higher dimensions
    @staticmethod
    def gaussian_blobs_withnoise(mean_loc, epsilon, alpha = 45, nsimpb=100, d = 3):
        # mean_loc: 1d array containing the x location
        # epsilon: the ratio (real number) between the largest to the smallest eval
        # rs = check_random_state(seed)
        nloc = np.shape(mean_loc)[0]
        total_nsim = (nloc**2) * nsimpb
        X = np.zeros((nsimpb*nloc**2,2))
        Y = np.zeros((nsimpb*nloc**2,2))
        Xcov = [[1,0],[0,1]]
        Q = np.array([[cos(alpha), sin(alpha)],[-sin(alpha), cos(alpha)]])
        S = np.array([[epsilon, 0],[0, 1]])
        Ycov = Q.dot(S.dot(Q.T))
        count = 0
        for ii in range(nloc):
            for jj in range(nloc):
                Xmean = [mean_loc[ii], mean_loc[jj]]
                X[range(count*nsimpb, (count+1)*nsimpb), 0], X[range(count*nsimpb, (count+1)*nsimpb), 1] = \
                    np.random.multivariate_normal(Xmean, Xcov, nsimpb).T
                Y[range(count * nsimpb, (count + 1) * nsimpb), 0], Y[range(count * nsimpb, (count + 1) * nsimpb), 1] = \
                    np.random.multivariate_normal(Xmean, Ycov, nsimpb).T
                count = count + 1
        if d - 2 <= 0:
            return X, Y
        elif d - 2 > 0:
            noiseX = np.reshape(np.random.standard_normal(total_nsim), (total_nsim,1))
            noiseY = np.reshape(np.random.standard_normal(total_nsim), (total_nsim,1))
            X = np.concatenate((X, noiseX), axis=1)
            Y = np.concatenate((Y, noiseY), axis=1)
            return X, Y
        else:
            raise NotImplementedError

    # 1 Dimensional Gaussian Mixture Distribution
    @staticmethod
    def gaussian_mixture_comparison(n, mean1, mean2, std, mixprop = 0.5):
        X = np.zeros((n,1))
        Y = np.zeros((n,1))
        
        uX = np.random.uniform(size=n)
        idx = np.where(uX < mixprop)
        length_idx = np.shape(idx)[1]
        X[idx[0],] = np.random.normal(0, 1, length_idx).reshape((length_idx,1))
        X[list(set(range(n)) - set(idx[0])), ] = np.random.normal(4, 1, n-length_idx).reshape((n-length_idx,1))
        
        uY = np.random.uniform(size=n)
        idx_y = np.where(uY < mixprop)
        length_idx_y = np.shape(idx_y)[1]
        Y[idx_y[0],] = np.random.normal(mean1, std, length_idx_y).reshape((length_idx_y,1))
        Y[list(set(range(n)) - set(idx_y[0])), ] = np.random.normal(mean2, std, n-length_idx_y).reshape(n-length_idx_y,1)
        return X, Y

    # Correlated 1 Dimensional Gaussian and Laplace Distributions
    @staticmethod
    def correlated_gaussian_laplace(n, loc=0., scale=1., correlation=0.3):
        mvnorm = stats.multivariate_normal(mean=[0,0], cov=[[1., correlation], [correlation, 1.]])
        x = mvnorm.rvs(n)
        norm = stats.norm()
        x_unif = norm.cdf(x) # transform the bivariate normal to 2d uniform through cdf
        m1 = stats.norm(loc=0, scale=1)
        m2 = stats.laplace(loc=loc, scale=scale)
        X = m1.ppf(x_unif[:, 0]) # transform the first dimension to normal
        Y = m2.ppf(x_unif[:, 1]) # transform the second dimension to laplace
        X = np.reshape(X, (n,1))
        Y = np.reshape(Y, (n,1))
        return X, Y


