'''
Bayes Factor with fixed theta value.
'''

import numpy as np
from numpy.random import normal, laplace
from kerpy.GaussianKernel import GaussianKernel
from HMCwithinGibbsBF import HMCwithinGibbsObject


def compute_ProbM1(X,Y,Z, thetaval, Independent=True):
    num_theta = np.shape(thetaval)[0]
    BF = np.zeros((num_theta,1))
    Prob_M1 = np.zeros((num_theta,1))
    for ii in range(num_theta):
        myobj = HMCwithinGibbsObject(X, Y, Z, None, None, 0, 0, 1., 0, 0, ifindependent=Independent)
        BF[ii, :], _ = myobj.sample_posterior_M(thetaval[ii])
        Prob_M1[ii, :] = 1 / np.float(BF[ii,:] + 1)
    return BF, Prob_M1


# Compute BF value over a range of theta averaged over 100 simulation of BF:
# For independent samples:
def compute_average_BF(n, m, nsim, Xmean, Xstd, Ymean, Ystd, Ydist='Normal',
                       thetaval = np.linspace(0.001, 60, 100), rdseed = 12231):
    if thetaval is None:
        ntheta = 1
    else:
        ntheta = np.shape(thetaval)[0]
    BFmat = np.zeros((nsim, ntheta))
    ProbM1mat = np.zeros((nsim, ntheta))

    for ll in range(nsim):
        np.random.seed(rdseed)
        X = np.reshape(normal(Xmean, Xstd, n), (n, 1))
        Zx = normal(Xmean, Xstd, int(m / 2))
        if Ydist == 'Normal':
            Y = np.reshape(normal(Ymean, Ystd, n), (n, 1))
            Zy = normal(Ymean, Ystd, int(m / 2))
        elif Ydist == 'Laplace':
            Y = np.reshape(laplace(Ymean, Ystd, n), (n, 1))
            Zy = laplace(Ymean, Ystd, int(m / 2))
        else:
            raise NotImplementedError

        Z = np.reshape(np.concatenate((Zx, Zy)), (m, 1))
        if thetaval is None:
            K = GaussianKernel()
            XY = np.reshape(np.concatenate((X, Y)), (2 * n, 1))
            median_heuristic_theta = K.get_sigma_median_heuristic(XY)
            BF_val, prob_M1_val = compute_ProbM1(X, Y, Z, np.array([median_heuristic_theta]), Independent=True)
        else:
            BF_val, prob_M1_val = compute_ProbM1(X, Y, Z, thetaval, Independent=True)
            median_heuristic_theta = None

        BFmat[ll, :] = BF_val.reshape(-1)
        ProbM1mat[ll,:] = prob_M1_val.reshape(-1)
        rdseed += 1
    return BFmat, ProbM1mat, median_heuristic_theta




# Example Normal vs Laplace comparison:
n = 500
m = 40
nsim = 10
Xmean = 0
Xstd = 1.

list_theta = np.linspace(0.01, 40, 60)

Ymean = 0
Ystd = 1.6
distributionY = 'Laplace'

BFmatNL, ProbM1NL, median_heuristic_theta = compute_average_BF(n, m, nsim, Xmean, Xstd,
                                                               Ymean, Ystd, Ydist=distributionY,
                                                               thetaval = list_theta)

BF_av = BFmatNL.sum(axis =0)/np.float(nsim)
M0_av = (BF_av /(1+ BF_av))
print("Average Prob of M = 0:", M0_av)
print("Average value of BF:", BF_av)
print("median heuristic bandwidth:", median_heuristic_theta)


idx = np.where(BF_av == np.min(BF_av))[0]
print("Best theta:", list_theta[idx[0]])
print("Best BF:", BF_av[idx])
print("Best M0:", M0_av[idx])












