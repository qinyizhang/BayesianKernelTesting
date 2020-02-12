'''Posterior Inference for M and theta'''
import numpy as np
from tools import GenericTests
from kerpy.GaussianKernel import GaussianKernel
from scipy.stats import multivariate_normal, bernoulli
from scipy.spatial.distance import squareform, pdist


class HMCwithinGibbsObject(object):
    def __init__(self, X, Y, Z, sm_null, sm_alter, num_HMCiter, num_totalsim, theta_initial,
                 NUTS_stepsize, NUTS_maxtreedepth, ifindependent=True):
        self.X = X
        self.Y = Y
        self.Z = Z
        self.sm_null = sm_null
        self.sm_alter = sm_alter
        self.num_HMCiter = num_HMCiter
        self.num_totalsim = num_totalsim
        self.theta_initial = theta_initial
        self.NUTS_stepsize = NUTS_stepsize
        self.NUTS_maxtreedepth = NUTS_maxtreedepth
        self.ifindependent = ifindependent

    def sample_posterior_theta(self, M):
        n, d = np.shape(self.X)
        m = np.shape(self.Z)[0]
        sm_data = {'n': n, 'm': m, 'd': d, 'x': self.X, 'y': self.Y, 'z': self.Z}
        num_iter = self.num_HMCiter
        NUTS_stepsizejitter = 0.02
        if M == 0:
            fit = self.sm_null.sampling(data=sm_data, iter=num_iter, chains=1, algorithm='NUTS', n_jobs=1,
                                        init = self.theta_initial,
                                        control={'stepsize': self.NUTS_stepsize,
                                                 'stepsize_jitter': NUTS_stepsizejitter,
                                                 'max_treedepth': self.NUTS_maxtreedepth},
                                        check_hmc_diagnostics=False, refresh=-1)
        elif M == 1:
            fit = self.sm_alter.sampling(data=sm_data, iter=num_iter, chains=1, algorithm='NUTS', n_jobs=1,
                                         init = self.theta_initial,
                                         control={'stepsize': self.NUTS_stepsize,
                                                  'stepsize_jitter': NUTS_stepsizejitter,
                                                  'max_treedepth': self.NUTS_maxtreedepth},
                                         check_hmc_diagnostics=False, refresh=-1)
        else:
            raise(NotImplementedError)
        self.NUTS_stepsize = fit.get_sampler_params(inc_warmup=True)[0]['stepsize__'][num_iter - 1]
        self.NUTS_maxtreedepth = fit.get_sampler_params(inc_warmup=True)[0]['treedepth__'][num_iter - 1]
        la = fit.extract(pars={'theta'}, permuted=False, inc_warmup=True)
        theta_inferred = la['theta']
        print("inferred theta:", theta_inferred)
        if num_iter == 1:
            return theta_inferred
        else:
            return theta_inferred[num_iter -1]


    def sample_posterior_M(self, thetaval):
        n = np.shape(self.X)[0]
        m = np.shape(self.Z)[0]
        K = GaussianKernel(thetaval)
        Kxz = K.kernel(self.X, self.Z)
        Kyz = K.kernel(self.Y, self.Z)
        G = Kxz - Kyz # Compute the observations
        Delta_val = np.mean(G, axis=0)
        Dzz = squareform(pdist(self.Z, 'sqeuclidean')) # Compute the R matrix
        R = np.exp(- Dzz / float(4 * thetaval ** 2)) + 10**(-8) * np.eye(m)
        H = np.eye(n) - np.ones((n, n)) / np.float(n)
        if self.ifindependent:
            Sigma1 = Kxz.T.dot(H.dot(Kxz)) / (n ** 2)
            Sigma2 = Kyz.T.dot(H.dot(Kyz)) / (n ** 2)
            Sigma = Sigma1 + Sigma2 + 10 ** (-8) * np.eye(m)
        else:
            Sigma = np.transpose(G).dot(H.dot(G)) / np.float(n ** 2) + 10 ** (-8) * np.eye(m)

        BF = multivariate_normal.pdf(Delta_val, cov=Sigma) / multivariate_normal.pdf(Delta_val, cov=R + Sigma)
        Prob_M1 = 1 / np.float(BF + 1)
        mm = bernoulli.rvs(Prob_M1, size=1)
        if mm == 0:
            M = 0
        else:
            M = 1
        return BF, M


    def sample_posterior_joint(self):
        nsim = self.num_totalsim
        BFmat = np.zeros((nsim,))
        Mmat = np.zeros((nsim,))
        thetamat = np.zeros((nsim+1,))
        thetamat[0] = self.theta_initial
        for ii in range(nsim):
            print("iteration", ii)
            BFmat[ii], Mmat[ii] = self.sample_posterior_M(thetamat[ii])
            print("M value:", Mmat[ii])
            thetamat[ii+1] = self.sample_posterior_theta(Mmat[ii])
            self.theta_initial = thetamat[ii+1]
            print("theta value:", thetamat[ii+1])
        return BFmat, Mmat, thetamat[range(0,nsim)]

