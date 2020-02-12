'''
Python script for 2 dimensional Gaussian experiment.
To be used in combination with the SLURM shell script.

We separate this experiment into small sample sizes (part1) and large sample sizes (part2) out of the consideration that the memory and time requirements in SLURM will be different.
'''

# Import necessary packages
import os
import itertools
from HMCwithinGibbsBF import HMCwithinGibbsObject
from time import sleep

from SyntheticDataGen import SyntheticDataGen
import numpy as np
import pickle


# Function we would like to calculate over a grid of parameters.
def posterior_simulation(para_tuple):
    sleep(0.1) #Sleep system for 0.1 seconds
    sm_null = pickle.load(open('NullModel.pkl', 'rb'))
    sm_alter = pickle.load(open('AlterModel.pkl', 'rb'))
    dist_param = para_tuple[0]
    Ymean, epsilonval = dist_param
    num_HMCiter = para_tuple[1]
    num_totalsim = 2000
    NUTS_stepsize = 0.05
    NUTS_maxtreedepth = 1.
    n = para_tuple[2]
    m = para_tuple[3]
    seed_val = para_tuple[4]
    np.random.seed(seed_val)
    X, Y = SyntheticDataGen.simple_gaussian_2d([10, 10], Ymean, epsilonval, nsim=n)
    Zx, Zy = SyntheticDataGen.simple_gaussian_2d([10, 10], Ymean, epsilonval, nsim=m)
    Z = np.concatenate((Zx, Zy), axis=0)
    results = {}
    results['X'] = X
    results['Y'] = Y
    results['Z'] = Z
    theta_initial = 1.
    mysimobj = HMCwithinGibbsObject(X, Y, Z, sm_null, sm_alter, num_HMCiter, num_totalsim, theta_initial,
                                    NUTS_stepsize, NUTS_maxtreedepth)
    BFmat, Mmat, thetamat = mysimobj.sample_posterior_joint()
    print('Parameters: ' + str(para_tuple))
    results['theta'] = thetamat
    results['M'] = Mmat
    results['BF'] = BFmat
    return results


# Obtain the environment variables to determine the parameters and the cpus per task
n_cpus = int(os.environ['SLURM_CPUS_PER_TASK'])
print('Using ' + str(n_cpus) + ' per job array task' )
slurm_id = int(os.environ['SLURM_ARRAY_JOB_ID'])
slurm_parameter = int(os.environ['SLURM_ARRAY_TASK_ID'])

# Our parameters we will like to grid search over
list_n = [500, 800]
list_m = [40]
list_distparam = [[[10, 10], 1], [[12, 10], 1], [[11.5, 11.5], 1], [[10,10], 2], [[10,10], 6], [[10, 10,], 10], [[10, 10], 20]]
list_HMCiter = [6]
list_seed = range(3376250, 3376260)


# Create a grid search vector using the itertools package
vector_grid = list(itertools.product(list_distparam, list_HMCiter, list_n, list_m, list_seed))


parameter = vector_grid[int(slurm_parameter)]
output = posterior_simulation(parameter)

# Save output and parameters to text file in the localhost node, which is where the computation is performed.
with open("/data/localhost/not-backed-up/qzhang/jobname_" + str(slurm_id) + '_' +
          str(slurm_parameter) + ".out", 'wb') as fh:
    pickle.dump(output, fh)

