'''
Python script for the chemistry dataset to be used in combination with the
SLURM shell script.
'''

# Import necessary packages
import os
import itertools
import multiprocessing as mp
from HMCwithinGibbsBF import HMCwithinGibbsObject
from time import sleep


import numpy as np
import pandas as pd
import pystan
import pickle


def posterior_simulation(para_tuple):
    sleep(0.1) #Sleep system for 0.1 seconds
    
    sm_null = pickle.load(open('NullModel.pkl', 'rb'))
    sm_alter = pickle.load(open('AlterModel.pkl', 'rb'))
    n = para_tuple[0]
    m = para_tuple[1]
    num_HMCiter = para_tuple[2]
    XYindependent =para_tuple[3]
    
    Xdata = pd.read_csv('ChemistryData/CREST_2D' + '.csv')
    Ydata = pd.read_csv('ChemistryData/COD_2D' + '.csv')
    
    nx = Xdata.shape[0]
    ny = Ydata.shape[0]
    
    X = Xdata.iloc[np.random.choice(nx,n, replace=False)]
    Y = Ydata.iloc[np.random.choice(ny,n, replace=False)]
    X = X.values
    Y = Y.values
    
    Z1 = Xdata.iloc[np.random.choice(nx, m/2, replace=False)]
    Z2 = Ydata.iloc[np.random.choice(ny, m/2, replace=False)]
    Z = np.concatenate((Z1, Z2), axis=0)
    
    num_totalsim = 4000
    NUTS_stepsize = 0.1
    NUTS_maxtreedepth = 1.
    
    results = {}
    results['X'] = X
    results['Y'] = Y
    results['Z'] = Z
    theta_initial = 1.
    mysimobj = HMCwithinGibbsObject(X, Y, Z, sm_null, sm_alter, num_HMCiter, num_totalsim,
                                    theta_initial, NUTS_stepsize, NUTS_maxtreedepth,
                                    ifindependent=XYindependent)
    BFmat, Mmat, thetamat = mysimobj.sample_posterior_joint()
    print('Parameters: ' + str(para_tuple))
    results['theta'] = thetamat
    results['M'] = Mmat
    results['BF'] = BFmat
    return results


# Obtain the environment variables
n_cpus = int(os.environ['SLURM_CPUS_PER_TASK'])
print('Using ' + str(n_cpus) + ' per job array task' )
slurm_id = int(os.environ['SLURM_ARRAY_JOB_ID'])
slurm_parameter = int(os.environ['SLURM_ARRAY_TASK_ID'])

list_n = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800]
list_m = [40]
list_HMCiter = [6]
list_XYindependent = [True, False]
vector_grid = list(itertools.product(list_n, list_m, list_HMCiter, list_XYindependent))


parameter = vector_grid[int(slurm_parameter)]
output = posterior_simulation(parameter)


# Save output and parameters to text file in the localhost node, which is where the computation is performed.
with open("/data/localhost/not-backed-up/qzhang/jobname_" + str(slurm_id) + '_' +
          str(slurm_parameter) + ".out", 'wb') as fh:
    pickle.dump(output, fh)

