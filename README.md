# Bayesian Kernel Two-Sample Testing
This directory provides code and some illustrative examples for the Bayesian Kernel Two-Sample Testing project (url to appear). 
- SyntheticDataGen.py generates synthetic datasets used in the project. 
- HMCwithinGibbsBF.py provides the posterior inference scheme for the proposed Bayesian kernel two-sample test. 
- BayesFactor_fixtheta_table.py computes the Bayes factor with fixed theta values. This can be used to reproduce results
on posterior inference with fixed theta value. 
- PreCompile_stanmodel.py precompiles the stan models needed for posterior inference which will save computational time. 
- 2DGaussian_part1.py and 2DGaussian_part2.py are files to reproduce the results in the project for the 2 dimensional 
Gaussian experiment. They are to be used in Slurm by 
calling: 
```
sbatch SLURM_2DGaussian_part1.sh
```
(and respectively for part 2).
- The ChemistryData folder contains the real chemistry data used in the project (Section 8.2). 
The results can be reproduced by calling
```
sbatch SLURM_BFchemistry.sh
```
- cov_file folder contains the covariance structures used to generate multivariate Gaussian networks (Section 8.1).

The other experiments in the project can be runned similarly by replacing the data generating function and the relevant 
parameters. Python 3.7 was used. 
