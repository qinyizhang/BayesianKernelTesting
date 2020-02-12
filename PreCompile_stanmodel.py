'''Precompile stan model'''

import pickle
import pystan

sm_alter = pystan.StanModel(file='nested-approach-primal-paired2samplestesting-multid-alter-SVD.stan')
sm_null = pystan.StanModel(file='nested-approach-primal-paired2samplestesting-multid-null-SVD.stan')

with open('NullModel.pkl', 'wb') as fnull:
    pickle.dump(sm_null, fnull)
with open('AlterModel.pkl', 'wb') as falter:
    pickle.dump(sm_alter, falter)


