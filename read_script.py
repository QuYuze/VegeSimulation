from netCDF4 import Dataset
import numpy as np

predictor2file = {'airTemp':'tas.nc', 'precip':'pr.nc', 'downwardLongwave':'rlns.nc', \
                  'downwardShortwave':'rsds.nc', 'plantAvailableWater':'paw.nc'}
# where is relative humidity?
predictors = {k:Dataset(v)['var1'][...].squeeze() for k,v in predictor2file.items()}
targetFile = 'jedi_output.nc'
targetVariable = 'NPP'
target = Dataset( targetFile)[targetVariable][...].squeeze()
landSeaMask = Dataset('landsea.nc')['var1'][...].squeeze()

# an example of making annual means
precip = predictors['precip']
annualMeanPrecip = precip.reshape((-1, 12, precip.shape[1], precip.shape[2])).mean( axis=1)
