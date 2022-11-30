import sys
import os
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib import animation, rc, rcParams, cycler
from IPython.display import HTML, display # NOTE: will require ffmpeg installation
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import rioxarray
from shapely.geometry import mapping
import seaborn as sns
from scipy.stats import pearsonr
from xarrayutils.utils import linear_trend, xr_linregress
from xeofs.xarray import EOF
import statsmodels.api as sm
import scipy
from sklearn.metrics import mean_squared_error
from itertools import product


import EOFdecomp
import surrogate_ts

# Define root folder path
################################################################
ROOTDIR = '/Users/smurugan9/research/aislens/aislens_emulation/'
################################################################

# Define parameters for ensemble run
################################################################
n_realizations = 5
################################################################

# DATA FILE PATHS
interim_data = 'data/interim/'
raw_data = 'data/raw/'
processed_data = 'data/processed/'
external_data = 'data/external/'

# Folder for output figures
figures = 'reports/figures/'
# Folder for scripts
src = 'src/'


# Input File Names
regriddedFluxSSH = 'RegriddedFluxSSH.nc' # Data containing regridded flux and SSH for 150 years
iceShelves = 'iceShelves.geojson' 		 # File contains all defined ice shelves
fluxClean = 'flux_clean.nc'				 # File with dedrafted landIceFreshwaterFlux time series for 125 years (removes 25 year "spin up" time)

# Load catchment masks
icems = gpd.read_file(inDirName + interim_data + iceShelves)
icems = icems.to_crs({'init': 'epsg:3031'});
crs = ccrs.SouthPolarStereo();

# Load cleaned flux data
fl = xr.open_dataset(inDirName + interim_data + fluxClean)
fl = fl.timeMonthly_avg_landIceFreshwaterFlux
fl = fl.dropna('time',how='all')
fl = fl.dropna('y',how='all')
fl = fl.dropna('x',how='all')
fl = fl.drop("month")

##############################
# EOF / PCA DECOMPOSITION
##############################

eofs, pcs, modes = EOFdecomp(flux_clean)

##############################
# FOURIER PHASE RANDOMIZATION 
##############################

t_length = pcs.shape[0] # Length of time series

# Generate multiple dataset realizations
# pcs[:,i] when using PCA outputs

new_fl = np.empty((n_realizations,t_length,modes))
for i in range(n_realizations):
	for m in range(modes):
		new_fl[i,:,m] = surrogate_ts(pcs[:,m])
