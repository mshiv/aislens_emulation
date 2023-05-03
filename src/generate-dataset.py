# GENERATE Normalized datasets

import sys
import os
import gc
import collections
import cartopy.crs as ccrs
import cartopy
from cartopy import mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams, cycler
from matplotlib import animation, rc
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import xarray as xr
import geopandas as gpd
import rioxarray
from xeofs.xarray import EOF, Rotator
import statsmodels.api as sm
import scipy
from scipy import signal
from scipy.signal import butter, lfilter, freqz
import pyproj
from shapely.geometry import mapping
import pandas as pd
import cmocean


def time_series(clipped_data):
    clipped_ts = clipped_data.sum(['y','x'])
    return clipped_ts

def generate_data(n_realization,mode,mode_skip):
    flux_reconstr = model.reconstruct_randomized_X(new_fl[n_realization],slice(1,mode,mode_skip))
    #flux_reconstr = flux_reconstr.dropna('time',how='all')
    #flux_reconstr = flux_reconstr.dropna('y',how='all')
    #flux_reconstr = flux_reconstr.dropna('x',how='all')
    #flux_reconstr = flux_reconstr.drop("month")
    return flux_reconstr

def clip_data(total_data, basin):
    clipped_data = total_data.rio.clip(icems.loc[[basin],'geometry'].apply(mapping))
    clipped_data = clipped_data.drop("month")
    return clipped_data

# Define project repo path
inDirName = '/Users/smurugan9/research/aislens/aislens_emulation/'
#inDirName = '/storage/home/hcoda1/6/smurugan9/p-arobel3-0/rmse/'

# DATA FILE PATHS
# Data containing regridded flux and SSH for 150 years
# regriddedFluxSSH_filepath = 'data/interim/RegriddedFluxSSH.nc'
# File contains all defined ice shelves
iceShelvesShape_filepath = 'data/interim/iceShelves.geojson'

# Relative directory paths for Data and Figures
figures_folderpath = 'reports/figures/'
interim_data_folder = 'data/interim/'
processed_data_folder = 'data/processed/'
randomized_realizations_path = 'randomized_realizations/'
cise_file_path = 'cise_data/'
std_file_path = 'standardized_rec_data/'
temp_filtered_path = 'temp_filtered_data/'

n_realizations = 3

# INPUT FILES

# Pre-processed data: detrended, deseasonalized, dedrafted
flux_clean = xr.open_dataset(inDirName+interim_data_folder+'flux_clean')
flux_clean = flux_clean.timeMonthly_avg_landIceFreshwaterFlux

# Read geoJSON region feature file as GeoDataFrame
iceshelvesmask = gpd.read_file(inDirName + iceShelvesShape_filepath)
# Convert to south polar stereographic projection
#icems = iceshelvesmask.to_crs({'init': 'epsg:3031'}); # This has been deprecated
icems = iceshelvesmask.to_crs('epsg:3031');
crs = ccrs.SouthPolarStereo();

print("Normalizing data..")
flux_clean_tmean = flux_clean.mean('time')
flux_clean_tstd = flux_clean.std('time')

flux_clean_demeaned = flux_clean - flux_clean_tmean
flux_clean_normalized = flux_clean_demeaned/flux_clean_tstd

print("Applying PCA on normalized data...")
model = EOF(flux_clean_normalized)
model.solve()
print("PCA/EOF complete...")
eofs = model.eofs()
pcs = model.pcs()
nmodes = model.n_modes
varexpl = model.explained_variance_ratio()
pcs_eig = model.pcs(1)
eofs_eig = model.eofs(1)


# Define number of random Fourier realizations
t_length = pcs.shape[0]
# xeofs_pcs[:,i] when using PCA outputs
new_fl = np.empty((n_realizations,pcs.shape[0],pcs.shape[1]))
# Time limits for plotting
t1 = 0
tf = int(t_length/2)

for i in range(n_realizations):
	for m in range(nmodes):
		fl = pcs[:,m] # fluxpcs[:,i] when using PCA outputs
		fl_fourier = np.fft.rfft(fl)
		random_phases = np.exp(np.random.uniform(0,2*np.pi,int(len(fl)/2+1))*1.0j)
		fl_fourier_new = fl_fourier*random_phases
		new_fl[i,:,m] = np.fft.irfft(fl_fourier_new)
print('calculated ifft for all realizations, all modes')

# Generate dataset realizations
# Note standardized input data to the initial PCA 

modes_used = [50, 100, 200, 500, 1500]


for mode_number in modes_used:
	for i in range(n_realizations):
		flux_reconstr = generate_data(i, mode_number, 1)
		flux_reconstr = (flux_reconstr*flux_clean_tstd)+flux_clean_tmean
		flux_reconstr.to_netcdf(inDirName+processed_data_folder+"REC_{}-modes_{}.nc".format(mode_number,i))
		del flux_reconstr
		gc.collect()
		print('reconstructed realization # {} for mode case: {}'.format(i,mode_number))