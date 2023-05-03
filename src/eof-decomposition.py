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

# Pre-processed data: detrended, deseasonalized, dedrafted
flux_clean = xr.open_dataset(inDirName+interim_data_folder+'flux_clean')
flux_clean = flux_clean.timeMonthly_avg_landIceFreshwaterFlux

# Read geoJSON region feature file as GeoDataFrame
iceshelvesmask = gpd.read_file(inDirName + iceShelvesShape_filepath)
# Convert to south polar stereographic projection
#icems = iceshelvesmask.to_crs({'init': 'epsg:3031'}); # This has been deprecated
icems = iceshelvesmask.to_crs('epsg:3031');
crs = ccrs.SouthPolarStereo();

"""
print("Applying PCA on original data...")
model = EOF(flux_clean)
model.solve()
print("PCA/EOF complete...")
eofs = model.eofs()
pcs = model.pcs()
nmodes = model.n_modes
varexpl = model.explained_variance_ratio()
pcs_eig = model.pcs(1)
eofs_eig = model.eofs(1)

print("Saving files...")
eofs.to_netcdf(inDirName+processed_data_folder+'eofs.nc')
pcs.to_netcdf(inDirName+processed_data_folder+'pcs.nc')
eofs_eig.to_netcdf(inDirName+processed_data_folder+'eofs_eig.nc')
pcs_eig.to_netcdf(inDirName+processed_data_folder+'pcs_eig.nc')
varexpl.to_netcdf(inDirName+processed_data_folder+'varexpl.nc')
"""

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

print("Saving files...")
eofs.to_netcdf(inDirName+processed_data_folder+'norm_eofs.nc')
pcs.to_netcdf(inDirName+processed_data_folder+'norm_pcs.nc')
eofs_eig.to_netcdf(inDirName+processed_data_folder+'norm_eofs_eig.nc')
pcs_eig.to_netcdf(inDirName+processed_data_folder+'norm_pcs_eig.nc')
varexpl.to_netcdf(inDirName+processed_data_folder+'norm_varexpl.nc')





