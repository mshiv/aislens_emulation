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

# Define local plotting parameters
#sns.set_theme(style="whitegrid")
#sns.set_theme(style="ticks")
sns.set_theme(style="white")
sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
plt.rcParams.update({'font.size': 15})
plt.rc('font', family='sans-serif') 
plt.rc('font', serif='Helvetica Neue')
plt.rc('text', usetex=True)

# To update following with relative repository paths once data and code is up on Zenodo
# Current version uses the project template on Github.

# Define project repo path
#inDirName = '/Users/smurugan9/research/aislens/aislens_emulation/'
inDirName = '/storage/home/hcoda1/6/smurugan9/p-arobel3-0/filtergen/'
resultsDir = 'results/'

# DATA FILE PATHS
# Data containing regridded flux and SSH for 150 years
#regriddedFluxSSH_filepath = 'data/interim/RegriddedFluxSSH.nc'
# File contains all defined ice shelves
#iceShelvesShape_filepath = 'iceShelves.geojson'


# Original forcing data: raw MPAS-O output, mapped to the 10km resolution grid
# flux is freshwater flux
# ssh is sea surface height, used here as a proxy for ice draft / depth.

# ssh is used to plot the scatterplot of flux vs. draft for different ice shelves 
# and as input for the linear regression used in "dedrafting" the dataset

# Pre-processed data: detrended, deseasonalized, dedrafted
flux_clean = xr.open_dataset(inDirName+'flux_clean')
flux_clean = flux_clean.timeMonthly_avg_landIceFreshwaterFlux

# Add reconstructed datasets - seed random

# Catchment boundary masks for Antarctica, taken from ice shelf definitions in MPAS-Dev/geometric-features
# Source: https://github.com/MPAS-Dev/geometric_features/tree/main/geometric_data
# These have been combined into one file with 133 defined regions (polygons and multipolygons), 
# readable via the Geopandas package


"""
# Read geoJSON region feature file as GeoDataFrame
iceshelvesmask = gpd.read_file(inDirName + iceShelvesShape_filepath)
# Convert to south polar stereographic projection
#icems = iceshelvesmask.to_crs({'init': 'epsg:3031'}); # This has been deprecated
icems = iceshelvesmask.to_crs('epsg:3031');
crs = ccrs.SouthPolarStereo();
# Plot boundaries
plt.figure(figsize=(5,5))
ax1 = plt.subplot(121,projection=ccrs.SouthPolarStereo())
ax1.gridlines(color='whitesmoke',zorder=4)
icems[34:133].plot(ax=ax1,color='antiquewhite', linewidth=0,zorder=1)
icems[34:133].boundary.plot(ax=ax1,color='r', linewidth=0.2,zorder=3)
#icems[34:133].boundary.plot(ax=ax1,linewidth=0.25,color='lightgray',zorder=4)
#ax1.coastlines(resolution='10m', zorder=6,linewidth=0.75)
ax1.patch.set_facecolor(color='lightsteelblue')
#ax1.add_feature(cartopy.feature.LAND, color='ghostwhite')
ax1.add_feature(cartopy.feature.LAND, color='ghostwhite', zorder=2)
plt.title('Catchment Boundaries');
"""

sec_per_year = 365*24*60*60
rho_fw = 1000

def time_series(clipped_data):
    clipped_ts = clipped_data.sum(['y','x'])
    return clipped_ts

def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)
    #return butter(order, cutoff, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_highpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='high', analog=False)
    #return butter(order, cutoff, btype='low', analog=False)
    
def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

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

# Filter requirements.
order = 5
fs = 1.0       # sample rate, Hz
n_realizations = 10

cutoff_period_list = [12,24,60,120,180]

for cutoff_period in cutoff_period_list:
	cutoff = 1/cutoff_period  # desired cutoff frequency of the filter, Hz
	T = 1500.0         # months
	n = int(T * fs) # total number of samples
	t = np.linspace(0, T, n, endpoint=False)
	y = flux_clean.shape[1]
	x = flux_clean.shape[2]
	flux_clean_filtered = np.empty((flux_clean.shape[0],y,x))
	for i in range(y):
		for j in range(x):
			fl_filt = butter_lowpass_filter(flux_clean[:,i,j], cutoff, fs, order)
			flux_clean_filtered[:,i,j] = fl_filt
		print("filtered at {},{}".format(i,j))
	print("Filtering complete...")
	flux_clean_filtered = xr.DataArray(flux_clean_filtered,dims=flux_clean.dims,coords=flux_clean.coords,attrs=flux_clean.attrs)
	flux_reconstr_noise = flux_clean - flux_clean_filtered
	flux_clean_filtered.to_netcdf(inDirName+resultsDir+"filtered_{}.nc".format(cutoff_period))
	flux_reconstr_noise.to_netcdf(inDirName+resultsDir+"filtnoise_{}.nc".format(cutoff_period))
	flux_clean_filtered = xr.open_dataset(inDirName+resultsDir+"filtered_{}.nc".format(cutoff_period))
	flux_reconstr_noise = xr.open_dataset(inDirName+resultsDir+"filtnoise_{}.nc".format(cutoff_period))
	flux_clean_filtered = flux_clean_filtered.__xarray_dataarray_variable__
	flux_reconstr_noise = flux_reconstr_noise.__xarray_dataarray_variable__
	print("Normalizing data")
	flux_clean_tmean = flux_clean_filtered.mean('time')
	flux_clean_tstd = flux_clean_filtered.std('time')
	flux_clean_demeaned = flux_clean_filtered - flux_clean_tmean
	flux_clean_normalized = flux_clean_demeaned/flux_clean_tstd
	print("Normalization complete, applying PCA...")
	model = EOF(flux_clean_normalized)
	model.solve()
	print("PCA/EOF complete...")
	eofs = model.eofs()
	pcs = model.pcs()
	nmodes = model.n_modes
	varexpl = model.explained_variance_ratio()
	pcs_eig = model.pcs(1)
	eofs_eig = model.eofs(1)
	print("Save eofs/pcs/varexpl as .nc files...")
	eofs.to_netcdf(inDirName+resultsDir+"filteofs_{}.nc".format(cutoff_period))
	pcs.to_netcdf(inDirName+resultsDir+"filtpcs_{}.nc".format(cutoff_period))
	varexpl.to_netcdf(inDirName+resultsDir+"filtvarexpl_{}.nc".format(cutoff_period))
	# Define number of random Fourier realizations
	t_length = pcs.shape[0]
	# xeofs_pcs[:,i] when using PCA outputs
	new_fl = np.empty((n_realizations,pcs.shape[0],pcs.shape[1]))
	# Time limits for plotting
	t1 = 0
	tf = int(t_length/2)
	for i in range(n_realizations):
		for m in range(nmodes):
			fl = rpcs[:,m] # fluxpcs[:,i] when using PCA outputs
			fl_fourier = np.fft.rfft(fl)
			random_phases = np.exp(np.random.uniform(0,2*np.pi,int(len(fl)/2+1))*1.0j)
			fl_fourier_new = fl_fourier*random_phases
			new_fl[i,:,m] = np.fft.irfft(fl_fourier_new)
	print('calculated ifft for all realizations, all modes')
	# Generate dataset realizations
	# Note standardized input data to the initial PCA
	print("Generate realizations...")
	for i in range(n_realizations):
		flux_reconstr = generate_data(i, nmodes, 1)
		flux_reconstr = (flux_reconstr*flux_clean_tstd)+flux_clean_tmean
		flux_reconstr = flux_reconstr + flux_clean_reconstr_noise
		flux_reconstr.to_netcdf(inDirName+resultsDir+"filt_{}_REC{}.nc".format(cutoff_period,i))
		del flux_reconstr
		gc.collect()
		print('reconstructed realization # {}'.format(i))
	print("Workflow complete for cutoff period = {}".format(cutoff_period))












