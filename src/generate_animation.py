import sys
import os
os.environ['USE_PYGEOS'] = '0'
import gc
import collections

import cartopy.crs as ccrs
import cartopy
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams, cycler
from matplotlib import animation, rc
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.signal import butter, lfilter, freqz, bessel

import numpy as np
import xarray as xr
import geopandas as gpd
import rioxarray
from xeofs.xarray import EOF

import statsmodels.api as sm
import scipy
from scipy import signal
from sklearn.metrics import mean_squared_error
from itertools import product
import pyproj
from shapely.geometry import mapping
from xarrayutils.utils import linear_trend, xr_linregress
import pandas as pd
from IPython.display import HTML, display
import cmocean

# NOTE: Local xeofs to be submitted as pull-request (addition of xeofs.model.reconstruct_randomized_X method)

# Define local plotting parameters
#sns.set_theme(style="whitegrid")
#sns.set_theme(style="ticks")
sns.set_theme(style="whitegrid")
sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
plt.rcParams.update({'font.size': 20})
plt.rc('font', family='sans-serif') 
plt.rc('font', serif='Helvetica Neue')
plt.rc('text', usetex=True)

# To update following with relative repository paths once data and code is up on Zenodo
# Current version uses the project template on Github.

# Define project repo path
inDirName = '/Users/smurugan9/research/aislens/aislens_emulation/'

# DATA FILE PATHS
# Data containing regridded flux and SSH for 150 years
regriddedFluxSSH_filepath = 'data/interim/RegriddedFluxSSH.nc'
# File contains all defined ice shelves
iceShelvesShape_filepath = 'data/interim/iceShelves.geojson'

# Relative directory paths for Data and Figures
figures_folderpath = 'reports/figures/'
interim_data_folder = 'data/interim/'
processed_data_folder = 'data/processed/'
flux_dedrafted_data_path = 'dedrafted_flux_IMBIE/'
randomized_realizations_path = 'randomized_realizations/'
flux_dedrafted_iceshelves_data_path = 'iceshelves_dedrafted_flux/'
reconstructions_neofs_path = 'reconstructions_neofs/'
cise_file_path = 'cise_data/'
std_file_path = 'standardized_rec_data/'


pca_1 = '1_pca/'
pca_2 = '2_pca_normalized/'
pca_3 = '3_pca_normalized_nmodes/'
pca_4 = '4_lowpass_ pca_normalized/'
pca_5 = '5_sparsepca/'


# Original forcing data: raw MPAS-O output, mapped to the 10km resolution grid
# flux is freshwater flux
# ssh is sea surface height, used here as a proxy for ice draft / depth.

# ssh is used to plot the scatterplot of flux vs. draft for different ice shelves 
# and as input for the linear regression used in "dedrafting" the dataset

regridded_data = xr.open_dataset(inDirName+regriddedFluxSSH_filepath)
flux = regridded_data.timeMonthly_avg_landIceFreshwaterFlux
ssh = regridded_data.timeMonthly_avg_ssh

# Pre-processed data: detrended, deseasonalized, dedrafted
flux_clean = xr.open_dataset(inDirName+interim_data_folder+'flux_clean')
flux_clean = flux_clean.timeMonthly_avg_landIceFreshwaterFlux

# Dedrafted flux
flux_dedrafted = xr.open_dataset(inDirName+
                                 interim_data_folder+
                                 flux_dedrafted_iceshelves_data_path+'iceshelves_dedrafted_total.nc')
flux_dedrafted = flux_dedrafted.timeMonthly_avg_landIceFreshwaterFlux

# Flux datapoints extracted for individual ice shelves, used for the scatter plots
catchments_scatter = np.load(inDirName+interim_data_folder+"catchments_scatter.npy")
catchments_scatter_xr = xr.DataArray(catchments_scatter,dims={'basin','x','y'})
sec_per_year = 365*24*60*60
rho_fw = 1000
catchments_scatter = catchments_scatter*sec_per_year/rho_fw

# Add reconstructed datasets - seed random

# Catchment boundary masks for Antarctica, taken from ice shelf definitions in MPAS-Dev/geometric-features
# Source: https://github.com/MPAS-Dev/geometric_features/tree/main/geometric_data
# These have been combined into one file with 133 defined regions (polygons and multipolygons), 
# readable via the Geopandas package

# Read geoJSON region feature file as GeoDataFrame
iceshelvesmask = gpd.read_file(inDirName + iceShelvesShape_filepath)

# Convert to south polar stereographic projection
#icems = iceshelvesmask.to_crs({'init': 'epsg:3031'}); # This has been deprecated
icems = iceshelvesmask.to_crs('epsg:3031');
crs = ccrs.SouthPolarStereo();

sec_per_year = 365*24*60*60
rho_fw = 1000

# Plot boundaries

# Input data
# Ensure this has been multipled by sec_per_year/rho_fw for melt rate units

flux_clean = xr.open_dataset(inDirName+interim_data_folder+'flux_clean')
#Fvgen01 = xr.open_dataset(inDirName+interim_data_folder+cise_file_path+std_file_path+"spca_a01/spca_a01_REC0.nc")
#Fvgen1 = xr.open_dataset(inDirName+interim_data_folder+cise_file_path+std_file_path+"spca_a1/spca_a1_REC0.nc")
Fvgen3a = xr.open_dataset(inDirName+interim_data_folder+cise_file_path+std_file_path+"spcabatch_a3/spcabatch_a3_REC0.nc")
#Fvgen3b = xr.open_dataset(inDirName+interim_data_folder+cise_file_path+std_file_path+"spcabatch_a3/spcabatch_a3_REC1.nc")
#Fvgen10 = xr.open_dataset(inDirName+interim_data_folder+cise_file_path+std_file_path+"spcabatch_a10/spcabatch_a10_REC0.nc")
Fvrec0 = xr.open_dataset(inDirName+processed_data_folder+pca_3+"REC0.nc")
#Fvrec1 = xr.open_dataset(inDirName+processed_data_folder+pca_3+"REC1.nc")
#Fvrec2 = xr.open_dataset(inDirName+processed_data_folder+pca_3+"REC2.nc")

Fv = flux_clean.timeMonthly_avg_landIceFreshwaterFlux*sec_per_year/rho_fw
#Fvgen01 = Fvgen01.__xarray_dataarray_variable__*sec_per_year/rho_fw
#Fvgen1 = Fvgen1.__xarray_dataarray_variable__*sec_per_year/rho_fw
Fvgen3a = Fvgen3a.__xarray_dataarray_variable__*sec_per_year/rho_fw
#Fvgen3b = Fvgen3b.__xarray_dataarray_variable__*sec_per_year/rho_fw
#Fvgen10 = Fvgen10.__xarray_dataarray_variable__*sec_per_year/rho_fw
Fvrec0 = Fvrec0.__xarray_dataarray_variable__*sec_per_year/rho_fw
#Fvrec1 = Fvrec1.__xarray_dataarray_variable__*sec_per_year/rho_fw
#Fvrec2 = Fvrec2.__xarray_dataarray_variable__*sec_per_year/rho_fw

Fv = Fv.rename('orig')
#Fvgen01 = Fvgen01.rename('gen01')
#Fvgen1 = Fvgen1.rename('gen1')
Fvgen3a = Fvgen3a.rename('gen3a')
#Fvgen3b = Fvgen3b.rename('gen3b')
#Fvgen10 = Fvgen10.rename('gen10')
Fvrec0 = Fvrec0.rename('rec0')
#Fvrec1 = Fvrec1.rename('rec1')
#Fvrec2 = Fvrec2.rename('rec2')

def time_series(clipped_data):
    clipped_ts = clipped_data.sum(['y','x'])
    return clipped_ts

# Reconstruct flux dataset using phase randomized PCs
# This section is to be called iteratively for ensemble runs with multiple realizations
# This method also takes 'modes' as a parameter: 
# which is used to reconstruct dataset with different number of selected modes

def clip_data(total_data, basin):
    clipped_data = total_data.rio.clip(icems.loc[[basin],'geometry'].apply(mapping))
    #clipped_data = clipped_data.dropna('time',how='all')
    #clipped_data = clipped_data.dropna('y',how='all')
    #clipped_data = clipped_data.dropna('x',how='all')
    #clipped_data = clipped_data.drop("month")
    return clipped_data


print("Clipping data to specified basin...")

basin = 0
Fvclip = clip_data(Fv, basin)
#Fvgen01 = Fvgen01.rename('gen01')
#Fvgen1 = Fvgen1.rename('gen1')
Fvgen3aclip = clip_data(Fvgen3a, basin)
#Fvgen3b = Fvgen3b.rename('gen3b')
#Fvgen10 = Fvgen10.rename('gen10')
Fvrec0clip = clip_data(Fvrec0, basin)

print("Loaded all datasets...now animating...")

##===========================================================
##==============ANIMATION====================================
##===========================================================
# Melt rate spatial variability in time: Contourf animation

sns.set_theme(style="whitegrid")

oc = 'k'
pcac = '#D55E00'#'#FC8D62'
spcac = '#0072B2' #'#CC79A7'

lwbg = 0.1
lworig = 1.8
lwgen = 0.75


#axs[0, 0].set_title('Axis [0, 0]')

# Plot the initial frame.
# vmin = np.min(flux), vmax = np.max(flux) obtained manually. These should be modified to skip ocean flux values

colorbarmax = np.nanmax([Fvclip, Fvrec0clip, Fvgen3aclip])
colorbarmin = np.nanmin([Fvclip, Fvrec0clip, Fvgen3aclip])
cbarfac = 5

# Get a handle on the figure and the axes
fig, axs = plt.subplots(1,3,figsize=(12,8), subplot_kw={'projection': ccrs.SouthPolarStereo()})

#plt.suptitle('Spatial variability in basal melt rates')
plt.subplot(131)
cax1 = Fvclip[100,:,:].plot(add_colorbar=False, cmap='cmo.balance', vmax = colorbarmax/cbarfac, vmin = colorbarmin/cbarfac)#,#cbar_kwargs={'extend':'neither'})
#plt.title("Original Data: " + str(orig_ameryn.coords['time'].values[frame])[:7])
plt.title("Model Data ($F_v$)", color= oc, fontweight='bold');


plt.subplot(132)
cax2 = Fvrec0clip[100,:,:].plot(add_colorbar=False, cmap='cmo.balance', vmax = colorbarmax/cbarfac, vmin = colorbarmin/cbarfac)#,#cbar_kwargs={'extend':'neither'})
#plt.title("Phase Randomized Data: " + str(rec_ameryn.coords['time'].values[frame])[:7])
plt.title("PCA Gen. Data ($F'_v$)", color= pcac, fontweight='bold');

plt.subplot(133)
cax3 = Fvgen3aclip[100,:,:].plot(add_colorbar=False, 
                           cmap='cmo.balance', vmax = colorbarmax/cbarfac, vmin = colorbarmin/cbarfac)#,
                           #cbar_kwargs={'extend':'neither'})
plt.title("Sparse PCA Gen. Data ($F'_v$)", color= spcac, fontweight='bold');

cbar1 = fig.colorbar(cax1, location="bottom",cmap="cmo.balance")
cbar2 = fig.colorbar(cax2, location="bottom",cmap="cmo.balance")
cbar3 = fig.colorbar(cax3, location="bottom",cmap="cmo.balance")


# Next we need to create a function that updates the values for the colormesh, as well as the title.
def animate(frame):
    cax1.set_array(Fvclip[frame,:,:].values.flatten())
    cax2.set_array(Fvgen3aclip[frame,:,:].values.flatten())
    cax3.set_array(Fvrec0clip[frame,:,:].values.flatten())
    #cax4.set_ydata(orig_basin_ts[:frame]) # ANIMATED
    #cax4.set_xdata(range(frame)) # ANIMATED
    #cax4.set_marker('.')
    #cax4.set_markersize(0.25)
    #cax5.set_ydata(rec_basin_ts[:frame])
    #cax5.set_xdata(range(frame))
    #cax5.set_marker('.')
    #cax5.set_markersize(0.25)
    #cax6.set_ydata(flux_ais_basin_ts[:frame])
    #cax6.set_xdata(range(frame))
    #cax6.set_marker('.')
    #cax6.set_markersize(0.25)

print("Now creating animation video...")
# Finally, we use the animation module to create the animation.
ani2 = animation.FuncAnimation(
    fig,             # figure
    animate,         # name of the function above
    frames=100,  # Could also be iterable or list
    interval=100     # ms between frames
)

print("Now saving animation video...")
#HTML(ani2.to_jshtml())
ani2.save(inDirName+figures_folderpath+'{}_generator_comparisons.mp4'.format(basin))
print("Animation saved")
