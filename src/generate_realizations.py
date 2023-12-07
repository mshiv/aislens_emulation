import numpy as np
import pickle
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Normalize, TwoSlopeNorm
import cmocean
import cartopy.crs as ccrs
import cartopy
import seaborn as sns
from matplotlib import rcParams, cycler
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from xeofs.xarray import EOF

import scipy
from scipy import signal
from shapely.geometry import mapping
from xarrayutils.utils import linear_trend, xr_linregress
import pandas as pd
import geopandas as gpd


main_dir = Path.cwd().parent # Main directory path of project repository - all filepaths are relative to this

# File path directories
DIR_external = 'data/external/'

# DATASET FILEPATHS
# Basal melt observations from Paolo 2023
DIR_basalMeltObs = 'data/external/Paolo2023/'
# Ocean model output - E3SM (SORRMv2.1.ISMF), data received from Darin Comeau / Matt Hoffman at LANL
DIR_SORRMv21 = 'data/external/SORRMv2.1.ISMF/regridded_output/'

# DATA FILENAMES
FILE_MeltDraftObs = 'ANT_G1920V01_IceShelfMeltDraft.nc'
FILE_SORRMv21 = 'Regridded_SORRMv2.1.ISMF.FULL.nc'
FILE_SORRMv21_DETRENDED = 'SORRMv21_detrended.nc'
FILE_iceShelvesShape = 'iceShelves.geojson'

# INTERIM GENERATED FILEPATHS
DIR_basalMeltObs_Interim = 'data/interim/Paolo2023/iceShelves_dedraft/iceShelfRegions/'
DIR_SORRMv21_Interim = 'data/interim/SORRMv2.1.ISMF/iceShelves_dedraft/iceShelfRegions/'

#modified to not return f - in calculation of RMSE, only Px required
def psd_calc_grid(data,y,x):
    f, Px = scipy.signal.welch(data[:,y,x])
    return Px

def time_series(clipped_data):
    clipped_ts = clipped_data.sum(['y','x'])
    return clipped_ts

# Reconstruct flux dataset using phase randomized PCs
# This section is to be called iteratively for ensemble runs with multiple realizations
# This method also takes 'modes' as a parameter: 
# which is used to reconstruct dataset with different number of selected modes

def generate_data(n_realization,mode,mode_skip):
    # mode can be any int in (1,nmodes), for cases 
    # when dimensionality reduction is preferred on the reconstructed dataset
    flux_reconstr = norm_model.reconstruct_randomized_X(new_fl[n_realization],slice(1,mode,mode_skip))
    #flux_reconstr = flux_reconstr.dropna('time',how='all')
    #flux_reconstr = flux_reconstr.dropna('y',how='all')
    #flux_reconstr = flux_reconstr.dropna('x',how='all')
    #flux_reconstr = flux_reconstr.drop("month")
    return flux_reconstr

def clip_data(total_data, basin):
    clipped_data = total_data.rio.clip(icems.loc[[basin],'geometry'].apply(mapping))
    #clipped_data = clipped_data.dropna('time',how='all')
    #clipped_data = clipped_data.dropna('y',how='all')
    #clipped_data = clipped_data.dropna('x',how='all')
    clipped_data = clipped_data.drop("month")
    return clipped_data

flux_clean_normalized = xr.open_dataset(main_dir / "data/interim/SORRMv2.1.ISMF/" / "flux_clean_6000_normalized.nc")
flux_clean_normalized = flux_clean_normalized.flux

flux_clean_tstd = xr.open_dataset(main_dir / "data/interim/SORRMv2.1.ISMF/" / "flux_clean_6000_tstd.nc")
flux_clean_tmean = xr.open_dataset(main_dir / "data/interim/SORRMv2.1.ISMF/" / "flux_clean_6000_tmean.nc")

file_pi = open(str(main_dir / "data/interim/SORRMv2.1.ISMF/" / "norm_model.obj"), 'rb')
norm_model = pickle.load(file_pi)

norm_eofs = xr.open_dataset(main_dir / "data/interim/SORRMv2.1.ISMF/EOF_PCA_modes/" / "sorrmv21_norm_eofs.nc" )
norm_pcs = xr.open_dataset(main_dir / "data/interim/SORRMv2.1.ISMF/EOF_PCA_modes/" / "sorrmv21_norm_pcs.nc" )
norm_varexpl = xr.open_dataset(main_dir / "data/interim/SORRMv2.1.ISMF/EOF_PCA_modes/" / "sorrmv21_norm_varexpl.nc" )

norm_eofs = norm_eofs.EOFs
norm_pcs = norm_pcs.PCs
nmodes = norm_eofs.mode.shape[0]

# Generate dataset realizations

## Standard EOF/PCA implementation
# Can use the xeofs-rand package, or directly generate using sklearn PCA.

for i in range(n_realizations):
    flux_reconstr = generate_data(i, 6000, 1)
    flux_reconstr = (flux_reconstr*flux_clean_tstd)+flux_clean_tmean
    # melt_reconstr = flux_reconstr*sec_per_year/rho_fw
    flux_reconstr = flux_reconstr.rename('flux_rec{}'.format(n_realizations))
    flux_reconstr.to_netcdf(main_dir / "data/interim/SORRMv2.1.ISMF/SORRM_6000_REC/flux_REC{}.nc".format(i))
    print('reconstructed realization # {}'.format(i))