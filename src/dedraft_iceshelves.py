# Dedraft both datasets (observations and model output)
# Refer dedraft

import sys
import os
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import rioxarray
from shapely.geometry import mapping
from xarrayutils.utils import linear_trend, xr_linregress
import gc

# inDirName = '/Users/smurugan9/research/aislens/aislens_emulation/' # LOCAL
inDirName = '/storage/home/hcoda1/6/smurugan9/scratch/aislens_emulation/' # PACE HPC
# File path directories
DIR_external = 'data/external/'

# DATASET FILEPATHS
# Basal melt observations from Paolo 2023
DIR_basalMeltObs = 'data/external/Paolo2023/'
# Ocean model output - E3SM (SORRMv2.1.ISMF), data received from Darin Comeau / Matt Hoffman at LANL
DIR_SORRMv21 = 'data/external/SORRMv2.1.ISMF/regridded_output/'

# INTERIM GENERATED FILEPATHS
DIR_basalMeltObs_Interim = 'data/interim/Paolo2023/iceShelves_dedraft/'
DIR_SORRMv21_Interim = 'data/interim/SORRMv2.1.ISMF/iceShelves_dedraft/'

# DATA FILENAMES
FILE_basalMeltObs = 'ANT_G1920V01_IceShelfMelt.nc'
FILE_MeltDraftObs = 'ANT_G1920V01_IceShelfMeltDraft.nc'
FILE_SORRMv21 = 'Regridded_SORRMv2.1.ISMF.FULL.nc'

FILE_iceShelvesShape = 'iceShelves.geojson'

MELTDRAFT_OBS = xr.open_dataset(inDirName+DIR_basalMeltObs+FILE_MeltDraftObs)
SORRMv21 = xr.open_dataset(inDirName+DIR_SORRMv21+FILE_SORRMv21)

ICESHELVES_MASK = gpd.read_file(inDirName+DIR_external+FILE_iceShelvesShape)
icems = ICESHELVES_MASK.to_crs({'init': 'epsg:3031'});
crs = ccrs.SouthPolarStereo();

# Specify projection for data/model file
MELTDRAFT_OBS.rio.write_crs("epsg:3031",inplace=True);
SORRMv21.rio.write_crs("epsg:3031",inplace=True);

# DEDRAFT 1000 yr SORRMv2.1.ISMF
SORRMv21.rio.write_crs("epsg:3031",inplace=True);
h = SORRMv21.timeMonthly_avg_ssh
h_mean = h.mean('Time')
    
for i in range(6,33):
    print('extracting data for catchment {}'.format(icems.name.values[i]))
    mlt = SORRMv21.timeMonthly_avg_landIceFreshwaterFlux.rio.clip(icems.loc[[i],'geometry'].apply(mapping),icems.crs,drop=False)
    mlt_mean = mlt.mean('Time')
    #h = ds.timeMonthly_avg_ssh
    #h_mean = h.mean('time')
    # Dedraft: Linear Regression with SSH over chosen basin
    print('calculating linear regression for catchment {}'.format(icems.name.values[i]))
    mlt_rgrs = xr_linregress(h, mlt_mean, dim='Time') # h = independent variable
    mlt_rgrs.to_netcdf(inDirName+DIR_basalMeltObs_Interim+'{}_rgrs_vals.nc'.format(icems.name.values[i]))
    mlt_prd = mlt_rgrs.slope*h_mean + mlt_rgrs.intercept
    #flx_ddrft = flx - flx_prd
    mlt_prd.to_netcdf(inDirName+DIR_basalMeltObs_Interim+'{}_rgrs.nc'.format(icems.name.values[i]))
    print('{} file saved'.format(icems.name.values[i]))
    del mlt, mlt_mean, mlt_rgrs, mlt_prd
    print('deleted interim variables')
    gc.collect()