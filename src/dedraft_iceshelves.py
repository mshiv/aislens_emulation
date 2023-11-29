# Dedraft both datasets (observations and model output)
# Refer dedraft
# TODO : Refactor to take input/output filepaths and type of regions as cli arguments
# TODO : Add Dask chunking for i/o when large files

import sys
import os
from pathlib import Path
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

# MELTDRAFT_OBS = xr.open_dataset(main_dir / DIR_basalMeltObs / FILE_MeltDraftObs)
SORRMv21 = xr.open_dataset(main_dir / DIR_SORRMv21 / FILE_SORRMv21, chunks={'Time':1200})
SORRMv21_DETRENDED = xr.open_dataset(main_dir / DIR_SORRMv21 / FILE_SORRMv21_DETRENDED, chunks={'Time':1200})

ICESHELVES_MASK = gpd.read_file(main_dir / DIR_external / FILE_iceShelvesShape)
icems = ICESHELVES_MASK.to_crs({'init': 'epsg:3031'});
crs = ccrs.SouthPolarStereo();

# Specify projection for data/model file
# MELTDRAFT_OBS.rio.write_crs("epsg:3031",inplace=True);
SORRMv21.rio.write_crs("epsg:3031",inplace=True);
SORRMv21_DETRENDED.rio.write_crs("epsg:3031",inplace=True);

# DEDRAFT 1000 yr SORRMv2.1.ISMF
# SORRMv21.rio.write_crs("epsg:3031",inplace=True);
# h = MELTDRAFT_OBS.draft
h = SORRMv21.timeMonthly_avg_ssh

if 'time' in h.dims:
    tdim = 'time'
elif 'Time' in h.dims:
    tdim = 'Time'

h_mean = h.mean(tdim)

IMBIEregions = range(6,33)
iceShelfRegions = range(33,133)

for i in iceShelfRegions:
    print('extracting data for catchment {}'.format(icems.name.values[i]))
    mlt = SORRMv21_DETRENDED.__xarray_dataarray_variable__.rio.clip(icems.loc[[i],'geometry'].apply(mapping),icems.crs,drop=False)
    # mlt = MELTDRAFT_OBS.melt.rio.clip(icems.loc[[i],'geometry'].apply(mapping),icems.crs,drop=False)
    mlt_mean = mlt.mean(tdim)
    # Dedraft: Linear Regression with SSH over chosen basin
    print('calculating linear regression for catchment {}'.format(icems.name.values[i]))
    mlt_rgrs = xr_linregress(h, mlt_mean, dim=tdim) # h = independent variable
    mlt_rgrs.to_netcdf(main_dir / DIR_SORRMv21_Interim / '{}_rgrs_vals.nc'.format(icems.name.values[i]))
    mlt_prd = mlt_rgrs.slope*h_mean + mlt_rgrs.intercept
    # flx_ddrft = flx - flx_prd
    mlt_prd.to_netcdf(main_dir / DIR_SORRMv21_Interim / '{}_rgrs.nc'.format(icems.name.values[i]))
    print('{} file saved'.format(icems.name.values[i]))
    del mlt, mlt_mean, mlt_rgrs, mlt_prd
    print('deleted interim variables')
    gc.collect()