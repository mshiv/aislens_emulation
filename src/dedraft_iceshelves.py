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


inDirName = '/Users/smurugan9/research/aislens/aislens_emulation/'
DIR_basalMeltObs = 'data/external/Paolo2023/'
DIR_external = 'data/external/'

DIR_basalMeltObs_Processed = 'data/interim/Paolo2023/iceShelves_dedraft/'
# PACE PATH
# inDirName = '/storage/home/hcoda1/6/smurugan9/data/basalmelt_obs_dedraft/'

FILE_basalMeltObs ='ANT_G1920V01_IceShelfMelt.nc'
# bedmachinefilename = 'BedMachineAntarctica-v3.nc'
FILE_iceShelvesShape = 'iceShelves.geojson'

melt = xr.open_dataset(inDirName+DIR_basalMeltObs+FILE_basalMeltObs)
# bedmachine = xr.open_dataset(inDirName+bedmachinefilename)
iceshelvesmask = gpd.read_file(inDirName+DIR_external+FILE_iceShelvesShape)
icems = iceshelvesmask.to_crs({'init': 'epsg:3031'});
crs = ccrs.SouthPolarStereo();

# Specify projection for data file
melt.rio.write_crs("epsg:3031",inplace=True);

h = melt.smb - melt.thickness
h_mean = h.mean('time')

for i in range(6,32):
    print('extracting data for catchment {}'.format(icems.name.values[i]))
    mlt = melt.melt.rio.clip(icems.loc[[i],'geometry'].apply(mapping),icems.crs,drop=False)
    mlt_mean = mlt.mean('time')
    #h = ds.timeMonthly_avg_ssh
    #h_mean = h.mean('time')
    # Dedraft: Linear Regression with SSH over chosen basin
    print('calculating linear regression for catchment {}'.format(icems.name.values[i]))
    mlt_rgrs = xr_linregress(h, mlt_mean, dim='time') # h = independent variable
    mlt_prd = mlt_rgrs.slope*h_mean + mlt_rgrs.intercept
    #flx_ddrft = flx - flx_prd
    mlt_prd.to_netcdf(inDirName+'dedraft_rgrs_iceshelves/{}_rgrs.nc'.format(icems.name.values[i]))
    print('{} file saved'.format(icems.name.values[i]))
    del ds, mlt, mlt_mean, mlt_rgrs, mlt_prd
    print('deleted interim variables')
    gc.collect()