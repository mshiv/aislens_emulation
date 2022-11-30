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

# Define project repo path
inDirName = '/Users/smurugan9/research/aislens/aislens_emulation/'

# Data file paths
regriddedFluxSSH_filepath = 'data/interim/RegriddedFluxSSH.nc' # Data containing regridded flux and SSH for 150 years
iceShelvesShape_filepath = 'data/interim/iceShelves.geojson' # File contains all defined ice shelves
#figures_folderpath = 'reports/figures/' # Folder for output figures

interim_data_folder = 'data/interim/'
#flux_dedrafted_data_path = 'dedrafted_flux_IMBIE/'

flux_dedrafted_iceshelves_data_path = 'iceshelves_dedrafted_flux/'

data = xr.open_dataset(inDirName + regriddedFluxSSH_filepath)
#flux = data.timeMonthly_avg_landIceFreshwaterFlux
#ssh = data.timeMonthly_avg_ssh
#lat = data.lat
#lon = data.lon

# Read geoJSON region feature file as GeoDataFrame
iceshelvesmask = gpd.read_file(inDirName + iceShelvesShape_filepath)
# Convert to south polar stereographic projection
icems = iceshelvesmask.to_crs({'init': 'epsg:3031'});
crs = ccrs.SouthPolarStereo();
# Specify projection for data file
data.rio.write_crs("epsg:3031",inplace=True);
# Specify projection for data file flux array
# flux.rio.write_crs("epsg:3031",inplace=True);

h = data.timeMonthly_avg_ssh
h_mean = h.mean('time')

for i in range(48,133):
    print('extracting data for catchment {}'.format(icems.name.values[i]))
    ds = data.rio.clip(icems.loc[[i],'geometry'].apply(mapping),icems.crs,drop=False)
    flx = ds.timeMonthly_avg_landIceFreshwaterFlux
    flx_mean = flx.mean('time')
    #h = ds.timeMonthly_avg_ssh
    #h_mean = h.mean('time')
    # Dedraft: Linear Regression with SSH over chosen basin
    print('calculating linear regression for catchment {}'.format(icems.name.values[i]))
    flx_rgrs = xr_linregress(h, flx_mean, dim='time') # h = independent variable
    flx_prd = flx_rgrs.slope*h_mean + flx_rgrs.intercept
    #flx_ddrft = flx - flx_prd
    flx_prd.to_netcdf(inDirName+interim_data_folder+flux_dedrafted_iceshelves_data_path+'{}_rgrs.nc'.format(icems.name.values[i]))
    print('{} file saved'.format(icems.name.values[i]))
    del ds, flx, flx_mean, flx_rgrs, flx_prd
    print('deleted interim variables')
    gc.collect()