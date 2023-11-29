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
DIR_interim = 'data/interim/'

# DATASET FILEPATHS
# Basal melt observations from Paolo 2023
DIR_basalMeltObs = 'data/external/Paolo2023/'
# Ocean model output - E3SM (SORRMv2.1.ISMF), data received from Darin Comeau / Matt Hoffman at LANL
DIR_SORRMv21 = 'data/external/SORRMv2.1.ISMF/regridded_output/'

DIR_basalMeltObs_Interim = 'data/interim/Paolo2023/iceShelves_dedraft/iceShelfRegions/'
DIR_SORRMv21_Interim = 'data/interim/SORRMv2.1.ISMF/iceShelves_dedraft/iceShelfRegions/'

# DATA FILENAMES
FILE_MeltDraftObs = 'ANT_G1920V01_IceShelfMeltDraft.nc'
FILE_SORRMv21 = 'Regridded_SORRMv2.1.ISMF.FULL.nc'
FILE_SORRMv21_DETRENDED = 'SORRMv21_detrended.nc'
FILE_iceShelvesShape = 'iceShelves.geojson'

MELTDRAFT_OBS = xr.open_dataset(main_dir / DIR_basalMeltObs / FILE_MeltDraftObs)
# SORRMv21 = xr.open_dataset(main_dir / DIR_SORRMv21 / FILE_SORRMv21)

data = MELTDRAFT_OBS
ds = data.melt
# ds = data.timeMonthly_avg_landIceFreshwaterFlux

np_flux_array = np.empty(ds[0].shape)
np_flux_array[:] = np.nan

iceshelves_rgrs_array = xr.DataArray(np_flux_array, coords=ds[0].coords, dims = ds[0].dims, attrs=ds.attrs)
#iceshelves_rgrs = xr.Dataset(data_vars=dict(timeMonthly_avg_landIceFreshwaterFlux=(iceshelves_rgrs_array)), coords=data.coords, attrs=data.timeMonthly_avg_landIceFreshwaterFlux.attrs)
iceshelves_rgrs = xr.Dataset(data_vars=dict(melt=(iceshelves_rgrs_array)))

IMBIEregions = range(6,33)
iceShelfRegions = range(33,133)

for i in iceShelfRegions:
    iceshelves_rgrs_catchment = xr.open_dataset(main_dir / DIR_basalMeltObs_Interim /'{}_rgrs.nc'.format(icems.name.values[i]))
    iceshelves_rgrs_catchment['melt'] = iceshelves_rgrs_catchment['__xarray_dataarray_variable__']
    iceshelves_rgrs_catchment = iceshelves_rgrs_catchment.drop(['__xarray_dataarray_variable__'])
    iceshelves_rgrs = xr.merge([iceshelves_rgrs, iceshelves_rgrs_catchment], compat='no_conflicts')

iceshelves_rgrs.to_netcdf(main_dir / DIR_basalMeltObs_Interim / "iceshelves_draft_dependence_parameters_total.nc")