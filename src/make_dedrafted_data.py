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

MELTDRAFT_OBS = xr.open_dataset(main_dir / DIR_basalMeltObs / FILE_MeltDraftObs)
# Note that MELTDRAFT_OBS has not been DETRENDED. TODO?
SORRMv21 = xr.open_dataset(main_dir / DIR_SORRMv21 / FILE_SORRMv21, chunks={'Time':1200})
SORRMv21_DETRENDED = xr.open_dataset(main_dir / DIR_SORRMv21 / FILE_SORRMv21_DETRENDED, chunks={'Time':1200})

ICESHELVES_MASK = gpd.read_file(main_dir / DIR_external / FILE_iceShelvesShape)
icems = ICESHELVES_MASK.to_crs({'init': 'epsg:3031'});
crs = ccrs.SouthPolarStereo();

obs23_draft_param = xr.open_dataset(main_dir / DIR_basalMeltObs_Interim / "iceshelves_draft_dependence_parameters_total.nc")
sorrmv21_draft_param = xr.open_dataset(main_dir / DIR_SORRMv21_Interim / "iceshelves_draft_dependence_parameters_total.nc")


obs23_clean = MELTDRAFT_OBS.melt - obs23_draft_param.timeMonthly_avg_landIceFreshwaterFlux
sorrmv21_clean = SORRMv21_DETRENDED.timeMonthly_avg_landIceFreshwaterFlux - sorrmv21_draft_param.timeMonthly_avg_landIceFreshwaterFlux

obs23_clean.to_netcdf(main_dir / DIR_basalMeltObs_Interim / "obs23_clean.nc")
sorrmv21_clean.to_netcdf(main_dir / DIR_SORRMv21_Interim / "sorrmv21_clean.nc")