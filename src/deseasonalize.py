from pathlib import Path
import numpy as np
import xarray as xr
import dask
import matplotlib.pyplot as plt

# Main directory path of project repository - all filepaths are relative to this
main_dir = Path.cwd().parent
DIR_external = 'data/external/'
DIR_interim = 'data/interim/'

DIR_SORRMv21 = 'data/external/SORRMv2.1.ISMF/regridded_output/'
FILE_SORRMv21 = 'Regridded_SORRMv2.1.ISMF.FULL.nc'
FILE_SORRMv21_DETRENDED = 'SORRMv21_detrended.nc'
FILE_SORRM_CLEAN = "sorrmv21_clean.nc"


# INTERIM GENERATED FILEPATHS
DIR_basalMeltObs_Interim = 'data/interim/Paolo2023/iceShelves_dedraft/iceShelfRegions/'
DIR_SORRMv21_Interim = 'data/interim/SORRMv2.1.ISMF/iceShelves_dedraft/iceShelfRegions/'


ds = xr.open_dataset(main_dir / DIR_SORRMv21_Interim / FILE_SORRM_CLEAN)
flux = ds.__xarray_dataarray_variable__

# Deseasonalize
# Remove climatologies to isolate anomalies / deseasonalize 

if 'time' in ds.dims:
    tdim = 'time'
elif 'Time' in ds.dims:
    tdim = 'Time'

flux_month = flux.groupby("Time.month")
flux_clm = flux_month.mean("Time") # Climatologies
flux_anm = flux_month - flux_clm # Deseasonalized anomalies

flux_anm.to_netcdf(main_dir / DIR_SORRMv21_Interim / "SORRMv21_DETRENDED_DEDRAFTED_DESEASONALIZED.nc")