from pathlib import Path
import numpy as np
import xarray as xr
import dask
import matplotlib.pyplot as plt


# Main directory path of project repository - all filepaths are relative to this
main_dir = Path.cwd().parent
DIR_external = 'data/external/'
DIR_interim = 'data/interim/'

# DATASET FILEPATHS
# Ocean model output - E3SM (SORRMv2.1.ISMF), data received from Darin Comeau / Matt Hoffman at LANL
DIR_SORRMv21 = 'data/external/SORRMv2.1.ISMF/regridded_output/'
FILE_SORRMv21 = 'Regridded_SORRMv2.1.ISMF.FULL.nc'
FILE_SORRMv21_DETRENDED = 'SORRMv21_detrended.nc'

ds = xr.open_dataset(main_dir / DIR_SORRMv21 / FILE_SORRMv21_DETRENDED)
flux = ds.__xarray_data_variable__

flux_ts = flux.sum(["x","y"])

flux_ts.to_netcdf(main_dir / DIR_SORRMv21 / "SORRMv21_detrended_ts.nc")