from pathlib import Path
import numpy as np
import xarray as xr
import dask
import matplotlib.pyplot as plt


# TODO : Dask implementation of the detrend function
# Refer: https://ncar.github.io/esds/posts/2022/dask-debug-detrend/

def detrend_dim(data, dim, deg):
    # detrend along a single dimension
    p = data.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(data[dim], p.polyfit_coefficients)
    return data - fit

# Main directory path of project repository - all filepaths are relative to this
main_dir = Path.cwd().parent
DIR_external = 'data/external/'
DIR_interim = 'data/interim/'

# DATASET FILEPATHS
# Ocean model output - E3SM (SORRMv2.1.ISMF), data received from Darin Comeau / Matt Hoffman at LANL
DIR_SORRMv21 = 'data/external/SORRMv2.1.ISMF/regridded_output/'
FILE_SORRMv21 = 'Regridded_SORRMv2.1.ISMF.FULL.nc'

# ds = xr.open_dataset(main_dir / DIR_SORRMv21 / FILE_SORRMv21, chunks={"Time":36})
ds = xr.open_dataset(main_dir / DIR_SORRMv21 / FILE_SORRMv21)
flux = ds.timeMonthly_avg_landIceFreshwaterFlux

flux_detrend = detrend_dim(flux,"Time",1)
flux_detrend = flux_detrend.compute()

flux_detrend.to_netcdf(main_dir / DIR_SORRMv21 / "SORRMv21_detrended.nc")