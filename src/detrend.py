import numpy as np
import xarray as xr
import dask
import matplotlib.pyplot as plt


def detrend_dim(data, dim, deg):
    # detrend along a single dimension
    p = data.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(data[dim], p.polyfit_coefficients)
    return data - fit

inDirName = '/storage/home/hcoda1/6/smurugan9/scratch/aislens_emulation/'
DIR_external = 'data/external/'
DIR_interim = 'data/interim/'

# DATASET FILEPATHS
# Ocean model output - E3SM (SORRMv2.1.ISMF), data received from Darin Comeau / Matt Hoffman at LANL
DIR_SORRMv21 = 'data/external/SORRMv2.1.ISMF/regridded_output/'
FILE_SORRMv21 = 'Regridded_SORRMv2.1.ISMF.FULL.nc'

ds = xr.open_dataset(inDirName+DIR_SORRMv21+FILE_SORRMv21, chunks={"Time":36})
flux = ds.timeMonthly_avg_landIceFreshwaterFlux

flux_detrend = detrend_dim(flux,"Time",1)
flux_detrend = flux_detrend.compute()

flux_detrend.to_netcdf(inDirName+DIR_SORRMv21+"Regridded_SORRMv2.1.ISMF.FULL.Detrended.nc")