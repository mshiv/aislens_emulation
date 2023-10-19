import xarray as xr
import sys
import glob, os
import re

inDirName = '/Users/smurugan9/research/aislens/aislens_emulation/'
sorrm_regridded_path = 'data/external/SORRMv2.1.ISMF/regridded_output/'

datasets = []

for file in sorted(glob.glob(inDirName+sorrm_regridded_path+"Regridded_SORRMv2.1.ISMF.*.6000.0x6000.0km_10.0km_Antarctic_stereo.nc")):
    (path, inFileName) = os.path.split(file)
    print(inFileName)
    #d.append(inFileName)
    ds = xr.open_dataset(inDirName+sorrm_regridded_path+inFileName)
    datasets.append(ds)

combined_ds = xr.concat(datasets,dim='Time')
combined_ds.to_netcdf(inDirName+sorrm_regridded_path+'Regridded_SORRMv2.1.ISMF.FULL.nc')
print("time coord assigned.")