import xarray
import sys
import glob, os
# from parse import parse

# Directory containing E3SM_lowres_newrun .nc files
# Files available on sermia. Copy these data files into this directory
inDirName = '/storage/home/hcoda1/6/smurugan9/scratch/ismip6-0721/landice/ismip6_run/ismip6_ais_proj2300/ctrlAE_08/output/'
# Directory to save output files. 
# These files will contain the landIceFreshwaterFlux and SSH data variables from the original dataset without any regridding, i.e., it will remain an unstructured mesh output.
outDirName = '/storage/home/hcoda1/6/smurugan9/scratch/ismip6-0721/landice/ismip6_run/ismip6_ais_proj2300/ctrlAE_08/output/'

datasets = []

for file in sorted(glob.glob(inDirName+"output_flux_all_timesteps_*.nc")):
    (path, inFileName) = os.path.split(file)
    print(inFileName)
    ds = xr.open_dataset(inDirName+inFileName)
    datasets.append(ds)
combined_ds = xr.concat(datasets,dim='Time')

combined_ds.to_netcdf(outDirName+'output_flux_all_timesteps_combined.nc')