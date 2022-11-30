import xarray
import sys
import glob, os
from parse import parse
import datetime as dt

##########################################################################################################################################
##### DEFINE DIRECTORY AND FILE PATHS  ###################################################################################################
##########################################################################################################################################


# Directory containing E3SM_lowres_newrun .nc files
# Files available on sermia. Copy these data files into this directory
inDirName = '/storage/home/hcoda1/6/smurugan9/scratch/E3SM_lowres_newrun/'
#inDirName = '/Users/smurugan9/research/aislens/aislens_emulation/data/raw/'

# Directory to save output files. 
# These files will contain the landIceFreshwaterFlux and SSH data variables from the original dataset without any regridding, i.e., it will remain an unstructured mesh output.
outDirName = '/storage/home/hcoda1/6/smurugan9/data/E3SM_lowres_newrun_fluxSSH/'
#outDirName = '/Users/smurugan9/research/aislens/aislens_emulation/data/processed/E3SM_lowres_newrun_FluxSSH/'

##########################################################################################################################################

for file in sorted(glob.glob(inDirName+"mpaso.hist.am.timeSeriesStatsMonthly.*.nc")):
   (path, inFileName) = os.path.split(file)
   print(inFileName)
   # Read only flux and ssh
   flux = xarray.open_dataset(inDirName + inFileName).timeMonthly_avg_landIceFreshwaterFlux
   ssh = xarray.open_dataset(inDirName + inFileName).timeMonthly_avg_ssh
   Dataset = xarray.merge([flux, ssh],combine_attrs='drop_conflicts')
   Dataset.attrs['history'] = 'Extracted variables from {} using {}'.format(inFileName, sys.argv)  # could add more provenenace info to history attribute
   print("Writing data file with extracted variables.")
   #outFileName = '{}'.format(inFileName)  # replace this whatever you want the regridded file to be called
   pattern = 'mpaso.hist.am.timeSeriesStatsMonthly.{timestamp}.nc'
   t = parse(pattern, inFileName)
   tstr=t['timestamp']
   dtobj = dt.datetime.strptime(tstr,'%Y-%m-%d')
   print(dtobj)
   timestampedDataset = Dataset.assign_coords(time=('Time', [tstr]))
   finalOutFileName = 'mpaso.FluxSSH.{}.nc'.format(tstr)
   timestampedDataset.to_netcdf(outDirName+finalOutFileName)
   print("Created file {} ...".format(finalOutFileName))
print("Writing output complete.")