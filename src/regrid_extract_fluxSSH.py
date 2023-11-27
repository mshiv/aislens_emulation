#!/usr/bin/env python
# This software is open source software available under the BSD-3 license.
#
# Copyright (c) 2018 Los Alamos National Security, LLC. All rights reserved.
# Copyright (c) 2018 Lawrence Livermore National Security, LLC. All rights
# reserved.
# Copyright (c) 2018 UT-Battelle, LLC. All rights reserved.
#
# Additional copyright and license information can be found in the LICENSE file
# distributed with this code, or at
# https://raw.githubusercontent.com/MPAS-Dev/MPAS-Analysis/master/LICENSE

'''
Creates a mapping file that can be used with ncremap (NCO) to remap MPAS files
to a grid on the BEDMAP2 projection.

Usage: Copy this script into the main MPAS-Analysis directory.
Modify file names and other values as suggested in the comments.

Requires MPAS-Analysis: https://github.com/MPAS-Dev/MPAS-Analysis

Best way to get it is through a conda environment:
conda create -n mpas_analysis -c conda-forge python=3.7 mpas-analysis

A generic MPAS-Analysis config file is also required.  Either download from Github,
or find it in the conda env like:
import pkg_resources
...
defaultConfig = pkg_resources.resource_filename('mpas_analysis','config.default')
config = MpasAnalysisConfigParser()
config.read(defaultConfig)
'''

# from pyremap import MpasMeshDescriptor, Remapper, get_polar_descriptor
import pyremap
import xarray
import sys
import glob, os
from parse import parse
import datetime as dt

# The first time through, create a mapping file

# the MPAS mesh name you want in the name of the mapping file
inGridName = 'SOwISC12to60E2r4'

# replace with the path to the desired mesh or output file to be used to create the mapping file
# (can be either the initial condition file or any output file - any file on the same mesh)
# inGridFileName = '/storage/home/hcoda1/6/smurugan9/data/SORRMv2.1.ISMF/mpaso.SOwISC12to60E2r4.rstFromG-anvil.210203.nc'

inGridFileName = '/Users/smurugan9/research/aislens/aislens_emulation/data/raw/SORRMv2.1.ISMF/mpaso.SOwISC12to60E2r4.rstFromG-anvil.210203.nc'

inDescriptor = pyremap.MpasMeshDescriptor(inGridFileName, inGridName)

# replace these numbers with the desired size and resolution of the output mesh
# Lx and Ly are mesh height and width in km, centered on South Pole
# dx and dy are the mesh spacing in km
outDescriptor = pyremap.get_polar_descriptor(Lx=6000., Ly=6000., dx=10., dy=10.,
                                     projection='antarctic')
outGridName = outDescriptor.meshName

mappingFileName = 'map_{}_to_{}.nc'.format(inGridName, outGridName) # definition of the name of the mapping file

print("Creating mapping file")
remapper = pyremap.Remapper(inDescriptor, outDescriptor, mappingFileName)

remapper.build_mapping_file(method='bilinear') # conservative remapping also an option, but bilinear is faster and I think will be adequate for your needs

print("Mapping file created.")

#exit()

print("Performing regridding.")
# Note: could loop over multiple files here
#inFileName = './oEC60to30v3wLI60lev.171031.nc'  # replace this with the output file you want to regrid

# inDirName = '/storage/home/hcoda1/6/smurugan9/data/SORRMv2.1.ISMF/'
inDirName = '/Users/smurugan9/research/aislens/aislens_emulation/data/raw/SORRMv2.1.ISMF/'
#os.chdir(inDirName)
for file in sorted(glob.glob(inDirName+"SORRMv2.1.ISMF.*.nc")):
   (path, inFileName) = os.path.split(file)
   print(inFileName)
   outFileName = inFileName[0:-2] + outGridName + ".nc"
   print(outFileName)
   #inFileName = inDirName + 'mpaso.hist.am.timeSeriesStatsMonthly.0151-09-01.nc'
   # inFileName = file
   # Read only flux
   flux = xarray.open_dataset(inDirName + inFileName).timeMonthly_avg_landIceFreshwaterFlux
   ssh = xarray.open_dataset(inDirName + inFileName).timeMonthly_avg_ssh
   #ds = xarray.open_dataset(inDirName + inFileName).timeMonthly_avg_landIceFreshwaterFlux
   #ds = ds.drop('weightsOnEdge') # an example of a way to drop a variable from the output, to keep file sizes smaller (this field is a large but unneeded mesh field)
   remappedFlux = xarray.Dataset()
   remappedSSH = xarray.Dataset()
   remappedFlux = remapper.remap(flux, renormalizationThreshold=0.01)  # perform remapping.  This threshold will apply data to any destination cell that has at least 1% overlap with the source grid.  Change this as desired.
   remappedSSH = remapper.remap(ssh, renormalizationThreshold=0.01)
   remappedDataset = xarray.Dataset()
   remappedDataset = xarray.merge([remappedFlux, remappedSSH],combine_attrs='drop_conflicts')
   remappedDataset.attrs['history'] = 'remapped from {} using {}'.format(inGridFileName, sys.argv)  # could add more provenenace info to history attribute
   print("Regridding complete.")
   print("Writing remapped data file with extracted variable.")
   finalOutFileName = 'RegriddedFluxSSH.nc'
   remappedDataset.close()
   remappedDataset.to_netcdf(finalOutFileName)
   print("Created file {} ...".format(finalOutFileName))
   #os.remove(inFileName)
print("Writing output complete.")

# TODO - Concatenate DataArrays along "Time" dimension.
# Possible Solutions:
# xarray.open_mfdataset("infilename", concat_dims="Time")

# How to delete a file : use os.remove

#import os
#if os.path.exists("demofile.txt"):
#  os.remove("demofile.txt")
#else:
#  print("The file does not exist")


