The Jupyter notebook generator_workflow.ipynb includes code for the generation method and additional documentation on dependencies.

Supplementary data files
catchments_scatter.npy: Time averaged freshwater flux for individual basins
flux_clean: netcdf file with detrended, deseasonalized, dedrafted flux data (3 dimensional)
iceShelves.geojson : catchment boundary definitions
iceshelves_dedrafted_total.nc: dedrafted flux data
ocean.ECwISC30to60E1r02.200408.nc: file to regrid from MPAS-Ocean variable resolution to regular resolution spatiotemporal dataset

Python scripts
regrid_extract_FluxSSH.py: Script to regrid monthly input variable resolution MPAS-Ocean datafiles (mpaso.hist.am.timeSeriesStatsMonthly.*.nc) into specified constant resolution grid (stereographic, of resolution specified in the script)
make_mpaso_dataset.py: Script to create the RegriddedFluxSSH.nc file, with input monthly input MPAS-Ocean datafiles (mpaso.hist.am.timeSeriesStatsMonthly.*.nc)
data_concat.py: Script to concatenate monthly data files into a time series.
make_iceshelves_draft_parameterization.py: Script for linear regression to remove melt-draft dependence.

Contact at smurugan9@gatech.edu