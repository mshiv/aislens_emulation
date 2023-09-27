#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 9, 2022

@author: Trevor Hillebrand, Matthew Hoffman

Adapted by Shivaprakash Muruganandham
Sep 1, 2023

Script to plot snapshot maps of MALI output for an arbitrary number of files,
variables, and output times. There is no requirement for all output files
to be on the same mesh. Each output file gets its own figure, each
variable gets its own row, and each time gets its own column. Three contours
are automatically plotted, showing intial ice extent (black), initial
grounding-line position (grey), and grounding-line position at the desired
time (white).

"""

import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.gridspec as gridspec
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Normalize, TwoSlopeNorm

timeLevs = 1
sec_per_year = 60. * 60. * 24. * 365.
rhoi = 910.
rhosw = 1028.

# Set bitmask values
initialExtentValue = 1
dynamicValue = 2
floatValue = 4
groundingLineValue = 256

def dist(i1, i2, xCell, yCell):  # helper distance fn
    dist = ((xCell[i1]-xCell[i2])**2 + (yCell[i1]-yCell[i2])**2)**0.5
    return dist

# Loop over runs
# Each run gets its own figure
# Each variable gets its own row
# Each time level gets its own column
varPlot = {}
figs = {}
gs = {}
for ii, run in enumerate(runs):
    if '.nc' not in run:
        run = run + '/output_state_2010t.nc'
    f = Dataset(run, 'r')
    if 'daysSinceStart' in f.variables.keys():
        yr = f.variables['daysSinceStart'][:] / 365.
    else:
        yr = [0.]

    f.set_auto_mask(False)

    # Get mesh geometry and calculate triangulation. 
    # It would be more efficient to do this outside
    # this loop if all runs are on the same mesh, but we
    # want this to be as general as possible.
    if args.mesh is not None:
       m = Dataset(mesh[ii], 'r')
    else:
       m = f  # use run file for mesh variables

    xCell = m.variables["xCell"][:]
    yCell = m.variables["yCell"][:]
    dcEdge = m.variables["dcEdge"][:]

    if args.mesh is not None:
       m.close()

    triang = tri.Triangulation(xCell, yCell)
    triMask = np.zeros(len(triang.triangles))
    # Maximum distance in m of edges between points.
    # Make twice dcEdge to be safe
    maxDist = np.max(dcEdge) * 2.0
    for t in range(len(triang.triangles)):
        thisTri = triang.triangles[t, :]
        if dist(thisTri[0], thisTri[1], xCell, yCell) > maxDist:
            triMask[t] = True
        if dist(thisTri[1], thisTri[2], xCell, yCell) > maxDist:
            triMask[t] = True
        if dist(thisTri[0], thisTri[2], xCell, yCell) > maxDist:
            triMask[t] = True
    triang.set_mask(triMask)

    # set up figure for this run
    figs[run] = plt.figure()
    figs[run].suptitle(run)
    nRows = len(variables)
    nCols = len(timeLevs) + 1

    # last column is for colorbars
    gs[run] = gridspec.GridSpec(nRows, nCols,
                           height_ratios=[1] * nRows,
                           width_ratios=[1] * (nCols - 1) + [0.1])
    axs = []
    cbar_axs = []
    for row in np.arange(0, nRows):
        cbar_axs.append(plt.subplot(gs[run][row,-1]))
        for col in np.arange(0, nCols-1):
            if axs == []:
                axs.append(plt.subplot(gs[run][row, col]))
            else:
                axs.append(plt.subplot(gs[run][row, col], sharex=axs[0], sharey=axs[0]))

    varPlot[run] = {}  # is a dict of dicts too complicated?
    cbars = []
    # Loop over variables
    for row, (variable, log, colormap, cbar_ax, vmin, vmax) in enumerate(
        zip(variables, log_plot, colormaps, cbar_axs, vmins, vmaxs)):
        if variable == 'observedSpeed':
            var_to_plot = np.sqrt(f.variables['observedSurfaceVelocityX'][:]**2 +
                                  f.variables['observedSurfaceVelocityY'][:]**2)
        else:
            var_to_plot = f.variables[variable][:]

        if len(np.shape(var_to_plot)) == 1:
           var_to_plot = var_to_plot.reshape((1, np.shape(var_to_plot)[0]))

        if 'Speed' in variable:
            units = 'm yr^{-1}'
            var_to_plot *= sec_per_year
        else:
            try:
                units = f.variables[variable].units
            except AttributeError:
                units='no-units'

        if log == 'True':
            var_to_plot = np.log10(var_to_plot)
            # Get rid of +/- inf values that ruin vmin and vmax
            # calculations below.
            var_to_plot[np.isinf(var_to_plot)] = np.nan
            colorbar_label_prefix = 'log10 '
        else:
            colorbar_label_prefix = ''

        varPlot[run][variable] = []

        # Set lower and upper bounds for plotting
        if vmin in ['None', None]:
            # 0.1 m/yr is a pretty good lower bound for speed
            first_quant = np.nanquantile(var_to_plot[timeLevs, :], 0.01)
            if 'Speed' in variable and log == 'True':
                vmin = max(first_quant, -1.)
            else:
                vmin = first_quant
        if vmax in ['None', None]:
            vmax = np.nanquantile(var_to_plot[timeLevs, :], 0.99)
        # Plot bedTopography on an asymmetric colorbar if appropriate
        if ( (variable == 'bedTopography') and
             (np.nanquantile(var_to_plot[timeLevs, :], 0.99) > 0.) and
             (colormap in divColorMaps) ):
            norm = TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=0.)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)

        if 'cellMask' in f.variables.keys():
            calc_mask = True
            cellMask = f.variables["cellMask"][:]
            floatMask = (cellMask & floatValue) // floatValue
            dynamicMask = (cellMask & dynamicValue) // dynamicValue
            groundingLineMask = (cellMask & groundingLineValue) // groundingLineValue
            initialExtentMask = (cellMask & initialExtentValue) // initialExtentValue
        elif ( 'cellMask' not in f.variables.keys() and
             'thickness' in f.variables.keys() and 
             'bedTopography' in f.variables.keys() ):
            print(f'cellMask is not present in output file {run}; calculating masks from ice thickness')
            calc_mask = True
            groundedMask = (f.variables['thickness'][:] > (-rhosw / rhoi * f.variables['bedTopography'][:]))
            groundingLineMask = groundedMask.copy()  # This isn't technically correct, but works for plotting
            initialExtentMask = (f.variables['thickness'][:] > 0.)
        else:
            print(f'cellMask and thickness and/or bedTopography not present in output file {run};'
                   ' Skipping mask calculation.')
            calc_mask = False

        # Loop over time levels
        for col, timeLev in enumerate(timeLevs):
            index = row * (nCols - 1) + col
            # plot initial grounding line position, initial extent, and GL position at t=timeLev
            if calc_mask:
                axs[index].tricontour(triang, groundingLineMask[0, :],
                                      levels=[0.9999], colors='grey',
                                      linestyles='solid')
                axs[index].tricontour(triang, groundingLineMask[timeLev, :],
                                      levels=[0.9999], colors='white',
                                      linestyles='solid')
                axs[index].tricontour(triang, initialExtentMask[timeLev, :],
                                      levels=[0.9999], colors='black',
                                      linestyles='solid')

            # Plot 2D field at each desired time. Use quantile range of 0.01-0.99 to cut out
            # outliers. Could improve on this by accounting for areaCell, as currently all cells
            # are weighted equally in determining vmin and vmax.
            varPlot[run][variable].append(
                              axs[index].tripcolor(
                                  triang, var_to_plot[timeLev, :], cmap=colormap,
                                  shading='flat', norm=norm))
            axs[index].set_aspect('equal')
            axs[index].set_title(f'year = {yr[timeLev]:0.2f}')

        cbars.append(Colorbar(ax=cbar_ax, mappable=varPlot[run][variable][0], orientation='vertical',
                 label=f'{colorbar_label_prefix}{variable} (${units}$)'))

    figs[run].tight_layout()
    if args.saveNames is not None:
        figs[run].savefig(saveNames[ii], dpi=400, bbox_inches='tight')
    
    f.close()