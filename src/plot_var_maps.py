import numpy as np
from netCDF4 import Dataset
import argparse
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.gridspec as gridspec
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Normalize, TwoSlopeNorm
import glob
import os

print("** Gathering information.  (Invoke with --help for more details. All arguments are optional)")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("-r", dest="runs", default=None, metavar="FILENAME",
                    help="path to .nc file or dir containing output.nc \
                    file (strings separated by commas; no spaces)")
parser.add_argument("-t", dest="timeLevels", default="-1",
                    help="integer time levels at which to plot \
                          (int separated by commas; no spaces)")
parser.add_argument("-v", dest="variables", default='thickness',
                    help="variable(s) to plot (list separated by commas; no spaces)")
parser.add_argument("-c", dest="colormaps", default=None,
                    help="colormaps to use for plotting (list separated by commas \
                          , no spaces). This overrides default colormaps.")

args = parser.parse_args()
runs = str(args.runs) #.split(',') # split run directories into list
variables = args.variables.split(',')
timeLevs = args.timeLevels.split(',')  # split time levels into list
# convert timeLevs to list of ints
timeLevs = [int(i) for i in timeLevs]
sec_per_year = 60. * 60. * 24. * 365.
rhoi = 910.
rhosw = 1028.

# Set up a dictionary of default colormaps for common variables.
# These can be overridden by the -c flag.
defaultColors = {'thickness' : 'Blues',
                 'surfaceSpeed' : 'plasma',
                 'basalSpeed' : 'plasma',
                 'bedTopography' : 'BrBG',
                 'floatingBasalMassBalApplied' : 'cividis'
                }

if args.colormaps is not None:
    colormaps = args.colormaps.split(',')
else:
    colormaps = []
    for variable in variables:
        if variable in defaultColors.keys():
            colormaps.append(defaultColors[variable])
        else:
            # All other variables default to viridis
            colormaps.append('viridis')

# Set bitmask values
initialExtentValue = 1
dynamicValue = 2
floatValue = 4
groundingLineValue = 256

# List of diverging colormaps for use in plotting bedTopography.
# I don't see a way around hard-coding this.
divColorMaps = ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
                      'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']

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

for ii, run in enumerate(glob.glob(runs+"output_flux_all_timesteps_*.nc")):
    (path, inFileName) = os.path.split(run)
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
    m = f  # use run file for mesh variables

    xCell = m.variables["xCell"][:]
    yCell = m.variables["yCell"][:]
    dcEdge = m.variables["dcEdge"][:]

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
    for row, (variable, colormap, cbar_ax) in enumerate(
        zip(variables, colormaps, cbar_axs)):
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

        colorbar_label_prefix = ''

        varPlot[run][variable] = []

        # Set lower and upper bounds for plotting
        # 0.1 m/yr is a pretty good lower bound for speed
        first_quant = np.nanquantile(var_to_plot[timeLevs, :], 0.01)
        if 'Speed' in variable and log == 'True':
            vmin = max(first_quant, -1.)
        else:
            vmin = first_quant
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
    figs[run].savefig(f'{runs}{variables[0]}_{ii+2000}.png', dpi=400, bbox_inches='tight')
    print("saved fig for {}".format(ii+2000))
    f.close()

plt.show()