import numpy as np
import xarray as xr
from xeofs.xarray import EOF
import scipy
from sklearn.metrics import mean_squared_error

############################## HELPER FUNCTIONS ####################################
# Reconstruct flux dataset using phase randomized PCs.
# This section is to be called iteratively for ensemble runs with multiple realizations.
# This method also takes 'modes' as a parameter - used to reconstruct dataset with different number of selected modes
def generate_data(mode,mode_skip):
	flux_reconstr = model.reconstruct_randomized_X(new_fl[0],slice(1,mode,mode_skip))
	return flux_reconstr

def rmse_calc(rec_data,orig_data):
	rmse = mean_squared_error(10*np.log10(rec_data),10*np.log10(orig_data),squared=False)
	return rmse

def dropna(total_data):
	total_data = total_data.dropna('time',how='all')
	return clipped_data

def time_series(clipped_data):
	clipped_ts = clipped_data.sum(['y','x'])
	return clipped_ts

#modified to not return f - in calculation of RMSE, only Px required
def psd_calc_grid(data,y,x):
	f, Px = scipy.signal.welch(data[:,y,x])
	return Px

def remove_nans(data):
	new_data = data[~np.isnan(data)]
	return new_data

############################## MAIN SCRIPT ####################################

# Directory Path to file
inDirName = '/storage/home/hcoda1/6/smurugan9/scratch/rmse_tests_220911/'

flux_clean = xr.open_dataset(inDirName+'flux_clean') #Provide directory pathnames
flux_clean = flux_clean.timeMonthly_avg_landIceFreshwaterFlux

# Drop all NaN-valued grid points
flux_clean = flux_clean.dropna('time',how='all')
flux_clean = flux_clean.dropna('y',how='all')
flux_clean = flux_clean.dropna('x',how='all')
flux_clean = flux_clean.drop("month")

##############################
##### EOF DECOMPOSITION ######
##############################

model = EOF(flux_clean)
model.solve()
xeofs_eofs = model.eofs()
xeofs_pcs = model.pcs()
xeofs_n_modes = model.n_modes

##############################
# FOURIER PHASE RANDOMIZATION 
##############################

print("Creating random phase distributed realizations...")

# Define number of random Fourier realizations
n_realizations = 1
t_length = xeofs_pcs.shape[0]

# xeofs_pcs[:,i] when using PCA outputs
new_fl = np.empty((n_realizations,xeofs_pcs.shape[0],xeofs_pcs.shape[1]))

# Time limits for plotting
t1 = 0
tf = int(t_length/2)

for i in range(n_realizations):
	for m in range(xeofs_n_modes):
		fl = xeofs_pcs[:,m] # fluxpcs[:,i] when using PCA outputs
		fl_fourier = np.fft.rfft(fl)
		random_phases = np.exp(np.random.uniform(0,2*np.pi,int(len(fl)/2+1))*1.0j)
		fl_fourier_new = fl_fourier*random_phases
		new_fl[i,:,m] = np.fft.irfft(fl_fourier_new)
		print('calculated ifft for realization i, mode: {}'.format(m))

print("Created phase randomized realizations")

flux_clean_mean = flux_clean.mean('time')

mode_skip = 1
xeofs_modes = list(range(1,xeofs_n_modes+1))
yxcoords = np.argwhere(np.array(flux_clean_mean))
yxcoordsna = np.nonzero(np.array(flux_clean_mean))

orig_grid_psd = np.load(inDirName+"orig_grid_psd.npy")

# sklearn.metrics has a mean_squared_error function with a squared kwarg (defaults to True). 
# Setting squared to False will return the RMSE.
rmse_grid_comparisons = np.empty((xeofs_eofs.shape[0],xeofs_eofs.shape[1],xeofs_eofs.shape[2]))

print("GENERATING RECONSTRUCTED DATASETS")
for mode in xeofs_modes:
	print("Generating reconstructed data for mode: {}".format(mode*mode_skip))
	flux_reconstr = generate_data(mode,mode_skip)
	print("PSD across grid")
	for yx in yxcoords:
		if remove_nans(flux_reconstr[:,yx[0],yx[1]]).shape[0]>0:
			print('calculating rmse: [{},{}]'.format(yx[0],yx[1]))
			rmse_grid_comparisons[yx[0],yx[1],mode] = rmse_calc(remove_nans(psd_calc_grid(flux_reconstr,yx[0],yx[1])),remove_nans(orig_grid_psd[yx[0],yx[1],1,:]))
	print("Saving RMSE grid value file for mode: {}".format(mode*mode_skip))
	xr.DataArray(rmse_grid_comparisons[:,:,mode],coords=flux_clean_mean.coords,dims = flux_clean_mean.dims,attrs=flux_clean_mean.attrs).to_netcdf("rmse_grid_comparisons_EOF_{}.nc".format(mode))

# Convert Numpy array to xarray
print("Saving COMBINED RMSE grid value file")
rmse_grid_comparisons = xr.DataArray(rmse_grid_comparisons,coords=xeofs_eofs.coords,dims = xeofs_eofs.dims,attrs=xeofs_eofs.attrs)
rmse_grid_comparisons.to_netcdf("rmse_grid_comparisons_EOF_COMBINED.nc")