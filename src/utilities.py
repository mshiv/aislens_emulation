import xarray as xr
import numpy as np
import xeofs.xarray as EOF


def EOFdecomp(data):
	model = EOF(data)
	model.solve()
	eofs = model.eofs()
	pcs = model.pcs()
	modes = model.n_modes
	return eofs, pcs, modes

def surrogate_ts(fl):
	# fl : Input time series, i.e., projection co-efficient series
	fl_fourier = np.fft.rfft(fl)
	random_phases = np.exp(np.random.uniform(0,2*np.pi, int(len(fl)/2+1))*1.0j)
	fl_fourier_new = fl_fourier*random_phases
	new_fl = np.fft.irfft(fl_fourier_new)
	return new_fl


