# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 09:20:38 2021

@author: jwbrooks
"""


##################################
# %% Import libraries

import numpy as _np
import matplotlib.pyplot as _plt
import xarray as _xr
from scipy.ndimage import gaussian_filter1d as _gaussian_filter1d

##################################
# %% Constants
from scipy.constants import c as _C, mu_0 as _MU_0, e as _E, m_e as _M_E, k as _K_B #, pi, , , h, hbar, G, g, , R, N_A, k as k_b, m_e, m_p, m_n

_AMU = 1.66054e-27 			# amu to kg
_EV = _E * 1.0				# ev to Joule
_EP_0 = 1.0 / (_MU_0 * _C**2)  # vacuum permittivity

_M_XE = 131.293 * _AMU		# mass of xenon
_M_AR = 39.948 * _AMU 		# mass of argon



##################################
# %% Plotting functions

def _update_title(axes, temperature_in_eV=_np.nan, n_e=_np.nan, debye=_np.nan, radius=_np.nan, V_float=_np.nan, V_plasma=_np.nan, f_p=_np.nan):
	new_title = 'Temperature = %.2f eV, \n Electron density = %.2e %s, \n Debye length = %.2e m, radius/debye = %.2e m \n V_float=%.2f V, V_plasma=%.2f V, f_plasma=%.2e Hz' % (temperature_in_eV, n_e, r'm$^{-3}$', debye, radius/debye, V_float, V_plasma, f_p)
	axes[0].set_title(new_title, fontsize=10)


def plot_IV_data(IV, 
				 fig=None, 
				 ax=None, 
				 label='', 
				 marker=None, 
				 linestyle=None, 
				 color=None, 
				 linthresh=1e-6, 
				 linewidth=2,
				 plot_deriv=True,
				 init=False
				 ): 
	""" subfunction for plotting IV data """
	
	if fig is None and ax is None:
		if plot_deriv is False:
			fig, ax = _plt.subplots()
		else:
			fig, ax = _plt.subplots(2, sharex=True)
			ax = list(ax)
	if type(ax) != list:
		ax = [ax]
		
	IV.plot(ax=ax[0], label=label, marker=marker, linestyle=linestyle, color=color, linewidth=linewidth)
	ax[0].legend(fontsize=8)
	ax[0].set_ylabel('Probe current [A]')
	ax[0].set_xlabel('Probe voltage [V]')
	if init is True:
		ax[0].set_yscale('symlog', linthresh=linthresh)
		ax[0].axhline(linestyle=':', color='grey')
		ax[0].axvline(linestyle=':', color='grey')
		fig.set_tight_layout(True)
	
	if plot_deriv is True:
		
		dIdV = _xr.DataArray(_np.gradient(IV) / _np.gradient(IV.V), 
					   coords=IV.coords)
		dIdV.plot(ax=ax[1], label='IV data')
		
		if init is True:
			ax[1].set_yscale('symlog', linthresh=linthresh)
			ax[1].axhline(linestyle=':', color='grey')
			ax[1].axvline(linestyle=':', color='grey')
		ax[1].set_ylabel(r'$\partial I / \partial V$, [A/V]')
		ax[1].set_xlabel('Probe voltage [V]')
			
	return fig, ax


##################################
# %% Misc subfunctions

def _smooth_IV_data(IV, smooth_width_in_volts=3, plot=False):
	""" smooths IV data using a gaussian filter with sigma = smooth_width_in_volts """
	
	if smooth_width_in_volts > 0:
		IV_smoothed = _xr.DataArray(_gaussian_filter1d(IV.data, sigma=smooth_width_in_volts), dims=IV.dims, coords=IV.coords)
	else:
		IV_smoothed = IV * 1.0
		
	if plot is True:
		fig, ax = _plt.subplots()
		IV.plot(ax=ax, marker='x', linestyle='', label='raw')
		IV_smoothed.plot(ax=ax, label='smoothed')
		ax.set_yscale('symlog', linthresh=1e-5)
		
	return IV_smoothed


def _polynomial_fit(da,	
				order=2, 
				plot=False,
				verbose=False):
	""" 
	Polynomial fit function.  This is effectively a wrapper for numpy.poly1d() but leverages xarray DataArrays
	
	Parameters
	----------
	da : xarray dataarray
		data with dimension
	order : int
		order of polynomial fit.  1 = linear, 2 = quadratic, etc.
	plot : bool
		Causes a plot of the fit to be generated
	verbose : bool
		print additional details
	
	Returns
	-------
	dfFit : pandas.core.frame.DataFrame with a the fit
		Dataframe with a single column
		index = independent variable
	ffit : numpy.poly1d
		fit function
	
	Example
	-------
	::
		import numpy as np
		import xarray as xr
		
		x = np.arange(0,1,0.1)
		x = xr.DataArray(x, coords={'x': x})
		y = 2 * x + 1
		_polynomial_fit(da=y, order=1, plot=True, verbose=True)
		 
	"""
	
	dim = da.dims[0]
	coord = da.coords[dim].data
	
	# perform fit
	fit_coefs, cov = _np.polyfit(	coord, 
							da.data, 
							deg=order,
							full=False,
							cov=True)
	fit_coef_error = _np.sqrt(_np.diag(cov))
		
	# create fit line from fit results
	ffit = _np.poly1d(fit_coefs)
	da_fit=_xr.DataArray(	ffit(coord),
							dims=dim,
							coords=[coord])
	
	if verbose is True:
		print("fit coeficients:")
		print(fit_coefs)
		print("fit errors:")
		print(fit_coef_error)
		
	if plot is True:
		
		x_highres = _np.linspace(coord[0], coord[-1], 1000)
		da_fit_highres = _xr.DataArray(	ffit(x_highres),
										dims=dim,
										coords=[_np.linspace(coord[0], coord[-1], 1000)])
		
		fig,ax = _plt.subplots()
		da.plot(ax=ax, label='raw' ,linestyle='', marker='x')
		da_fit_highres.plot(ax=ax, label='fit')
		
		ax.legend()
		
	return da_fit, ffit, fit_coefs, fit_coef_error


##################################
# %% Plasma parameters

def _calc_debye_length(density, temperature_in_eV):
	"""
	
	References
	----------
	* Merlino, 2007.  Eq. 4. https://aapt.scitation.org/doi/10.1119/1.2772282
	"""
	
	return _np.sqrt(_EP_0 * temperature_in_eV / ( density * _E) )


def _density_to_plasma_frequency(n0_in_m3):
 	""" Calculates the plasma frequency in Hz """
 	mass = _M_E
 	return _np.sqrt(n0_in_m3 * _E**2 / (mass * _EP_0)) * 1 / (2 * _np.pi)


##################################
# %% Langmuir analysis subfunctions

def _calc_electron_density(I_electron, 
						   V_plasma, 
						   temperature_in_eV, 
						   probe_area_in_m2):
	""" calculates electron density """
	I_e_sat = float(I_electron.interp(V=V_plasma))
	n_e = (I_e_sat / (_E * probe_area_in_m2)) * _np.sqrt(2 * _np.pi * _M_E / (temperature_in_eV * _EV))
	return n_e


def _calc_Vp_from_theory(temperature_in_eV, 
						 V_float, 
						 ion_mass):
	return _np.log(_np.sqrt(ion_mass / (2 * _np.pi * _M_E))) * temperature_in_eV + V_float
	

def _calc_Vp_from_deriv(I_electron_smoothed, 
						   V_float, 
						   plot=False):
	""" calculates V_plasma from the derivative of the smoothed V_float """
	
	# take derivative
	dIdV = _xr.DataArray(_np.gradient(I_electron_smoothed) / _np.gradient(I_electron_smoothed.V), dims=I_electron_smoothed.dims, coords=I_electron_smoothed.coords)
	
	# find the maximum value of dIdV above the floating potential
	Vp = float(dIdV.where(dIdV.V > V_float).idxmax())
	
	if plot is True:
		fig, (ax1, ax2) = _plt.subplots(2, sharex=True)
		I_electron_smoothed.plot(ax=ax1)
		dIdV.plot(ax=ax2)
		ax1.plot([Vp, Vp], [float(I_electron_smoothed.min()), float(I_electron_smoothed.max())], color='r', linestyle='--', label='V_plasma')
		ax2.plot([Vp, Vp], [float(dIdV.min()), float(dIdV.max())], color='r', linestyle='--', label='V_plasma')
		for a in [ax1, ax2]:
			a.legend()
		ax1.set_yscale('symlog', linthresh=1e-5)
		
	return Vp


def _calc_temp_from_electron_current(I_electron, 
								   exp_fit_bounds, 
								   dV, 
								   plot=False):
	""" Calculates temperature by fitting a linear fit to the natural log of the electron current (roughly equivalent of fitting an exponential fit to the unmodified electron current) """
	
	## window the electron current to the "valid" range
	Ie_windowed = I_electron[(I_electron.V <= exp_fit_bounds[1]) & (I_electron.V >= exp_fit_bounds[0])]
	
	## check that the windowed electron current is valid
	if len(Ie_windowed) == 2:
		print("Warning: 3 points minimum are required between V_float and V_plasma.  2 found.  Expanding window to include a 3rd point.  ")
		exp_fit_bounds = [exp_fit_bounds[0], exp_fit_bounds[1] + dV]
		Ie_windowed = I_electron[(I_electron.V <= exp_fit_bounds[1]) & (I_electron.V >= exp_fit_bounds[0])]
	elif len(Ie_windowed) < 2 :
		raise Exception("3 points minimum are required between V_float and V_plasma.  Possible reasons:  1) not enough points.  2) Your guess for temperature or V_plasma is too low.")
	if bool((Ie_windowed < 0).sum() > 0):
		raise Exception("Electron current must be positive to perform this fit.  Check that your I_sat fit is correct. " )
		
	## take the natural log of the electron current and apply a linear fit
	ln_Ie = _xr.DataArray(_np.log(Ie_windowed), dims=Ie_windowed.dims, coords=Ie_windowed.coords)
	lin_fit, ffit, _, _ = _polynomial_fit(ln_Ie, order=1, plot=False, verbose=False)
	
	## calculate temperature
	temperature_in_eV = 1 / ffit.coeffs[0]
	
	## convert the linear fit results to exponetial
	exp_fit = _np.exp(lin_fit)
	
	if plot is True:
		fig, ax = _plt.subplots()
		I_electron.plot(ax=ax, label='I_electron', ls='', marker='x')
		exp_fit.plot(ax=ax, label='fit', color='tab:blue')
	
	return temperature_in_eV, exp_fit


def _find_V_float(IV, 
					 smooth_width_in_volts=1.0, 
					 plot=False):
	""" Finds V_float by interporloating V(I)=0 """
	
	I_min = _np.abs(float(IV.min()))
	
	if smooth_width_in_volts > 0:
		IV = _smooth_IV_data(IV, plot=False, smooth_width_in_volts=smooth_width_in_volts)
		
	# swamp I(V) to V(I)
	VI = _xr.DataArray(IV.V.data, dims='I', coords=[IV.data])
	
	# find the zero crossing in V(I)
	# TODO write code to handle if there is more than one zero crossing
	V_float = float(VI.where((VI.I>=-I_min) & (VI.I<=I_min)).interp(I=0))
	
	if plot is True:
		fig, ax = _plt.subplots(2, sharex=True)
		plot_IV_data(IV, fig=fig, ax=ax, label='IV')
		ax.axvline(V_float, linestyle='--', color='r')

	return float(V_float)


##################################
# %% Probe analysis

def IV_sweep_analysis(	IV, 
						probe_area_m2, 
						ion_mass, 
						V_lim=[], 
						V_ion_current_exp_range=[],
						V_isat_fit_range=[],
						temperature_in_eV_guess=1.0,
						Vp_guess=None,
						smooth_width_in_volts=0.0,
						plot=True, 
						calc_Vp_from_deriv=False, # False is the "safer" option.
						verbose=False, 
						plot_intermediate_steps=False,
						remove_Vp_guess_at_end=True,
						probe_geometry='planar', # 'planar', 'cylindrical', 'spherical'
						): 
	"""
	Performs a single Langmuir Probe analysis.
	This code is primarily based on the work in reference [1].
	

	Parameters
	----------
	IV : xarray.DataArray
		Dataarray containing current (in Amps) with coordinate V (in volts)
	probe_area_m2 : float
		Area of the probe with units in m^2
	ion_mass : float
		Mass of the ion in kg.  
	V_lim : sequence of two floats (or empty list)
		Trims the IV data between the two V values in V_lim=[V_lim_lower, V_lim_upper]. The default is an empty list which means no trimming.
	V_ion_current_exp_range : sequence of two floats (or empty list)
		Defines the voltage range where the exponential fit is used to determine the temperature.
		If an empty list, the "exponential" range of the I_electron is auto calculated from V_float and V_plasma
		If a sequence of two floats, this range is used instead (i.e. it's a manual overide) 
	V_isat_fit_range : sequence of two floats (or empty list)
		Defines the voltage range where a linear fit is used to model the ion saturation current
		If an empty list, the range of the I_electron is auto calculated from V_float.
		If a sequence of two floats, this range is used instead (i.e. it's a manual overide) 
   	temperature_in_eV_guess : float
		Initial guess at the temperature.  Units in eV. This is used to determining an initial guess at V_plasma (assuming Vp_guess = None)
   	Vp_guess : float
		Initial guess at V_plasma.  If Vp_guess = None, Vp_guess is calculated using temperature_in_eV_guess.  Default is None.
	smooth_width_in_volts : float
		Used for smoothing the IV data before various steps.  The value is sigma in a gaussian smoothing function and therefore has units in Volts.  If 0, no smoothing is applied.
	plot : bool
		Optionally generates a plot of the results.  
	calc_Vp_from_deriv : bool, optional
		There are two methods for calculating V_plasma.  If True, this code uses the derivative of IV to calculate V_plasma.  If False, it uses the theory of V_plasma. Note that the derivative method is more likely to fail (for various reasons), and therefore the theory method is default.   
	verbose : bool, optional
		Optionally prints the results of various intermediate results. Useful for debugging.
	plot_intermediate_steps : bool, optional
		Plots the results of various itermediate steps.  Useful for debugging.
	plot_intermediate_steps : bool, optional
		Removes V_plasma guess from the plot at the end of the code.  Leaving it there is useful for debugging.

	Returns
	-------
	temperature_in_eV : float
		Temperature with units in eV.
	n_e : float
		Electron density, units in m^-3.
	V_float : float
		Floating potential, units in V.
	V_plasma : float
		Plasma potential, units in V.
	debye_length : float
		Debye length, units in m.
	f_p : float
		Plasma frequency, units in Hz.
				
	Notes
	-----
	 * This work is provided "as is" with no gaurantees as to its accuracy or the presence of bugs.  
	 * There are a lot of "nuances" associated with this code and Langmuir probe analysis in general.  I.e. this code can be confusing, make assumptions invalid for your application, there are many "conventions" concerning Langmuir probe analysis, etc.  
	 * PLEASE read through all references and this docstring first before submitting any questions to me.  This being said, please let me know if you discover any bugs or have other suggestions for improvements.  
	 * This code (at present) does not calculate the final ion density as discussed in [1].  
	 * V_plasma can be calculated in two ways in this code.  First is using the derivative of a smoothed electron current.  Note that this method does not always work particularly if SNR is low, there aren't enough points, and if the sheath is 'thick'.  The second method uses the definition of V_plasma in Lamguir probe theory.  While potentially less accurate, this method is much more robust. 
	 * If you need to debug the code (or debug how the code processes your data), set vebose=True and plot_intermediate_steps=True.   
		
	References
	----------
	 * [1] Lobbia and Beal, "Recommended Practice for Use of Langmuir Probes in Electric Propulsion Testing" https://doi.org/10.2514/1.B35531
	 * [2] Merlino, "Understanding Langmuir probe current-voltage characteristics" https://aapt.scitation.org/doi/10.1119/1.2772282
		
	"""

	if verbose: print("Trimming voltage range if requested. ")
	if type(V_lim) is list:
		if len(V_lim) == 2:
			IV = IV.where((IV.V >= V_lim[0]) & (IV.V <= V_lim[1])).dropna('V')
		
	if verbose: print("Performing misc. calculations. ")
	dV = _np.mean(IV.V[1:].values - IV.V[:-1].values) # calculate the average voltage step
	radius = _np.sqrt(probe_area_m2 / _np.pi) # probe radius, assuming circular
	
	if verbose: print("Step 1: Finding the floating potential by idenfying the zero intercept. ")
	V_float = _find_V_float(IV, smooth_width_in_volts=smooth_width_in_volts, plot=plot_intermediate_steps)

	if verbose: print("Step 2: Calculating the ion current by performing a linear fit. ")
	if len(V_isat_fit_range) == 0:
		V_isat_fit_range = [-_np.inf, (V_float - IV.V[0].data) / 1.25 + IV.V[0].data] # TODO This 1.25 factor is somewhat arbitrary.  Is there a better way to implement this?
	I_windowed_for_isat = IV[IV.V < V_isat_fit_range[1]]
	I_ion_fit, I_ion_fit_func, _, _ = _polynomial_fit(I_windowed_for_isat, order=1, plot=plot_intermediate_steps)	## (optional) initialize plot
	
	if plot is True:
		if verbose: print("Initializing plot. ")	
		linthresh = 10 ** (_np.floor(_np.log10(_np.abs(float(_np.abs(I_ion_fit).min())))) - 1)		
		fig, axes = plot_IV_data(IV, label='IV data', marker='x',  linestyle='', linthresh=linthresh, plot_deriv=True, init=True, )
		for a in axes:
			a.axvline(V_float, label='V_float', linestyle='--', color='k')
		plot_IV_data(I_ion_fit.where(I_ion_fit.V<=V_float), fig=fig, ax=axes,  label='Ion current fit', linewidth=2, plot_deriv=False, )
		_update_title(axes, V_float=V_float, radius=radius)

	if verbose: print("Step 3: Solving for the electron current by subtracting the ion saturation linear fit from I(V). ")
	I_electron = IV.copy() - I_ion_fit_func(IV.V)
	I_electron_smoothed = _smooth_IV_data(I_electron, smooth_width_in_volts=smooth_width_in_volts, plot=False)
	if plot is True:
		plot_IV_data(I_electron, fig=fig, ax=axes, label='Electron current', marker='+', linestyle='', plot_deriv=False,)
		_update_title(axes, V_float=V_float, radius=radius)
		
	if verbose: print("Step 4: Guessing at V_plasma either by 1) using the provided guess or 2) calculating it from the provided temperature guess. ")
	if type(Vp_guess) is not type(None):
		V_plasma = Vp_guess
	else:
		V_plasma = _calc_Vp_from_theory(temperature_in_eV_guess, V_float, ion_mass, )	
	if plot is True:
		for a in axes:
			a.axvline(V_plasma, label='V_plasma guess', linestyle='--', color='grey')
		axes[0].legend()
		_update_title(axes, V_float=V_float, radius=radius)
			
	if verbose: print("Step 5: Iteratively solving for V_plasma and temperature until convergence. ")
	for i in range(10): # TODO presently this is a brute-force interation of 10 cycles.  I can do this more intelligently.
		
		## Determines the bounds for the exponential region for fitting
		if len(V_ion_current_exp_range) == 2:
			exp_fit_bounds = V_ion_current_exp_range
		else:
			exp_fit_bounds = [V_float, (V_plasma - V_float) / 1.5 + V_float] # TODO This 1.5 factor is somewhat arbitrary.  Is there a better way to implement this?
		
		## Calclutes kTe/e (temperature with units in eV)
		temperature_in_eV, log_fit = _calc_temp_from_electron_current(I_electron, 
																exp_fit_bounds=exp_fit_bounds, 
																dV=dV, 
																plot=plot_intermediate_steps)
		 			
		## Calculate V_plasma
		if calc_Vp_from_deriv is True:
			V_plasma = _calc_Vp_from_deriv(I_electron_smoothed, V_float, plot=plot_intermediate_steps)
		else:
			V_plasma = _calc_Vp_from_theory(temperature_in_eV, V_float, ion_mass)
			
		if verbose:
			print("Converging on V_plasma and temp: ", V_plasma, temperature_in_eV)
			
	if plot is True:
		for a in axes:
			a.axvline(V_plasma, label='V_plasma', linestyle='--', color='r')
		plot_IV_data(log_fit, fig=fig, ax=axes, label='Electron current fit', linewidth=2, linthresh=linthresh, plot_deriv=False,)
		_update_title(axes, V_float=V_float, radius=radius, temperature_in_eV=temperature_in_eV, V_plasma=V_plasma)

	if verbose: print("Step 6: Calcuating electron density, debye length, and plasma frequency ")
	n_e = _calc_electron_density(I_electron, V_plasma, temperature_in_eV, probe_area_in_m2=probe_area_m2, )
	debye = _calc_debye_length(n_e, temperature_in_eV)
	f_p = _density_to_plasma_frequency(n_e)
	if plot is True:
		_update_title(axes, V_float=V_float, radius=radius, temperature_in_eV=temperature_in_eV, n_e=n_e, debye=debye, V_plasma=V_plasma, f_p=f_p)
		
	if verbose: print("Step 7: Iteratively solving for the ion density.  This is not implemented because I can't  the method described in reference #1 to work.  Possible typo in reference?  ")
		
	if plot is True:
		if verbose: print("Finalizing figure. ")
		if remove_Vp_guess_at_end is True:
			# remove V_plasma guess and update legend
			axes[0].lines.pop(6) 
			axes[1].lines.pop(4) 
			axes[0].legend(fontsize=8)
		
	plasma_params = {'temperature_in_eV': temperature_in_eV, 
				  'n_e_in_m3': n_e, 
				  'V_float': V_float, 
				  'V_plasma': V_plasma, 
				  'debye_length_in_m': debye, 
				  'f_p': f_p}
		
	return plasma_params


# %% Examples

def _example_1():
		
	V = _np.array([-100,  -98,  -96,  -94,  -92,  -90,  -88,  -86,  -84,  -82,  -80,
			        -78,  -76,  -74,  -72,  -70,  -68,  -66,  -64,  -62,  -60,  -58,
			        -56,  -54,  -52,  -50,  -48,  -46,  -44,  -42,  -40,  -38,  -36,
			        -34,  -32,  -30,  -28,  -26,  -24,  -22,  -20,  -18,  -16,  -14,
			        -12,  -10,   -8,   -6,   -4,   -2,    0,    2,    4,    6,    8,
			         10,   12,   14,   16,   18,   20,   22,   24,   26,   28,   30,
			         32,   34,   36,   38,   40,   42,   44,   46,   48,   50,   52,
			         54,   56,   58,   60,   62,   64,   66,   68,   70,   72,   74,
			         76,   78,   80,   82,   84,   86,   88,   90,   92,   94,   96,
			         98,  100], dtype=_np.int64)
	V = _xr.DataArray(V, coords={'V': V})
	
	IV = _np.array([-1.23276900e-04, -1.22001080e-04, -1.20853020e-04, -1.20487620e-04,
					-1.19268280e-04, -1.18300420e-04, -1.17165280e-04, -1.16233260e-04,
					-1.16530440e-04, -1.15633040e-04, -1.13892340e-04, -1.14477320e-04,
					-1.13450540e-04, -1.12470120e-04, -1.12349080e-04, -1.11128680e-04,
					-1.10094840e-04, -1.09742440e-04, -1.09405480e-04, -1.08979680e-04,
					-1.08711440e-04, -1.05849160e-04, -1.04700300e-04, -1.03686260e-04,
					-1.02397100e-04, -9.95354100e-05, -9.93067500e-05, -9.97792140e-05,
					-9.73350000e-05, -9.68748660e-05, -9.39701360e-05, -9.52584700e-05,
					-9.07786520e-05, -8.95539980e-05, -8.71274700e-05, -8.61539800e-05,
					-8.44392260e-05, -8.40099840e-05, -8.10891440e-05, -7.92816260e-05,
					-7.75247040e-05, -7.65311840e-05, -7.42542520e-05, -7.24106240e-05,
					-7.04281680e-05, -6.85515700e-05, -6.63375440e-05, -6.39776280e-05,
					-6.15081140e-05, -5.86534380e-05, -5.53086700e-05, -5.16001340e-05,
					-4.73786300e-05, -4.23687600e-05, -3.63233180e-05, -2.93519880e-05,
					-2.07794340e-05, -1.02661894e-05,  2.37433140e-06,  1.72913800e-05,
					3.82262440e-05,  6.47717940e-05,  9.75442640e-05,  1.42001240e-04,
					2.02602000e-04,  2.83107120e-04,  3.89824300e-04,  5.27021640e-04,
					7.11715460e-04,  9.35655740e-04,  1.24722080e-03,  1.64790500e-03,
					2.14126600e-03,  2.71130120e-03,  3.31452180e-03,  3.90771700e-03,
					4.43719840e-03,  4.90140700e-03,  5.33486520e-03,  5.72717280e-03,
					6.10344460e-03,  6.43154540e-03,  6.75174820e-03,  7.05013120e-03,
					7.35083360e-03,  7.63306280e-03,  7.92161560e-03,  8.19307780e-03,
					8.44763480e-03,  8.73286540e-03,  8.97885120e-03,  9.24260080e-03,
					9.48413020e-03,  9.71585420e-03,  9.97254480e-03,  1.01798780e-02,
					1.04262940e-02,  1.06384120e-02,  1.08609340e-02,  1.10935240e-02,
					1.13296100e-02])
	IV = _xr.DataArray(IV, coords={'V': V})
	
	results = IV_sweep_analysis(IV, 
						probe_area_m2=10e-5, 
						plot=True, 
						ion_mass=_M_AR, 
						temperature_in_eV_guess=2.0,
						calc_Vp_from_deriv=True,
						remove_Vp_guess_at_end=False,
						)
	print(results)
	fig = _plt.gcf()
	fig.savefig('Example_1_results.png', dpi=150)


def _example_2():

	V = _np.array([-1.499973e+01, -1.484968e+01, -1.469975e+01, -1.454971e+01,
			       -1.439975e+01, -1.424961e+01, -1.409971e+01, -1.394971e+01,
			       -1.379974e+01, -1.364971e+01, -1.349979e+01, -1.334976e+01,
			       -1.319979e+01, -1.304976e+01, -1.289977e+01, -1.274974e+01,
			       -1.259981e+01, -1.244977e+01, -1.229981e+01, -1.214978e+01,
			       -1.199978e+01, -1.184979e+01, -1.169986e+01, -1.154984e+01,
			       -1.139975e+01, -1.124979e+01, -1.109980e+01, -1.094980e+01,
			       -1.079979e+01, -1.064981e+01, -1.049987e+01, -1.034986e+01,
			       -1.019985e+01, -1.004980e+01, -9.899834e+00, -9.749847e+00,
			       -9.599815e+00, -9.449823e+00, -9.299861e+00, -9.149871e+00,
			       -8.999863e+00, -8.849897e+00, -8.699842e+00, -8.549867e+00,
			       -8.399829e+00, -8.249869e+00, -8.099882e+00, -7.949901e+00,
			       -7.799897e+00, -7.649949e+00, -7.499926e+00, -7.349809e+00,
			       -7.199850e+00, -7.049883e+00, -6.899891e+00, -6.749852e+00,
			       -6.599858e+00, -6.449876e+00, -6.299905e+00, -6.149883e+00,
			       -5.999815e+00, -5.849849e+00, -5.699894e+00, -5.549869e+00,
			       -5.399874e+00, -5.249872e+00, -5.099908e+00, -4.949860e+00,
			       -4.799900e+00, -4.649913e+00, -4.499886e+00, -4.349853e+00,
			       -4.199869e+00, -4.049881e+00, -3.899863e+00, -3.749862e+00,
			       -3.599859e+00, -3.450253e+00, -3.300259e+00, -3.150145e+00,
			       -3.000186e+00, -2.850183e+00, -2.700122e+00, -2.550167e+00,
			       -2.400219e+00, -2.250255e+00, -2.100219e+00, -1.950238e+00,
			       -1.800149e+00, -1.650200e+00, -1.500146e+00, -1.350178e+00,
			       -1.200158e+00, -1.050123e+00, -9.001283e-01, -7.501342e-01,
			       -6.001464e-01, -4.500615e-01, -3.000392e-01, -1.500380e-01,
			       -3.784377e-04,  1.495834e-01,  2.995661e-01,  4.495486e-01,
			        5.995997e-01,  7.499099e-01,  8.998775e-01,  1.049913e+00,
			        1.199937e+00,  1.349902e+00,  1.499926e+00,  1.649957e+00,
			        1.799877e+00,  1.949888e+00,  2.099878e+00,  2.249976e+00,
			        2.399876e+00,  2.549787e+00,  2.699841e+00,  2.849850e+00,
			        2.999810e+00,  3.149664e+00,  3.299942e+00,  3.449826e+00,
			        3.599701e+00,  3.749762e+00,  3.899835e+00,  4.049822e+00,
			        4.199820e+00,  4.349881e+00,  4.499726e+00,  4.649821e+00,
			        4.799754e+00,  4.949655e+00,  5.099750e+00,  5.249698e+00,
			        5.399639e+00,  5.549645e+00,  5.699758e+00,  5.849683e+00,
			        5.999548e+00,  6.149669e+00,  6.299622e+00,  6.449628e+00,
			        6.599553e+00,  6.749552e+00,  6.899547e+00,  7.049587e+00,
			        7.199537e+00,  7.349543e+00,  7.499639e+00,  7.649571e+00,
			        7.799484e+00,  7.949542e+00,  8.099564e+00,  8.249532e+00,
			        8.399451e+00,  8.549488e+00,  8.699456e+00,  8.849481e+00,
			        8.999370e+00,  9.149478e+00,  9.299845e+00,  9.449759e+00,
			        9.599732e+00,  9.749868e+00,  9.899672e+00,  1.004977e+01,
			        1.019977e+01,  1.034958e+01,  1.049974e+01,  1.064964e+01,
			        1.079960e+01,  1.094965e+01,  1.109969e+01,  1.124950e+01,
			        1.139960e+01,  1.154961e+01,  1.169968e+01,  1.184950e+01,
			        1.199959e+01,  1.214960e+01,  1.229962e+01,  1.244938e+01,
			        1.259958e+01,  1.274939e+01,  1.289953e+01,  1.304946e+01,
			        1.319948e+01,  1.334938e+01,  1.349940e+01,  1.364932e+01,
			        1.379943e+01,  1.394927e+01,  1.409934e+01,  1.424928e+01,
			        1.439936e+01,  1.454933e+01,  1.469946e+01,  1.484933e+01,
			        1.499934e+01])
	V = _xr.DataArray(V, coords={'V': V})
	
	IV = _np.array([-7.145543e-05, -7.121362e-05, -7.141394e-05, -7.123117e-05,
			       -7.122692e-05, -7.125218e-05, -7.120864e-05, -7.125502e-05,
			       -7.117562e-05, -7.110740e-05, -7.106102e-05, -7.115946e-05,
			       -7.111306e-05, -7.115379e-05, -7.109898e-05, -7.089018e-05,
			       -7.095840e-05, -7.102656e-05, -7.087403e-05, -7.090987e-05,
			       -7.098580e-05, -7.091688e-05, -7.092464e-05, -7.081568e-05,
			       -7.079108e-05, -7.083883e-05, -7.099000e-05, -7.076502e-05,
			       -7.093516e-05, -7.063431e-05, -7.071585e-05, -7.053165e-05,
			       -7.063149e-05, -7.062514e-05, -7.070670e-05, -7.061181e-05,
			       -7.067156e-05, -7.060896e-05, -7.049228e-05, -7.039245e-05,
			       -7.066946e-05, -7.045786e-05, -7.025469e-05, -7.047821e-05,
			       -7.020120e-05, -7.035238e-05, -7.025252e-05, -7.042193e-05,
			       -7.005711e-05, -7.016470e-05, -7.016747e-05, -7.017311e-05,
			       -6.986943e-05, -6.982725e-05, -6.985674e-05, -7.001214e-05,
			       -6.978645e-05, -6.984764e-05, -6.986590e-05, -6.964448e-05,
			       -6.971612e-05, -6.967886e-05, -6.959244e-05, -6.957415e-05,
			       -6.951933e-05, -6.934358e-05, -6.930841e-05, -6.942656e-05,
			       -6.905325e-05, -6.900541e-05, -6.899839e-05, -6.875729e-05,
			       -6.874604e-05, -6.853019e-05, -6.846622e-05, -6.822513e-05,
			       -6.809717e-05, -6.806696e-05, -6.779349e-05, -6.749897e-05,
			       -6.730774e-05, -6.732671e-05, -6.678966e-05, -6.656260e-05,
			       -6.628839e-05, -6.580617e-05, -6.521148e-05, -6.458931e-05,
			       -6.389550e-05, -6.342803e-05, -6.230114e-05, -6.115039e-05,
			       -5.962986e-05, -5.751532e-05, -5.518566e-05, -5.211228e-05,
			       -4.800619e-05, -4.299120e-05, -3.623142e-05, -2.699860e-05,
			       -1.571450e-05, -1.238188e-06,  1.728141e-05,  4.108515e-05,
			        6.983356e-05,  1.064010e-04,  1.502969e-04,  2.041361e-04,
			        2.685668e-04,  3.440138e-04,  4.301176e-04,  5.220416e-04,
			        6.150635e-04,  7.155293e-04,  8.135985e-04,  9.017167e-04,
			        9.847557e-04,  1.071977e-03,  1.153845e-03,  1.229339e-03,
			        1.304662e-03,  1.376961e-03,  1.447490e-03,  1.503644e-03,
			        1.577138e-03,  1.639192e-03,  1.692412e-03,  1.754970e-03,
			        1.814801e-03,  1.861009e-03,  1.908812e-03,  1.962658e-03,
			        2.013219e-03,  2.069958e-03,  2.107877e-03,  2.154376e-03,
			        2.207538e-03,  2.257471e-03,  2.301678e-03,  2.341955e-03,
			        2.375652e-03,  2.418524e-03,  2.460209e-03,  2.503856e-03,
			        2.553390e-03,  2.594628e-03,  2.633608e-03,  2.681029e-03,
			        2.709393e-03,  2.754273e-03,  2.791478e-03,  2.827741e-03,
			        2.874978e-03,  2.908827e-03,  2.945527e-03,  2.987957e-03,
			        3.022253e-03,  3.061570e-03,  3.088473e-03,  3.136570e-03,
			        3.170972e-03,  3.205859e-03,  3.238039e-03,  3.273305e-03,
			        3.307571e-03,  3.358213e-03,  3.385822e-03,  3.430119e-03,
			        3.454209e-03,  3.493532e-03,  3.522139e-03,  3.561349e-03,
			        3.583335e-03,  3.617687e-03,  3.661373e-03,  3.690861e-03,
			        3.723268e-03,  3.768160e-03,  3.795108e-03,  3.822195e-03,
			        3.860311e-03,  3.881651e-03,  3.918089e-03,  3.953086e-03,
			        3.984898e-03,  4.017197e-03,  4.046915e-03,  4.078487e-03,
			        4.121232e-03,  4.155159e-03,  4.178535e-03,  4.212283e-03,
			        4.230295e-03,  4.274660e-03,  4.319109e-03,  4.340771e-03,
			        4.373666e-03,  4.398801e-03,  4.418454e-03,  4.461163e-03,
			        4.487215e-03])
	IV = _xr.DataArray(IV, coords={'V': V})
	
	results = IV_sweep_analysis(IV,  
								probe_area_m2=5e-5, 
								plot=True, 
								ion_mass=_M_AR, 
								temperature_in_eV_guess=2.0,
								calc_Vp_from_deriv=True,
								remove_Vp_guess_at_end=False,
								)
	print(results)
	fig = _plt.gcf()
	fig.savefig('Example_2_results.png', dpi=150)


def _example_3():
	
	V = _np.array([-1.499971e+01, -1.462474e+01, -1.424964e+01, -1.387467e+01,
			       -1.349977e+01, -1.312471e+01, -1.274975e+01, -1.237478e+01,
			       -1.199978e+01, -1.162481e+01, -1.124979e+01, -1.087484e+01,
			       -1.049987e+01, -1.012479e+01, -9.749844e+00, -9.374825e+00,
			       -8.999907e+00, -8.624867e+00, -8.249838e+00, -7.874883e+00,
			       -7.499931e+00, -7.124850e+00, -6.749890e+00, -6.374926e+00,
			       -5.999762e+00, -5.624877e+00, -5.249894e+00, -4.874905e+00,
			       -4.499984e+00, -4.124847e+00, -3.749755e+00, -3.375207e+00,
			       -3.000221e+00, -2.625212e+00, -2.250260e+00, -1.875177e+00,
			       -1.500156e+00, -1.125132e+00, -7.501688e-01, -3.750290e-01,
			       -4.579620e-04,  3.745723e-01,  7.498841e-01,  1.124853e+00,
			        1.499875e+00,  1.874840e+00,  2.249921e+00,  2.624787e+00,
			        2.999825e+00,  3.374828e+00,  3.749765e+00,  4.124767e+00,
			        4.499752e+00,  4.874761e+00,  5.249667e+00,  5.624671e+00,
			        5.999593e+00,  6.374692e+00,  6.749636e+00,  7.124612e+00,
			        7.499596e+00,  7.874525e+00,  8.249495e+00,  8.624478e+00,
			        8.999470e+00,  9.374757e+00,  9.749790e+00])
	V = _xr.DataArray(V, coords={'V': V})
	
	IV = _np.array([-2.354910e-05, -2.337689e-05, -2.333330e-05, -2.315684e-05,
			       -2.312873e-05, -2.287987e-05, -2.272804e-05, -2.252345e-05,
			       -2.242224e-05, -2.214035e-05, -2.191048e-05, -2.152805e-05,
			       -2.135232e-05, -2.087289e-05, -2.046797e-05, -1.988239e-05,
			       -1.913654e-05, -1.839560e-05, -1.754149e-05, -1.626701e-05,
			       -1.522872e-05, -1.410255e-05, -1.255321e-05, -1.052233e-05,
			       -8.897056e-06, -6.824698e-06, -4.168166e-06, -1.135552e-06,
			        3.266419e-06,  9.238087e-06,  1.759983e-05,  2.910872e-05,
			        4.353001e-05,  6.330292e-05,  9.093568e-05,  1.305602e-04,
			        1.892568e-04,  2.781675e-04,  4.061890e-04,  5.840449e-04,
			        7.966693e-04,  9.889465e-04,  1.138533e-03,  1.263847e-03,
			        1.376419e-03,  1.476015e-03,  1.566066e-03,  1.655117e-03,
			        1.733397e-03,  1.819523e-03,  1.891250e-03,  1.970499e-03,
			        2.048515e-03,  2.123594e-03,  2.184927e-03,  2.257848e-03,
			        2.320800e-03,  2.395195e-03,  2.454572e-03,  2.519317e-03,
			        2.589291e-03,  2.647418e-03,  2.720309e-03,  2.770979e-03,
			        2.836579e-03,  2.894709e-03,  2.950749e-03])
	IV = _xr.DataArray(IV, coords={'V': V})
	
	results = IV_sweep_analysis(IV,  
								probe_area_m2=20e-5, 
								plot=True, 
								ion_mass=_M_XE, 
								temperature_in_eV_guess=0.5,
								calc_Vp_from_deriv=True,
								V_isat_fit_range=[-_np.inf, -10],
								remove_Vp_guess_at_end=False,
								smooth_width_in_volts=0.0,
								)
	print(results)
	fig = _plt.gcf()
	fig.savefig('Example_3_results.png', dpi=150)
	
	
# TODO Put some "bad" data in here to test various borderline and failure cases

	
# %% if __name__ == "__main__"

if __name__ == "__main__":
	_example_1()
	_example_2()
	_example_3()