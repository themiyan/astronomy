import glob
import numpy as np
from astropy.io import ascii, fits
import matplotlib.pyplot as plt
import pandas as pd
from lmfit.models import GaussianModel



class generate_telluric:

	def __init__(self, band, star_name , path_A, path_C, star_temperature, obs_date, cen_lp):

		self.star_name   = star_name
		self.star_file_A = path_A
		self.star_file_C = path_C
		self.star_temp   = star_temperature
		self.band		 = band
		self.obs_date 	 = obs_date
		self.center_of_spatial_profile = cen_lp


		### until issues with long2pos is sorted out use only posA
		self.star_flux, self.star_wave, self.star_error = self.combine_pA_and_pC()

		### mask out unobsrved region of the star
		### this should be a temporary measure (confirm when long2pos delta_lambda is checked)
		### so the slit position dependant wavelength coverage is taken into account
		### the main reason for this is for the stellar H line masking
		observed_range_mask = [self.star_error!=0]
		self.star_flux, self.star_wave, self.star_error = self.star_flux[observed_range_mask], self.star_wave[observed_range_mask], self.star_error[observed_range_mask]


	def combine_pA_and_pC(self):

		"""
		Combine the poA and posC together
		THERE ARE ISSUES WITH VARIABLE FLUXES 
		So ideally the code should check if fluxes are with 10% between A and B for combining: mainly only 
		relavant for the flux standard: need to investigate if normalization issues in telluric

		"""

		import copy

		import pdb

		star_flux_A, star_wave_A, star_error_A = extract_1d(self, self.star_file_A, long2pos='_posA')

		star_flux_C, star_wave_C, star_error_C = extract_1d(self, self.star_file_C, long2pos='_posC')

		assert (star_wave_A.all()==star_wave_C.all()), 'wavelength solution of the two stars does not match' 

		### band centre + (FWHM/2) is chosen as the range to be the common area between A and C
		AC_mask = [(star_wave_A>(band_center_wave-(band_fwhm/2))) & (star_wave_A<(band_center_wave+(band_fwhm/2)))] 
		AC_mask = np.asarray(AC_mask)[0]

		# pdb.set_trace()


		### check what is the accuracy of the total flux between the two positions
		
		A_mask_flux = np.nansum(star_flux_A[AC_mask] * np.diff(star_wave_A[AC_mask])[0])
		C_mask_flux = np.nansum(star_flux_C[AC_mask] * np.diff(star_wave_C[AC_mask])[0])
		A_C_flux_diff = ((A_mask_flux - C_mask_flux)/C_mask_flux)
		print('Flux difference between A an C is ', A_C_flux_diff)
		assert  A_C_flux_diff<0.30, 'Pos A and C flux around band centre differs by >30%'

		star_flux, star_error = copy.deepcopy(star_flux_C), copy.deepcopy(star_error_C)

		# pdb.set_trace()

		star_flux[AC_mask] = np.nanmean([star_flux_A[AC_mask], star_flux_C[AC_mask]], axis=0)

		star_error[AC_mask] = ((star_error_A[AC_mask]**2)+(star_error_C[AC_mask]**2))**0.5

		# pdb.set_trace()

		#### IMPORTANT VERIFY THAT C COVERS THE BLUE END
		#### Need a better way to generalize this
		#### Error is generally nan for uncovered regions 
		star_flux[(~AC_mask) & (star_wave_A>band_center_wave)] = star_flux_A[(~AC_mask) & (star_wave_A>band_center_wave)]
		star_flux[(~AC_mask) & (star_wave_C<band_center_wave)] = star_flux_C[(~AC_mask) & (star_wave_C<band_center_wave)]

		### Check error propogation...actually this error spectrum is only used for optimal 1D extraction
		star_error[(~AC_mask) & (star_wave_A>band_center_wave)] = star_error_A[(~AC_mask) & (star_wave_A>band_center_wave)]
		star_error[(~AC_mask) & (star_wave_C<band_center_wave)] = star_error_C[(~AC_mask) & (star_wave_C<band_center_wave)]

		# pdb.set_trace()

		return star_flux, star_wave_A, star_error


	def smooth_stellar_spectra(self):
		"""
		smooth a 1D spectrum using a gaussian kernel from astropy
		smoothing is done using a stddev=3 kernel. This seems to give 
		a good balance between smoothing and retaining atmosphereic features. 
		Change this as required. 
		Note: if the stdev is too low the solution will be have too much stellar features incoporated.
		
		input: self
		returns: smoothed flux, smoothed error

		"""
		from astropy.convolution import Gaussian1DKernel, convolve

		g = Gaussian1DKernel(stddev=3)

		self.star_flux_smoothed  = convolve(np.interp(self.star_wave, self.star_wave, self.star_flux), g)
		self.star_error_smoothed = convolve(np.interp(self.star_wave, self.star_wave, self.star_flux), g)




	def compute_blackbody(self):
		"""
		Compute a blackbody spectrum using the astropy BlackBody1D function

		input: self

		returns: flux of the blackbody function normalized to the mean flux within the
					spectroscopic window of the star

		"""
		from astropy.modeling.models import BlackBody1D
		from astropy.modeling.blackbody import FLAM
		from astropy import units as u

		bb = BlackBody1D(temperature=self.star_temp*u.K)
		
		bb_wav = self.star_wave * u.AA

		bb_flux = bb(bb_wav).to(FLAM, u.spectral_density(bb_wav))

		bb_flux = bb_flux / np.mean(bb_flux)

		return bb_flux


	def mask_stellar_H_lines(self):
		"""
		compare the wavelength coverage in terms of the Paschen and Brackett series
		and mask out the relevant H absorption lines in the stellar spectra. 
		
		1. check which lines are within the wavelength window

		2. fit a gaussian to the abs line/s

		3. fit a ploynomial across the abs feature masking 3*sigma for the gaussian

		Once all H lines are masked and interpolated over, the resulting spectrum will be trated
		as the 'real' telluric star for the correction purposes. 


		"""

		def mask_and_interpolate(self, line, delta ):
			"""
			Mask the specified H line and interpolate over it. 
			"""

			import copy
			from astropy.modeling import models, fitting


			# This selectes a cutout around the H line
			line_mask = [(self.star_wave>(line-delta)) & (self.star_wave<(line+delta))]
			H_line_masked_wave = self.star_wave[line_mask]
			H_line_masked_flux = self.star_flux_smoothed[line_mask]

			# fit a polynomial around the H line to remove the continuum
			# this is done by selecting the first 20 and last 20 indexes 
			# of the cutout
			pf_val_initial = np.polyfit(H_line_masked_wave[np.r_[0:20, -21:-1]],
			                    H_line_masked_flux[np.r_[0:20, -21:-1]],
	                    1 )
			pf_initial = np.polyval(pf_val_initial,H_line_masked_wave )

			# fit a gaussian
			sp = -pf_initial+H_line_masked_flux
			x = H_line_masked_wave

			g_init = models.Gaussian1D(amplitude=-1.0, 
			                           mean=line,
			                           stddev=1.)

			fit_g = fitting.LevMarLSQFitter()
			g = fit_g(g_init, x, sp)

			# select the 3-sigma window as the mask 
			# and repeat the polynomial fitting
			start = int(g.mean.value - (g.stddev.value*3))
			end = int(g.mean.value + (g.stddev.value*3))


			H_polynomial_mask = [(H_line_masked_wave<(start)) | (H_line_masked_wave>(end))]

			pf_val_best_fit = np.polyfit(H_line_masked_wave[H_polynomial_mask],
			                    H_line_masked_flux[H_polynomial_mask], 1 )

			pf_best_fit = np.polyval(pf_val_best_fit, H_line_masked_wave[~H_polynomial_mask[0]])

			# now replace the stellar flux within this range using the polynomial
			replace_mask = [(self.star_wave>start) & (self.star_wave<end)]

			self.star_flux_H_interpolated = copy.deepcopy(self.star_flux_smoothed)

			self.star_flux_H_interpolated[replace_mask]= pf_best_fit

			return x, g(x) + pf_initial


		#Paschen and Brackett series
		# https://www.gemini.edu/sciops/instruments/nearir-resources/astronomical-lines/h-lines
		H_lines = 10 * np.array([901.2,922.6,954.3,1004.6,1093.5, 1282.4, 1874.5,
		1570.7, 1588.7, 1611.5, 1641.3, 1681.3, 1736.9, 1818.1, 1945.1, 2166.1]) 


		## what lines of the abs falls within the spectral window of the tell star
		lines_within_spectra = H_lines[(np.min(self.star_wave) < H_lines) & (np.max(self.star_wave)> H_lines)]

		print("There are %s lines within the observed spectral range"%len(lines_within_spectra))
		print("We will now go through each one of them and perfom a single gaussian fit")

		
		

		gauss_waves, gauss_fluxes = [], []


		result = [mask_and_interpolate(self, line, delta) for line in lines_within_spectra]
		gauss_waves, gauss_fluxes = zip(*result)


		self.make_H_mask_spectra(len(lines_within_spectra), gauss_waves, gauss_fluxes)






	def make_H_mask_spectra(self, N_of_lines, gauss_waves, gauss_fluxes):

	    """
	    Once the H stellar H lines are masked, this function will
	    generate a figure to the user to show if the masking + interpolation
	    is successful.
	    
	    """
	   
	    def make_plot(self, ax, gauss_wave, gauss_flux):
	    
	        
	        ax.plot( self.star_wave, self.star_flux, color='k', ls='-', label='Observed spectrum')
	        
	       	ax.plot( self.star_wave, self.star_flux_smoothed, color='orange', ls='-', label='Smoothed spectrum')

	        ax.plot( gauss_wave, gauss_flux, color='r', ls='--', label='Gaussian Fit')
	        
	        ax.plot( self.star_wave, self.star_flux_H_interpolated, color='blue', 
	                ls='-', label='Interpolated spectrum')

	        ax.set_xlabel('wavelength (Angstroms)')
	        ax.set_ylabel('flux (counts)')
	        ax.set_xlim(np.min(gauss_wave)*0.95, np.max(gauss_wave)*1.05)
	        ax.set_ylim(np.min(gauss_flux)*0.5, np.max(gauss_flux)*2)
	        ax.legend()
	        
	    
	    import matplotlib.pyplot as plt
	    from matplotlib import gridspec
	    import math

	    cols = 2
	    rows = int(math.ceil(N_of_lines / cols))

	    fig = plt.figure(figsize=(8,4))
	    gs = gridspec.GridSpec(rows, cols)

	    for n in range(N_of_lines):
	        ax = fig.add_subplot(gs[n])
	        make_plot(self, ax, gauss_waves[n], gauss_fluxes[n])


	    fig.tight_layout()
	    
	    plt.savefig(out_path + 'star_H_mask_'+self.star_name+'_'+self.obs_date+'_'+self.band+'.pdf')
	    
	    plt.show()






	def fit_ploynomial(self, flux, wave, error):
		"""
		NOT USED ATM
		Fit an polynomial to the smoothed stellar spectra to average over the 
		H absorption lines (and some other stellar features). 
		The error spectrum is used as the weights.

		input: smoothed flux, wavelength, smoothed error

		returns: 

		"""
		wave_mask = [(wave>15000) & (wave<18000)]

		pf_val = np.polyfit(n2_ts_pC_wave[n2_ts_pC_mask],n2_ts_pC_flux[n2_ts_pC_mask], 6 )
		pf = np.polyval(pf_val,n2_ts_pC_wave[n2_ts_pC_mask] )
		pf_wave = n2_ts_pC_wave[n2_ts_pC_mask]


	def write_spectra_to_disk(self, corrected_star_flux,  blackbody_flux, correction):
		"""
		Write the telluric corrected spectra to disk as a csv file. 
		"""

		df = pd.DataFrame()
		df['wavelength'] 					= self.star_wave
		df['star_extracted_1d_flux'] 		= self.star_flux
		df['star_extracted_1d_flux_smooth']	= self.star_flux_smoothed
		df['star_extracted_1d_H_interpolated']	= self.star_flux_H_interpolated
		df['star_telluric_corrected']		= corrected_star_flux	
		df['blackbody_flux']				= blackbody_flux
		df['derived_sensitivity']			= correction

		df.set_index('wavelength', inplace=True)

		df.to_csv(out_path + 'sensitivity_curve_telluric_'+self.star_name+'_'+self.obs_date+'_'+self.band+'.csv')

		return 


	def make_comparison_figure(self,corrected_star_flux, blackbody_flux, correction, sky_transmission, filter_transmission):
		"""
		Make a figure for the telluric correcction routine
		"""

		## first write the derived values to disk
		self.write_spectra_to_disk(corrected_star_flux,blackbody_flux, correction)

		import matplotlib.pyplot as plt
		# plt.plot(pf_wave, pf/np.median(pf), c='b')

		fig = plt.subplots(figsize=(8,4))

				
		plt.plot(self.star_wave, self.star_flux/np.median(self.star_flux), 
			c='brown', lw=0.5, label='original stellar spectra')

		plt.plot(self.star_wave, self.star_flux_H_interpolated/np.median(self.star_flux_H_interpolated), 
			c='red', lw=0.5, label='stellar spectra H lines interpolated')

		plt.plot(self.star_wave, self.star_flux_smoothed/np.median(self.star_flux_smoothed), 
			c='b', lw=0.5, label='smoothed stellar spectra')

		plt.plot(self.star_wave, blackbody_flux, c='orange', label='blackbody function')

		plt.plot(self.star_wave, correction, c='g', lw=0.5, label='derived sensitivity')

		plt.plot(self.star_wave, corrected_star_flux/np.nanmedian(corrected_star_flux), c='magenta', 
			lw=0.5, label='Telluric corrected star')

		plt.plot(sky_transmission['wavelength(micron)']*1e4, sky_transmission['transmission'], c='k', 
			label='sky transmission')

		plt.plot(filter_transmission['wavelength']*1e4, filter_transmission['response'], c='cyan',
			label='filter transmission')

		plt.legend(loc='best', ncol=2)

		plt.xlim(self.star_wave.min() *0.99 , self.star_wave.max()*1.01)

		plt.xlabel('wavelength (A)')
		plt.ylabel('N')

		plt.axhline(y=1.0, ls='--', c='k')

		plt.savefig(out_path + 'comp_fig_'+self.star_name+'_'+self.obs_date+'_'+self.band+'.pdf')
		
		plt.show()

		return 





def cal_values(star_file, header=False):

	"""
	Extract the flux, wavelength, and error information from the DRP 1D fits files
	input: fits object: IMPORTANT: need to be the 1D extracted fits object from the DRP
	returns: flux, wavelength, error 

	"""
	print('opening ' + star_file)
	eps = fits.open(star_file)
	
	star_error_file = star_file.replace('_eps', '_sig')
	print('opening' + star_error_file)
	eps_error = fits.open(star_error_file)

	scidata, hdr_sci , sddata, hdr_err = eps[0].data, eps[0].header, eps_error[0].data, eps_error[0].header
	CRVAL1, CD1_1 , CRPIX1 = hdr_sci['CRVAL1'], hdr_sci['CD1_1'], hdr_sci['CRPIX1']
	i_w        = np.arange(len(scidata[0]))+1 #i_w should start from 1
	wavelength = ((i_w - CRPIX1) * CD1_1 ) + CRVAL1 
	
	flux = scidata; error = sddata

	if header: 
		return flux, wavelength, error, hdr_sci, hdr_err
	else:
		return flux, wavelength, error



def make_spatial_plot(data_obj, sp_x,sp_y , fit, flux, wave, error, long2pos):
		"""
		Make a spatial profile of the optimal 1D extraction
		"""

		import matplotlib.pyplot as plt

		fig, axs = plt.subplots(figsize=(6,3), nrows=1, ncols=2)


		axs[0].plot(sp_x,     sp_y , c='k' , ls='-', label='spatial profile')
		axs[0].plot(sp_x, fit(sp_x), c='r',  ls='-', label='Fit' )

		axs[0].set_xlabel('y')
		axs[0].set_ylabel('counts')
		axs[0].legend(loc='best')
		axs[0].set_title(long2pos)

		axs[1].plot(wave, flux,  c='b', label='flux')
		axs[1].plot(wave, error, c='r', label='error')

		axs[1].set_xlabel('wavelength (A)')
		axs[1].set_ylabel('counts')
		axs[1].legend(loc='best')
		axs[1].set_title(long2pos)


		plt.tight_layout()
		plt.savefig(out_path + 'sp_'+str(data_obj.star_name)+'_'+str(data_obj.obs_date)+'_'+str(data_obj.band)+long2pos+'.pdf')
		plt.show()

		return 




def extract_1d(data_obj, star_file, long2pos=None):
	"""
	Extract a 1D spectrum from a 2D spectrum. 
	Follows Horne1986 prescription. 

	"""

	from astropy.modeling import models, fitting


	flux, wave, error = cal_values(star_file)

	sp = np.nansum(flux[:, 1000:1500], axis=1)
	x = np.arange(len(sp))

	g_init = models.Gaussian1D(amplitude=1., mean=data_obj.center_of_spatial_profile, stddev=1.)
	fit_g = fitting.LevMarLSQFitter()
	g = fit_g(g_init, x, sp)

	start = int(g.mean.value - (g.stddev.value*5))
	end = int(g.mean.value + (g.stddev.value*5))

	weights = 1./(abs(g.mean.value-np.arange(start, end+1, 1))/g.stddev.value)

	print(np.nansum(weights))
	weights = weights/np.nansum(weights)
	print(weights)
	### This is what i had before
	# flux_1d  = np.nansum(flux[start:end+1,:] * weights[:, None], axis=0)
	# error_1d = np.nansum(error[start:end+1,:] ** 2, axis=0)**0.5

	flux_numerator = np.nansum(flux[start:end+1,:] * weights[:, None]/(error[start:end+1,:]**2), axis=0)

	flux_denominator = np.nansum((weights[:, None]**2)/(error[start:end+1,:]**2), axis=0)

	flux_1d  = flux_numerator/flux_denominator

	error_1d = np.nansum(1./(weights[:, None]**2)/(error[start:end+1,:]**2), axis=0)**0.5

	
	make_spatial_plot(data_obj, x, sp , g, flux_1d, wave, error_1d, long2pos)


	return flux_1d, wave, error_1d



def find_nearest(array,value):

    idx = (np.abs(array-value)).argmin()
    return idx


def compute_flux_convt_factor(star_flux, star_wave, band_center_wave, band_zero, band_mag):
	"""
	compute a conversion factor to convert eps to ers/s/A. 
	"""

	wave_index=find_nearest(star_wave, band_center_wave)

	star_flux_center = star_flux[wave_index] # bb_j is the centre of the band--> matches the centre of the band with the centre of the 2mass mag

	band_flux = band_zero*(10**(band_mag/(-2.5))) # convert the 2mass mag to abs flux, erg/s/cm^2/A

	gain = 1.0

	exposure_star = 1.0

	fluxconvt=(exposure_star/gain)*band_flux/star_flux_center # the conversion factor 

	return fluxconvt

def apply_calibs_to_data(eps_fluxes, tel_sens_wave, tel_sens_curve, flux_conversion_factor):
	"""
	Apply the telluric corrections and flux calibrations to the observed 2D spectra. 
	This is done in a single step for now. 
	"""

	for index, files  in enumerate(eps_fluxes):

			flux, wave, error, header_flux, header_error = cal_values(eps_fluxes[index], header=True)

			tel_sens_curve_interp = np.interp(wave, tel_sens_wave, tel_sens_curve)

			flux  = flux  * tel_sens_curve_interp * flux_conversion_factor

			error = error * tel_sens_curve_interp * flux_conversion_factor


			primary_hdu = fits.PrimaryHDU(flux, header=header_flux)
			new_hdul = fits.HDUList([primary_hdu])
			new_hdul.writeto(eps_fluxes[index].replace('.fits', '_tc_fc.fits'))
			new_hdul.close()

			eps_error = eps_fluxes[index].replace('_eps', '_sig')

			primary_hdu = fits.PrimaryHDU(error, header=header_error)
			new_hdul = fits.HDUList([primary_hdu])
			new_hdul.writeto(eps_error.replace('.fits', '_tc_fc.fits'))
			new_hdul.close()

	return 



def main():

	"""
	1. load the 2D extracted spectra and extract 1D spectra of the standard
		1.1 if long2pos combine the positions to generate a single 1d spectra
	2. smooth the spectra 
	3. fit a ploynomial around H absorption lines to remove strong absorption intrinsic to the star
	4. generate a blackbody function to the star
	5. generate a sensitivity correction
	5.1 apply sensitivity curve to telluric star
	6. generate flux conversion factor for standard star
	6.1 apply the flux calibration to standard star
	7. apply telluric corrections + flux calibrations to the data


	"""

	global out_path, band_center_wave, band_fwhm, delta

	######USER INPUTS#######
	band ='K'
	date ='2018nov29'
	star_name='HD201941'
	star_temperature = 5000 #The temperature of the star in K (average AoV star T=9600K)

	csp = 100 #centre of the spatial profile of the positive image (used as the first guess for the gaussian)
	
	# this is the width used for H masking of the star
	# if you are not happy with the continnum this value should be changed as required
	# the main issue is the balance when there is a H line abs close to a telluric line
	delta=75 

	magJ = 6.678
	magH = 6.673
	magK = 6.602

	out_path  = '../outputs/' #where the outputs will be written to 
	sky_transmission = ascii.read('/Users/themiya/Dropbox/mosdrp/analysis/Tables/instrument_throughputs/trans_16_15.dat')
	### night2 H band Telluric star
	path_A = '../reduced/long2pos/'+str(date)+'/'+str(band)+'/'+str(star_name)+'_POSA_NARROW_'+str(band)+'_POSA_eps.fits'
	path_C = '../reduced/long2pos/'+str(date)+'/'+str(band)+'/'+str(star_name)+'_POSC_NARROW_'+str(band)+'_POSC_eps.fits'

	twod_flux_names  = glob.glob('../reduced/H_band_COSMOS/2019apr25/H/H_band_COSMOS_H_*_eps.fits')
	# twod_error_names = glob.glob('../reduced/H_band_COSMOS/2019apr25/H/H_band_COSMOS_H_*_sig.fits')

	j_zero = 3.129e-09*1e7/1e4/1e4     # convert from W/m2/um -> erg/cm2/A
	j_wave = 1.2350*1e4                # 1.25 um to Ang, of effective J-band wavelength
	j_fwhm = 2000

	h_zero = 1.133e-09*1e7/1e4/1e4     # convert from W/m2/um -> erg/cm2/A
	h_wave = 1.662*1e4                # 1.25 um to Ang, of effective J-band wavelength
	h_fwhm = 3410

	k_zero = 0.4283e-09*1e7/1e4/1e4     # convert from W/m2/um -> erg/cm2/A
	k_wave = 2.159*1e4    # 1.25 um to Ang, of effective J-band wavelength
	k_fwhm = 4830            

	if band =='J':
		band_mag         = magJ
		band_center_wave = j_wave
		band_zero        = j_zero
		band_fwhm        = j_fwhm
		mosfire_tp 		 = ascii.read('/Users/themiya/Dropbox/mosdrp/analysis/Tables/instrument_throughputs/MOSFIRE/J.dat')


	elif band =='H':
		band_mag         = magH
		band_center_wave = h_wave
		band_zero        = h_zero
		band_fwhm        = j_fwhm
		mosfire_tp 		 = ascii.read('/Users/themiya/Dropbox/mosdrp/analysis/Tables/instrument_throughputs/MOSFIRE/H.dat')


	elif band =='K':
		band_mag         = magK
		band_center_wave = k_wave
		band_zero        = k_zero
		band_fwhm        = j_fwhm
		mosfire_tp 		 = ascii.read('/Users/themiya/Dropbox/mosdrp/analysis/Tables/instrument_throughputs/MOSFIRE/K.dat')


	####STEP 1####################
	telluric_star = generate_telluric(band, star_name, path_A, path_C, star_temperature, date, csp )


	###STEP 2####################

	telluric_star.smooth_stellar_spectra()

	# ###STEP 3####################
	# ### This is not correct 

	telluric_star.mask_stellar_H_lines()

	# ###STEP 4####################

	blackbody_flux = telluric_star.compute_blackbody()

	# ###STEP 5 ##################

    ## where should this normalization go? 
	correction = (telluric_star.star_flux_H_interpolated/np.median(telluric_star.star_flux_H_interpolated))/blackbody_flux

	# correction = (telluric_star_smoothed_flux/np.median(telluric_star_smoothed_flux))/blackbody_flux

	# ascii.write( {'wavelength': telluric_star.star_wave, 'sensitivity':correction } , 
	#   '../outputs/tel_corr_function'+str(star_name)+'_'+str(date)+'_'+str(band)+'.dat' 	)

	corrected_star_flux = telluric_star.star_flux /correction


	telluric_star.make_comparison_figure(corrected_star_flux, blackbody_flux, correction, sky_transmission, mosfire_tp)
	
	# ###STEP 6 ##################

	
	flux_convt_factor = compute_flux_convt_factor(telluric_star.star_flux, telluric_star.star_wave, 
		band_center_wave, band_zero, band_mag)

	flux_calibrated_star = corrected_star_flux * flux_convt_factor


	# ###STEP 8 ##################

	# apply_calibs_to_data(twod_flux_names, telluric_star.star_wave, 
	# 	correction, flux_convt_factor)


if __name__ == '__main__':
    main()













