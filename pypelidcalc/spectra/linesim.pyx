#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

import os
import numpy as np
cimport numpy as np

from cython.view cimport array as cvarray

from libc cimport math
cimport cython_gsl as gsl

import time

import logging

from pypelidcalc.cutils cimport histogram as H
from pypelidcalc.cutils cimport rng

from pypelidcalc.utils import consts

from . galaxy cimport Galaxy, galaxy_struct


cdef class LineSimulator:
	""" Simulate emission line spectra.

	Parameters
	----------
	optics : pypelidcalc.instrument.Optics
		Optics instance
	extraction_window : int
		the full extraction window is :math:`2 w + 1`
	"""

	def __init__(self,
				optics,
				extraction_window=3,
				extraction_sigma=2.0,
				extraction_weights=True,
				photon_shoot_limit=100000,
				isotropize=True,
				seed=None):
		""" """
		cdef int i

		self.dispersion = optics.config['pix_disp'][1]  # A/pixel
		self.plate_scale = optics.config['pix_size'] # arcsec/pixel
		self.lambda_min, self.lambda_max = optics.config['lambda_range']

		self.npix = int(np.round((self.lambda_max - self.lambda_min) / self.dispersion))
		self.lambda_max = self.lambda_min + self.npix * self.dispersion

		self.extraction_window = <int> math.round(extraction_window * extraction_sigma)

		self.photon_shoot_limit = photon_shoot_limit

		self.isotropize = int(isotropize)

		self.binsx = np.arange(self.npix + 1) - 0.5
		self.binsy = np.arange(-self.extraction_window, self.extraction_window + 2) - 0.5

		self.init_extraction_weights(extraction_weights, extraction_sigma)

		self.nx = self.binsx.shape[0]-1

		self.PSF = optics.PSF
		self.transmission_func = optics.grism_transmission[1]

		self.c_kms = consts.c_kms # speed of light in km/s

		self.image = cvarray(shape=(self.nx,), itemsize=sizeof(double), format='d')
		self.image_tmp = cvarray(shape=(self.nx,), itemsize=sizeof(double), format='d')
		self.image_noisy = cvarray(shape=(self.nx,), itemsize=sizeof(double), format='d')
		self.noise_image = cvarray(shape=(self.nx,), itemsize=sizeof(double), format='d')

		for i in range(self.nx):
			self.image[i] = 0
			self.noise_image[i] = 0

		self.init_noise_template()

	def init_extraction_weights(self, extraction_weights, extraction_sigma):
		""" """
		cdef double norm

		bin_c = np.arange(-self.extraction_window, self.extraction_window + 1)

		# logging.info("extraction window %i with weights %s", len(bin_c), str(extraction_weights))

		if extraction_weights:
			self.extraction_weights = np.exp(-0.5 * bin_c**2 / extraction_sigma**2)
		else:
			self.extraction_weights = np.ones(len(bin_c))

		assert self.extraction_weights.shape[0] == 2 * self.extraction_window + 1

		norm = 0
		for i in range(self.extraction_weights.shape[0]):
			norm += self.extraction_weights[i] * self.extraction_weights[i]
		self.extraction_norm = norm


	cdef void init_noise_template(self):
		""" """
		cdef int i
		cdef double wavelength, t
		cdef double [:] noise_template

		noise_template = cvarray(shape=(self.nx,), itemsize=sizeof(double), format='d')

		for i in range(self.nx):
			wavelength = self.pixel_to_wavelength(i)
			t = self.transmission_func.scalar(wavelength)
			if t > 0:
				noise_template[i] = 1. / (t * t * wavelength * wavelength)
			else:
				noise_template[i] = 0.

		self.noise_template = noise_template

	cdef int check_outside_extraction_window(self, double y) nogil:
		""" """
		if y > self.extraction_window:
			return 1
		if -y > self.extraction_window:
			return 1
		return 0

	cdef double get_extraction_weight(self, double y) nogil:
		""" """
		cdef int i

		i = <int>(y + self.extraction_window + 0.5)

		if i < 0:
			i = 0

		if i > self.extraction_window*2:
			i = <int>self.extraction_window*2

		return self.extraction_weights[i]

	cpdef double render_line(self, Galaxy g, double loc_x, double loc_y, double velocity_disp, long n, double [:] image):
		""" Render an emission line.

		Parameters
		----------
		galaxy : Galaxy
			Galaxy instance
		loc_x : float
		    Location of emission line in dispersion direction (pixel)
		loc_y : float
		    Location of emission line in perpendicular direction (pixel)
		velocity_disp : float
		    Dispersion (1-sigma Guassian) to add to the line (pixel)
		n : int
		    Number of photons to synthesize
		image : numpy.ndarray
		    Output 1 dimensional spectrum

		Returns
		-------
		photon count : float
		    number of photons that were synthesized in the image weighted by the extraction weight.
		"""
		cdef int i, bin, nx
		cdef double bin_start, bin_step, x, y, dx_vel, count, w, bulge_scale_pix, disk_scale_pix
		cdef double [:,:] xy
		cdef double [:,:] psf_xy
		cdef double [:] binsx = self.binsx
		cdef double plate_scale = self.plate_scale
		cdef int isotropize = self.isotropize
		cdef galaxy_struct * gal = g.gal

		if n <= 0:
			return 0

		bin_start = binsx[0]
		nx = binsx.shape[0]-1
		bin_step = binsx[1] - binsx[0]

		dx_vel = 0

		# logging.info("n %g platescale %g", n, plate_scale)
		xy = g.sample(n, plate_scale, isotropize)

		psf_xy = self.PSF.sample(n)

		count = 0.
		cdef double count_outside = 0

		# logging.info("nphot %i", n)

		with nogil:
			for i in range(n):
				y = xy[i, 1] + psf_xy[i, 1] + loc_y

				if self.check_outside_extraction_window(y) == 1:
					count_outside += 1
					continue

				if velocity_disp > 0:
					dx_vel = rng.rng.gaussian(velocity_disp)

				x = loc_x + xy[i, 0] + psf_xy[i, 0] + dx_vel

				bin = H.histogram_bin(x, bin_start, bin_step)
				if bin < 0: continue
				if bin >= nx: continue

				w = self.get_extraction_weight(y)

				image[bin] += w
				count += w

		# logging.info("counts %g outside %g", count, count_outside)

		return count

	cpdef void make_noise_spectrum(self, Galaxy g, double scale, double[:] noise_out, int window=10):
		""" Render the background noise spectrum.  The noise variance estimate at the location of each
		emission line is used to extrapolate the noise over the full spectrum.
		The extrapolation is based on the template self.noise_template.

		Parameters
		----------
		g : Galaxy
		    Galaxy instance
		scale : float
		    value to scale flux by
		noise_out : numpy.ndarray
		    output array
		window : int
		    Size of window around emission lines (pixels)

		Returns
		-------
		None
		"""
		cdef int i, n, j, k, pix
		cdef double amp, norm
		cdef double [:] noise
		cdef double [:] noise_template
		cdef int [:] noise_count
		cdef galaxy_struct * gal = g.gal

		noise_template = self.noise_template
		noise = self.image_tmp
		noise_count = cvarray(shape=(self.nx,), itemsize=sizeof(int), format='i')

		with nogil:

			for i in range(self.nx):
				noise[i] = 0
				noise_count[i] = 0

			n = gal.line_count

			amp = 0
			norm = 0
			for i in range(n):
				line_var = gal.emission_line[i].background
				# with gil:
					# logging.info("line bg %g %g", line_var, line_var*scale)
				if line_var > 0:
					pix = <int>self.wavelength_to_pixel(gal.emission_line[i].wavelength_obs)
					if pix < 0: continue
					if pix >= self.nx: continue
					amp += line_var*noise_template[pix]
					norm += noise_template[pix]*noise_template[pix]

					for j in range(2 * window + 1):
						k = pix + j - window
						if k < 0: continue
						if k >= self.nx: continue
						noise[k] += line_var * scale
						noise_count[k] += 1

			for i in range(self.nx):
				if noise_count[i] > 0:
					noise_out[i] += noise[i] / noise_count[i]
				noise[i] = 0

			if norm > 0:
				amp /= norm
			else:
				amp = 1.0

			# with gil:
				# logging.info("amp %g", amp)

			for i in range(self.nx):
				if noise_count[i] == 0:
					noise_out[i] += amp * noise_template[i] * scale

		return

	cdef double compute_snr(self, double [:] signal, double [:] var) nogil:
		""" Compute the signal-to-noise ratio from the 1D spectrum and variance.

		Parameters
		----------
		signal : ndarray
		  signal vector
		var : ndarray
		  variance vector

		Returns
		-------
		snr : float
		  signal-to-noise ratio
		"""
		cdef int i, nx
		cdef double snr

		nx = signal.shape[0]

		snr = 0
		for i in range(nx):
			if var[i] > 0:
				snr += (signal[i] * signal[i]) / var[i]

		if snr > 0:
			snr = math.sqrt(snr)

		return snr

	cpdef sample_spectrum(self, Galaxy g):
		"""
		"""
		cdef int line_i, flag, i
		cdef double x0, scale, sigma_vel, velocity_disp_pix, nphot, nbg

		cdef galaxy_struct * gal = g.gal

		cdef double[:] image = self.image
		cdef double[:] image_tmp = self.image_tmp
		cdef double[:] image_noisy = self.image_noisy
		cdef double[:] noise_image = self.noise_image

		for i in range(self.nx):
			image[i] = 0
			image_tmp[i] = 0
			noise_image[i] = 0

		velocity_disp_pix = gal.velocity_disp / self.c_kms / self.dispersion

		for line_i in range(gal.line_count):

			if gal.emission_line[line_i].flux <= 0:
				continue

			if gal.emission_line[line_i].variance <= 0:
				continue

			nphot = gal.emission_line[line_i].flux**2 / gal.emission_line[line_i].variance
			scale = nphot / gal.emission_line[line_i].flux

			# scale converts between counts and flux units
			# here is the derivation such that the poisson variance of nphot
			# is equal to the line variance.
			# ------------------------------------
			# var(nphot / scale) = nphot / scale^2
			#                    = nphot / (nphot^2 / flux^2)
			#                    = flux^2 / nphot
			#                    = flux^2 / (flux^2 / var)
			#                    = var

			if (self.photon_shoot_limit > 0) and (nphot > self.photon_shoot_limit):
				# Put a limit on the number of photons.  This sets a ceiling on SNR which depends on the profile.
				# Recompute the scale
				scale = self.photon_shoot_limit / gal.emission_line[line_i].flux
				nphot = self.photon_shoot_limit + rng.rng.gaussian(self.photon_shoot_limit/math.sqrt(nphot))
			else:
				# Poisson sample
				# If N>~100 use a Gaussian distribution instead of Poisson
				if nphot < POISSON_SAMPLE_LIMIT:
					nphot = <double> rng.rng.poisson(nphot)
				else:
					nphot += rng.rng.gaussian(math.sqrt(nphot))

			if nphot <= 0:
				continue

			x0 = self.wavelength_to_pixel(gal.emission_line[line_i].wavelength_obs) + gal.offx

			if x0 < 0:
				continue

			if x0 > self.npix:
				continue

			sigma_vel = gal.emission_line[line_i].wavelength_obs * velocity_disp_pix

			self.render_line(g, x0, gal.offy, sigma_vel, <long> nphot, image_tmp)

			for i in range(self.nx):
				image[i] += image_tmp[i] / scale  # to convert to flux units
				image_tmp[i] = 0

		self.make_noise_spectrum(g, self.extraction_norm, noise_image)

		self.sample_gaussian_noise(image, noise_image, image_noisy)

		return image_noisy, image, noise_image

	cpdef sample_noise(self, Galaxy g):
		""" """
		cdef int i
		cdef galaxy_struct * gal = g.gal

		cdef double[:] image = self.image
		cdef double[:] noise_image = self.noise_image

		for i in range(self.nx):
			image[i] = 0
			noise_image[i] = 0

		self.make_noise_spectrum(g, 1.0, noise_image)
		self.sample_gaussian_noise(image, noise_image, image)

		return image, noise_image

	cdef void sample_gaussian_noise(self, double [:] image, double [:] var, double [:] image_out) nogil:
		""" """
		cdef int i, nx
		nx = image.shape[0]
		for i in range(nx):
			if var[i] > 0:
				image_out[i] = image[i] + rng.rng.gaussian(math.sqrt(var[i]))
			else:
				image_out[i] = image[i]

	cdef double wavelength_to_pixel(self, double wave) nogil:
		""" Convert observed wavelength to pixel coordinate in the dispersion direction.

		Parameters
		----------
		wave : float
			observed wavelength

		Returns
		-------
		nearest pixel coordinate : float
		"""
		return (wave - self.lambda_min) / self.dispersion

	cdef double pixel_to_wavelength(self, double pix) nogil:
		""" Convert pixel coordinate in the dispersion direction to observed wavelength.

		Parameters
		----------
		wave : float
			observed wavelength

		Returns
		-------
		nearest pixel coordinate : float
		"""
		return (<double>pix) * self.dispersion + self.lambda_min

