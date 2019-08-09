""" This is a list of constants to use as a python module.

	Markovic, June, 2015
"""
import numpy as np
import scipy.special as special

c = 2.99792458e18       #: speed of light (angstrom/sec)

c_kms = 2.99792458e5	#: speed of light (km/sec)

planck = 6.6260693E-27  #: Planck constant (erg sec)


ABzp = 48.6        #: AB magnitude zeropoint 


halpha = 6564.614  #: Angstrom in vacuum! (6562.801 Angstrom in the air)
o3 = 4984.268      #: Mean wavelength of blended OIII doublet at 4960.295 & 5008.240

angstrom_to_micron = 1e-4 #: convert angstrom to micron
micron_to_angstrom = 1e4  #: convert micron to angstrom

fwhm_to_sigma = 1./(2*(2*np.log(2))**0.5) #: conversion factor from full-width at half-max to sigma for a Gaussian
frac_one_sigma = 1 - 2 * special.erfc(1)  #: Gaussian p-value corresponding to 1-sigma

gauss1d_norm = 1./np.sqrt(2*np.pi)

#: conversion from half-light radius to sigma for Gaussian 2D profile
RADIUS_TO_SIGMA = 1./np.sqrt(2*np.log(2))

#: conversion factor from 2D half-light radius to 1D projected half-light radius
PROJECTED_HALF_LIGHT_RADIUS = 0.54

#: minimum effective FWHM when binning onto a pixel grid.  Corresponds to sigma=1/12
BINNING_MIN_FWHM = 0.2

nan = float('nan') #: nan

FULLCIRCLE = 360.0 #: In degrees because assume RA & Dec in survey file are in degrees.
ARCSEC_TO_DEG = 1./3600. #: arcsec to degree
DEG_TO_ARCSEC = 3600. #: degree to arcsec

#: emission line list (vacuum wavelengths in angstroms)
line_list = {
	'O2': 3727.092,
	'Ha': 6564.61,
	'N2a': 6549.86,
	'N2b': 6585.27,
	'S2': 6718.29,   # for backward compatability
	'S2a': 6718.29,
	'S2b': 6732.67,
	'Hb': 4862.68,
	'O3a': 4960.295,
	'O3b': 5008.240
}

#: effective wavelengths of photometric bands in angstroms
bandpasses = {
	'mh': 16300.,
	'mj': 12200.,
}


