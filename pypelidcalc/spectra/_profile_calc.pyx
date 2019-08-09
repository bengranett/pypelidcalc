#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

""" The two-dimensional galaxy image is defined by the profile, axis ratio and position angle.

	Notes
	-----
	The profile is given by :math:`I(r)`. The integrated profile is

	.. math:: F(r) = \int_0^r I(r') 2\pi r' dr'


	With :math:`F` normalized to 1, the half-light radius :math:`r_{half}` is defined by

	.. math:: F(r_{half})=\\frac{1}{2}
"""
import time
import logging

import numpy as np
cimport numpy as np

from libc cimport math
from cython.view cimport array as cvarray

cimport cython_gsl as gsl

from pypelidcalc.cutils cimport interpolate

from pypelidcalc.utils import consts


DEF DV_COEF=7.67
DEF DV_INDX=0.25
DEF DV_NORM=7.67**8/40320

cpdef double disk_profile(double r, double scale) nogil:
	"""
	"""
	cdef double x

	if scale <= 0:
		return 0
	x = r * 1./scale
	return 1./(2 * math.M_PI * scale * scale) * math.exp(-x)


cdef double _disk_integrated(double r, double scale, double *deriv) nogil:
	""" Cumulative integral of exponential disk profile

	Notes
	-----
	The integrated profile is

	.. math::
	   F(x) = 1 - \left(1+x\\right)e^{-x}

	Parameters
	----------
	r : numpy.ndarray
		radius coordinate
	scale : float
		scale length

	Returns
	-------
	double:
		profile
	"""
	cdef double x, e

	if scale <= 0:
		return 0
	x = r * 1./ scale

	e = math.exp(-x)

	if deriv:
		deriv[0] = 1./scale * x * e

	return 1 - (1 + x) * e


def disk_integrated(r, scale):
	return _disk_integrated(r, scale, NULL)


cpdef double bulge_profile(double r, double scale) nogil:
	""" """
	cdef double x, norm

	if scale <= 0:
		return 0

	x = DV_COEF * math.pow(r * 1./scale, DV_INDX)

	norm = DV_NORM / math.M_PI / (scale * scale)

	return norm * math.exp(-x)


cdef double _bulge_integrated(double r, double scale, double *deriv) nogil:
	""" Cumulative integral of De Vaucouleurs profile

	Notes
	-----
	The integrated profile is

	.. math::
	   F(x) = 1 - e^{-x} \left(1 + \sum_{i=1}^{7} \\frac{x^{i}}{i!} \\right)

	Parameters
	----------
	r : numpy.ndarray
		radius coordinate
	scale : float
		scale length

	Returns
	-------
	numpy.ndarray:
		profile

	"""
	cdef double x, sum, e
	if scale <= 0:
		return 0

	x = DV_COEF * math.pow(r / scale, DV_INDX)

	sum = 1 + x                      \
			+ 1./2 * x*x             \
			+ 1./6 * x*x*x           \
			+ 1./24 * x*x*x*x        \
			+ 1./120 * x*x*x*x*x     \
			+ 1./720 * x*x*x*x*x*x   \
			+ 1./5040 * x*x*x*x*x*x*x # pow 7

	d = 1./5040 * x*x*x*x*x*x*x

	e = math.exp(-x)

	if deriv:
		deriv[0] = d * e * x / r * DV_INDX

	return 1 - e * sum


def bulge_integrated(r, scale):
	return _bulge_integrated(r, scale, NULL)


cdef class BaseProfile:
	"""
	"""

	def __init__(self, seed=0, res=10., rmax=100., prec=1e-4):
		""" """
		assert rmax > 0
		assert rmax < 1000
		assert res >= 1
		assert res < 1000
		assert prec > 0
		assert prec < 1

		self.res = res
		self.step = 1./res
		assert self.step > 0
		assert self.step <= 1
		self.rmax = rmax
		self.prec = prec

		self.setup_interp()

	def __cinit__(self, *args, **kwargs):
		""" """
		cdef unsigned long int seed = kwargs['seed']
		# logging.info("seed: %i", seed)
		self.rng = gsl.gsl_rng_alloc(gsl.gsl_rng_taus)
		# logging.info("rng: %s",gsl.gsl_rng_name(self.rng))
		gsl.gsl_rng_set(self.rng, seed)

	def __deallocate__(self):
		""" """
		gsl.gsl_rng_free(self.rng)

	def __str__(self):
		""" """
		return "<{} : rmax={}, res={}, step={}>".format(
			self.__class__.__name__,
			self.rmax,
			self.res,
			self.step
		)

	cdef void setup_interp(self):
		""" """
		cdef int i, n, flag,count
		cdef double tol
		cdef double [:] x
		cdef double [:] y

		n = <int>(self.res * self.rmax) + 1
		x = cvarray(shape=(n,), itemsize=sizeof(double), format='d')
		y = cvarray(shape=(n,), itemsize=sizeof(double), format='d')

		with nogil:
			x[0] = 0
			y[0] = 0

			tol = 1 - self.prec
			flag = 0
			count = 0
			for i in range(1, n):
				x[i] = x[i-1] + self.step
				y[i] = self.integrated(x[i], 1.0)

				if y[i] > tol:
					count = i
					flag = 1
					break
				if y[i] <= y[i-1]:
					flag = 2
					count = i - 1
					break

		if flag:
			x = x[:count+1]
			y = y[:count+1]
			n = x.shape[0]

		assert y[n-1] <= 1

		if y[n-1] < tol:
			logging.warning("Profile '%s' does not converge to 1: at rmax=%f, y=%f, but should be 1.  You can increase rmax.",self.__class__.__name__, x[n-1], y[n-1])

		self.radius_interp = interpolate.interp1d(y, x, fill_low=0, fill_high=x[n-1])

	cdef double integrated(self, double r, double scale) nogil:
		return 0

	cdef double profile(self, double r, double scale) nogil:
		return 0

	cpdef double radius(self, double scale, double axis_ratio=1, double frac=0.5):
		""" """
		return self.radius_interp.evaluate(frac) * scale

	cpdef double projected_radius(self, double scale, double axis_ratio=1, double frac=0.5):
		""" """
		if frac != 0.5:
			logging.warning("Projected light radius is only calibrated at the half-light point.  You asked for the radius containing %f.", frac)
		return self.radius(scale, axis_ratio, frac) * consts.PROJECTED_HALF_LIGHT_RADIUS

	cpdef double[:,:] sample(self, double scale=1, double axis_ratio=1, double pa=0, int n=1000000):
		""" Draw samples x,y from the galaxy image.

		Parameters
		----------

		n : int
			number of samples to draw
		return_xy : bool
			Return x,y samples or only radius

		Returns
		-------
		x : numpy.ndarray
		y : numpy.ndarray
		"""
		cdef int i, rotate
		# cdef double [:,:] x
		cdef double r, e, theta, theta0, costh,sinth, x_temp, root_axis_ratio

		cdef double [:,:] x = cvarray(shape=(n, 2), itemsize=sizeof(double), format='d')

		if axis_ratio <= 0 or scale <= 0:
			return np.array(x)

		rotate = 0
		if pa != 0:
			rotate = 1
			theta0 = math.M_PI/180 * pa
			costh = math.cos(theta0)
			sinth = math.sin(theta0)

		root_axis_ratio = math.sqrt(axis_ratio)

		with nogil:
			for i in range(n):

				e = gsl.gsl_rng_uniform(self.rng)
				r = self.radius_interp.evaluate(e) * scale

				theta = gsl.gsl_rng_uniform(self.rng) * 2 * math.M_PI
				x[i, 0] = r * root_axis_ratio * math.cos(theta)
				x[i, 1] = r / root_axis_ratio * math.sin(theta)

				if rotate:
					x_temp = x[i,0] * costh + x[i,1] * sinth
					x[i,1] = - x[i,0] * sinth + x[i,1] * costh
					x[i,0] = x_temp

		return np.array(x)



cdef class _Disk(BaseProfile):

	cdef double profile(self, double r, double scale) nogil:
		return disk_profile(r, scale)

	cdef double integrated(self, double r, double scale) nogil:
		return _disk_integrated(r, scale, NULL)


cdef class _Bulge(BaseProfile):

	cdef double profile(self, double r, double scale) nogil:
		return bulge_profile(r, scale)

	cdef double integrated(self, double r, double scale) nogil:
		return _bulge_integrated(r, scale, NULL)


# Create singleton instances

_bulge = None
_disk = None

def Bulge(seed=None, **kwargs):
	global _bulge
	if _bulge is None:
		if seed is None:
			seed = time.time() * 1000000
		_bulge = _Bulge(seed=seed, **kwargs)
	return _bulge

def Disk(seed=None, **kwargs):
	global _disk
	if _disk is None:
		if seed is None:
			seed = time.time() * 1000000
		_disk = _Disk(seed=seed, **kwargs)
	return _disk
