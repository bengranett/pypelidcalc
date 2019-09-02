#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

""" The two-dimensional galaxy image is defined by the profile, axis ratio and position angle.

	Notes
	-----
	The profile is given by :math:`I(r)`. The integrated profile is

	.. math:: F(r) = \int_0^r I(r') 2\pi r' dr'


	With :math:`F` normalized to 1, the half-light radius :math:`r_{half}` is defined by

	.. math:: F(r_{half})=\\frac{1}{2}
"""
import numpy as np
cimport numpy as np

from libc cimport math
from cython.view cimport array as cvarray

cimport cython_gsl as gsl

from pypelidcalc.cutils cimport interpolate

cpdef double disk_profile(double r, double scale) nogil
cdef double _disk_integrated(double r, double scale, double *deriv) nogil

cpdef double bulge_profile(double r, double scale) nogil
cdef double _bulge_integrated(double r, double scale, double *deriv) nogil


cdef class BaseProfile:

	cdef double res, rmax, step, prec
	cdef int logscale
	cdef interpolate.interp1d radius_interp

	cdef void setup_interp(self)
	cdef double integrated(self, double r, double scale) nogil
	cdef double profile(self, double r, double scale) nogil
	cpdef double radius(self, double scale, double axis_ratio=*, double frac=*)
	cpdef double projected_radius(self, double scale, double axis_ratio=*, double frac=*)
	cpdef double [:,:] sample(self, double scale=*, double axis_ratio=*, double pa=*, int n=*, int isotropize=*)


cdef class _Disk(BaseProfile):
	cdef double integrated(self, double r, double scale) nogil
	cdef double profile(self, double r, double scale) nogil


cdef class _Bulge(BaseProfile):
	cdef double integrated(self, double r, double scale) nogil
	cdef double profile(self, double r, double scale) nogil
