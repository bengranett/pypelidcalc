import logging

import numpy as np
cimport numpy as np

from libc cimport math
cimport cython_gsl as gsl
from cython.view cimport array as cvarray

from pypelidcalc.cutils cimport interpolate, rng

cdef class PSF_model:
    """ Point spread function model """

    cdef int is_null
    cdef double amp, var1, var2
    cdef double range_max, step

    cdef interpolate.interp1d _integ_prof

    cdef double prof_scalar(self, double r) nogil
    cpdef double[:] prof(self, double[:] r)
    cdef double evaluate_scalar(self, double r) nogil
    cpdef double[:] evaluate(self, double[:] r)
    cdef void _init_model(self)
    cpdef double[:,:] sample(self, int n)
    cpdef double[:,:] sample(self, int n)
    cpdef double radius(self, double frac=?)