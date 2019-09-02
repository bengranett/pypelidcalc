#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True


import numpy as np
cimport numpy as np

from libc cimport math
from cython.view cimport array as cvarray

cimport cython_gsl as gsl



cdef class RNG:
    cdef gsl.gsl_rng * _rng

    cdef void seed(self, unsigned long int seed) nogil
    cdef double uniform(self) nogil
    cdef double gaussian(self, double sigma) nogil
    cdef unsigned int poisson(self, double n) nogil

cpdef RNG rng

