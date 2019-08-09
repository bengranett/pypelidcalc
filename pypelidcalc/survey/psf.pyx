#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

import logging

import numpy as np
cimport numpy as np

import time

from libc cimport math
cimport cython_gsl as gsl
from cython.view cimport array as cvarray

from pypelidcalc.cutils cimport interpolate


cdef class PSF_model:
    """ Point spread function model """

    def __init__(self, double amp, double scale1, double scale2, double range_max=3.0, double step=0.1, seed=None):
        """ """
        cdef double smax

        self.amp = amp
        self.is_null = 1

        self.var1 = 0.
        self.var2 = 0.

        if scale1 > 0:
            self.var1 = 1./(2 * scale1 * scale1)
            self.is_null = 0

        if scale2 > 0:
            self.var2 = 1./(2 * scale2 * scale2)
            self.is_null = 0

        if scale1 > scale2:
            smax = scale1
        else:
            smax = scale2

        self.range_max = range_max * smax
        self.step = step * smax
        self._init_model()

    def __cinit__(self, *args, **kwargs):
        """ """
        cdef unsigned long int seed
        if kwargs['seed'] is not None:
            seed = kwargs['seed']
        else:
            seed = time.time()
        # logging.info("seed: %i", seed)
        self.rng = gsl.gsl_rng_alloc(gsl.gsl_rng_taus)
        # logging.info("rng: %s",gsl.gsl_rng_name(self.rng))
        gsl.gsl_rng_set(self.rng, seed)

    def __deallocate__(self):
        """ """
        gsl.gsl_rng_free(self.rng)

    cdef double prof_scalar(self, double r) nogil:
        """ """
        return (self.amp*self.var1*math.exp(-r*r*self.var1) + (1-self.amp)*self.var2*math.exp(-r*r*self.var2))/math.M_PI

    cpdef double[:] prof(self, double[:] r):
        """ """
        cdef int i, n
        cdef double[:] profile

        n = r.shape[0]
        profile = cvarray(shape=(n,), itemsize=sizeof(double), format='d')

        if self.is_null:
            return profile

        with nogil:
            for i in range(n):
                profile[i] = self.prof_scalar(r[i])

        return profile

    cdef double evaluate_scalar(self, double r) nogil:
        """ """
        return self.amp * (1 - math.exp(-r * r * self.var1)) + (1 - self.amp) * (1 - math.exp(-r * r * self.var2))

    cpdef double[:] evaluate(self, double[:] r):
        """ Evalutate integrated profile. """
        cdef int i, n
        cdef double[:] profile

        n = r.shape[0]
        profile = cvarray(shape=(n,), itemsize=sizeof(double), format='d')

        if self.is_null:
            return profile

        with nogil:
            for i in range(n):
                profile[i] = self.evaluate_scalar(r[i])

        return profile

    cdef void _init_model(self):
        """ """
        cdef int i, n
        cdef double [:] r
        cdef double [:] y

        if self.is_null:
            return

        assert self.step > 0

        n = <int>(self.range_max / self.step)

        r = cvarray(shape=(n,), itemsize=sizeof(double), format='d')
        y = cvarray(shape=(n,), itemsize=sizeof(double), format='d')

        with nogil:
            r[0] = 0
            y[0] = 0
            for i in range(1, n):
                r[i] = i * self.step
                y[i] = self.evaluate_scalar(r[i])

        logging.debug("integ prof r=%f: %f, r=%f: %f", r[0], y[0], r[n-1], y[n-1])
        self._integ_prof = interpolate.interp1d(y, r, fill_low=0, fill_high=1)

    cpdef double[:,:] sample(self, int n):
        """ Draw samples x,y from PSF

        Parameters
        ----------
        n : int
            number of samples to draw

        Returns
        -------
        x : numpy.ndarray
        y : numpy.ndarray
        """
        cdef int i
        cdef double r, e, theta

        cdef double [:,:] x = cvarray(shape=(n, 2), itemsize=sizeof(double), format='d')

        if self.is_null:
            return x

        with nogil:
            for i in range(n):

                e = gsl.gsl_rng_uniform(self.rng)
                r = self._integ_prof.evaluate(e)

                theta = gsl.gsl_rng_uniform(self.rng) * 2 * math.M_PI
                x[i, 0] = r * math.cos(theta)
                x[i, 1] = r * math.sin(theta)

        return x


    cpdef double radius(self, double frac=0.5):
        """ Compute radius that contains fraction of total flux.
        For example, set frac=0.5 to get the half-light radius.

        The radius is returned in pixel units.

        Parameters
        ----------
        frac : float
            fraction

        Returns
        -------
        float : radius
        """
        if self.is_null:
            return 0
        return self._integ_prof.evaluate(frac)
