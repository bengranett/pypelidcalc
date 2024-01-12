#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

import time
import logging

import numpy as np
cimport numpy as np

from libc cimport math
from cython.view cimport array as cvarray

cimport cython_gsl as gsl

from pypelidcalc.cutils.rootfinder cimport RootFinder
from . import _profile_calc as pc
from . cimport _profile_calc as pc


ctypedef struct bulgy_disk_params:
    double bulge_scale
    double disk_scale
    double bulge_fraction
    double frac


cpdef double bulgy_disk_profile(double r, double bulge_scale, double disk_scale, double bulge_fraction) noexcept nogil:
    """ """
    cdef double y1, y2

    y1 = 0
    y2 = 0

    if bulge_fraction < 1:
        y2 = pc.disk_profile(r, disk_scale) * (1 - bulge_fraction)

    if bulge_fraction > 0:
        y1 = pc.bulge_profile(r, bulge_scale) * bulge_fraction

    return y1 + y2


cdef double _integrated(double r, double bulge_scale, double disk_scale, double bulge_fraction, double *deriv) noexcept nogil:
    """ """
    cdef double y1, y2, d1, d2
    cdef double *p1
    cdef double *p2

    if r <= 0:
        if deriv:
            deriv[0] = 0
        return 0

    y1 = 0
    y2 = 0
    d1 = 0
    d2 = 0

    p1 = NULL
    p2 = NULL

    if deriv:
        p1 = &d1
        p2 = &d2

    if bulge_fraction < 1:
        y2 = pc._disk_integrated(r, disk_scale, p2) * (1 - bulge_fraction)
        if deriv:
            d2 *= 1 - bulge_fraction

    if bulge_fraction > 0:
        y1 = pc._bulge_integrated(r, bulge_scale, p1) * bulge_fraction
        if deriv:
            d1 *= bulge_fraction

    if deriv:
        deriv[0] = d1 + d2

    return y1 + y2

def bulgy_disk_integrated(r, bulge_scale, disk_scale, bulge_fraction):
    return _integrated(r, bulge_scale, disk_scale, bulge_fraction, NULL)


cdef double _f(double r, void * p) noexcept nogil:
    """ """
    cdef bulgy_disk_params * params = <bulgy_disk_params *>p;
    return _integrated(r, params.bulge_scale, params.disk_scale, params.bulge_fraction, NULL) - params.frac

cdef double _df(double r, void * p) noexcept nogil:
    """ """
    cdef double f, df
    cdef bulgy_disk_params * params = <bulgy_disk_params *>p;
    f = _integrated(r, params.bulge_scale, params.disk_scale, params.bulge_fraction, &df) - params.frac
    return df

cdef void _fdf(double r, void * p, double *f, double *df) noexcept nogil:
    """ """
    cdef bulgy_disk_params * params = <bulgy_disk_params *>p;
    f[0] = _integrated(r, params.bulge_scale, params.disk_scale, params.bulge_fraction, df) - params.frac


cpdef double[:] bulgy_disk_radius(
                            double [:] bulge_scale,
                            double [:] disk_scale,
                            double [:] bulge_fraction,
                            double [:] axis_ratio,
                            double frac=0.5,
                            double low=0,
                            double high=10,
                            int maxiter=20,
                            double tol_abs=1e-3,
                            double tol_rel=1e-2):
    """ Compute half-light radius with root-finding method.

    Parameters
    ----------
    scales : list
        List of parameter tuples in which the first is the scale length
        [(scale1,), (scale2,), ...]
    weights : list
        List of weights for mixing profiles [w1, w2, ...]
    frac : float
        light-fraction
    method : function
        root-finding function to call
    a : float
        lower bound
    b : float
        upper bound
    kwargs : keyword arguments
        additonal options to pass to the root-finding routine

    Returns
    -------
    radius : np.ndarray
    """
    cdef int i, n, status
    cdef double guess
    cdef double [:] x

    cdef gsl.gsl_function_fdf F
    cdef RootFinder R

    cdef bulgy_disk_params params

    F.fdf = &_fdf
    F.f = &_f
    F.df = &_df
    F.params = &params

    params.frac = frac

    R = RootFinder.create(&F, tol_abs, tol_rel, maxiter)

    n = bulge_scale.shape[0]
    assert disk_scale.shape[0] == n
    assert bulge_fraction.shape[0] == n
    assert axis_ratio.shape[0] == n

    x = cvarray(shape=(n,), itemsize=sizeof(double), format='d')

    status = gsl.GSL_FAILURE

    with nogil:
        for i in range(n):
            params.bulge_scale = bulge_scale[i]
            params.disk_scale = disk_scale[i]
            params.bulge_fraction = bulge_fraction[i]
            if bulge_fraction[i]<0.1:
                guess = disk_scale[i]
            elif bulge_fraction[i]>0.9:
                guess = bulge_scale[i]
            else:
                guess = min(bulge_scale[i], disk_scale[i])
            status = R.solve(&params, guess, &x[i])
            if status != gsl.GSL_SUCCESS:
                break

    if status != gsl.GSL_SUCCESS:
        logging.critical("root finding error: %i %f %f %f", status, bulge_scale[i], disk_scale[i], bulge_fraction[i])
        raise Exception

    return np.array(x)


cpdef double [:,:] bulgy_disk_sample(double bulge_scale, double disk_scale, double bulge_fraction, double axis_ratio, double pa, int n, int isotropize):
    """ """
    cdef int nbulge, ndisk
    cdef double [:,:] x

    cdef pc._Bulge bulge
    cdef pc._Disk disk

    if axis_ratio <= 0:
        raise ValueError("axis ratio must be greater than 0")

    nbulge = <int>(bulge_fraction * n)
    ndisk = n - nbulge

    x = cvarray(shape=(n, 2), itemsize=sizeof(double), format='d')

    if nbulge > 0:
        bulge = pc.Bulge()
        x[:nbulge] = bulge.sample(bulge_scale, axis_ratio, pa, nbulge, isotropize)
    if ndisk > 0:
        disk = pc.Disk()
        x[nbulge:] = disk.sample(disk_scale, axis_ratio, pa, ndisk, isotropize)

    return np.array(x)

