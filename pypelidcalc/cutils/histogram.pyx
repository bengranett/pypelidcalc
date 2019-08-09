#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

import time
import logging

import numpy as np
cimport numpy as np

from cython.view cimport array as cvarray


cdef int histogram_bin(double x, double start, double step) nogil:
    """ """
    return <int>  ((x - start) / step)


cpdef double [:] histogram(double [:] x, double [:] bins):
    """ Compute the histogram of 1-dimensional data.

    Notes
    -----
    Assumes that bins are regular.

    Parameters
    ----------
    x : numpy.ndarray
        data vector
    bins : nump.ndarray
        bin edges

    Returns
    -------
    histogram : numpy.ndarray
        histogram
    """
    cdef int n, m, i, j
    cdef double step
    cdef double [:] h

    n = x.shape[0]
    m = bins.shape[0] - 1

    step = bins[1] - bins[0]

    h = cvarray(shape=(m,), itemsize=sizeof(double), format='d')

    with nogil:
        for j in range(m):
            h[j] = 0

        for i in range(n):
            if x[i] < bins[0]:
                continue
            if x[i] > bins[m]:
                continue
            j = <int> ((x[i] - bins[0]) / step)
            if j < 0:
                continue
            if j > m - 1:
                continue
            h[j] += 1

    return h


cdef int _histogram2d(double [:,:] x, double [:] bins_y, double [:] bins_x, double [:,:] h) nogil:
    """ Compute the histogram of 1-dimensional data.

    Notes
    -----
    Assumes that bins are regular.

    Parameters
    ----------
    x : numpy.ndarray
        data vector
    y : numpy.ndarray
        data vector
    bins_x : nump.ndarray
        bin edges
    bins_y : nump.ndarray
        bin edges

    Returns
    -------
    histogram : numpy.ndarray
        histogram
    """
    cdef int n, mx, my, i, jx, jy
    cdef double step_x, step_y

    n = x.shape[0]
    mx = bins_x.shape[0] - 1
    my = bins_y.shape[0] - 1

    if h.shape[0] != my or h.shape[1] != mx:
        return FAILURE

    step_x = bins_x[1] - bins_x[0]
    step_y = bins_y[1] - bins_y[0]

    for i in range(n):
        if x[i,0] < bins_x[0]:
            continue
        if x[i,0] > bins_x[mx]:
            continue
        if x[i,1] < bins_y[0]:
            continue
        if x[i,1] > bins_y[my]:
            continue
        jx = <int>((x[i,0] - bins_x[0]) / step_x)
        jy = <int>((x[i,1] - bins_y[0]) / step_y)
        if jx < 0:
            continue
        if jx > mx - 1:
            continue
        if jy < 0:
            continue
        if jy > my - 1:
            continue
        h[jy, jx] += 1

    return SUCCESS

cpdef double [:,:] histogram2d(double [:,:] x, double [:] bins_y, double [:] bins_x):
    """ """
    cdef int i, j, mx, my, status
    cdef double[:,:] h
    
    mx = bins_x.shape[0] - 1
    my = bins_y.shape[0] - 1

    h = cvarray(shape=(my, mx), itemsize=sizeof(double), format='d')

    with nogil:
        for i in range(my):
            for j in range(mx):
                h[i,j] = 0

        status = _histogram2d(x, bins_y, bins_x, h)

    if status != SUCCESS:
        raise ValueError("_histogram2d returned error code %i"%status)

    return h
