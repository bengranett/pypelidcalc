cimport libc.math as math

cimport cython_gsl as gsl

cdef class RootFinder:
    cdef int maxiter
    cdef double tol_abs
    cdef double tol_rel
    cdef gsl.gsl_function_fdf * F
    cdef gsl.gsl_root_fdfsolver * solver

    @staticmethod
    cdef create(gsl.gsl_function_fdf *, double tol_abs=*, double tol_rel=*, int maxiter=*)
    cdef int solve(self, void *params, double guess, double * result) nogil