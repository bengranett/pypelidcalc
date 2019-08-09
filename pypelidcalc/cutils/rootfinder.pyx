cimport cython_gsl as gsl
from libc cimport stdio
    

cdef class RootFinder:

    def __init__(self, double tol_abs=1e-2, double tol_rel=1e-3, int maxiter=20):
        """ """
        self.tol_abs = tol_abs
        self.tol_rel = tol_rel
        self.maxiter = maxiter

    @staticmethod
    cdef create(gsl.gsl_function_fdf * F, double tol_abs=1e-2, double tol_rel=1e-3, int maxiter=20):
        me = RootFinder(tol_abs, tol_rel, maxiter)
        me.F = F
        return me

    def __cinit__(self, *args, **kwargs):
        """ """
        self.solver = gsl.gsl_root_fdfsolver_alloc(gsl.gsl_root_fdfsolver_newton)

    def __dealloc__(self):
        """ """
        gsl.gsl_root_fdfsolver_free(self.solver)

    cdef int solve(self, void *params, double guess, double * result) nogil:
        """ """
        cdef int loop, status
        cdef double root,x0
        cdef int maxiter = self.maxiter
        cdef double tol_abs = self.tol_abs
        cdef double tol_rel = self.tol_rel
        cdef gsl.gsl_root_fdfsolver * solver = self.solver

        root = guess

        self.F.params = params

        gsl.gsl_root_fdfsolver_set(self.solver, self.F, root)

        for loop in range(self.maxiter):
            status = gsl.gsl_root_fdfsolver_iterate(solver)
            if status != gsl.GSL_SUCCESS:
                break
            x0 = root
            root = gsl.gsl_root_fdfsolver_root(solver);
            status = gsl.gsl_root_test_delta (root, x0, 0, 1e-3);

            if (status == gsl.GSL_SUCCESS):
                break

        result[0] = root
        return status
