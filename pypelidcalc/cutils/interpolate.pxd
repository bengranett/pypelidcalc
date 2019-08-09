cimport cython_gsl as gsl

cdef class interpolate_regular:
	cdef long n
	cdef double [:] dy
	cdef double [:] y
	cdef double xmin, xmax, step
	cdef double fill_low, fill_high

	cpdef double[:] evaluate(self, double [:] x)
	cpdef double[:] product(self, double [:] x, double [:] out)
	cpdef double scalar(self, double x) nogil



cdef class interp1d:
	cdef int n
	cdef gsl.gsl_interp *interp
	cdef gsl.gsl_interp_accel *accel
	cdef double [:] x
	cdef double [:] y
	cdef double fill_low, fill_high

	cdef double evaluate(self, double xnew) nogil