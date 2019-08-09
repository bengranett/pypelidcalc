#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
import numpy as np
cimport numpy as np
cimport libc.math as math

cimport cython
cimport cython_gsl as gsl

@cython.final
cdef class interpolate_regular:
	""" """
	def __init__(self, double[:] x, double[:] y, double fill_low=0, double fill_high=0):
		""" """
		cdef long i
		cdef double[:] dy
		cdef double step#, step2
		cdef long n = x.shape[0]
		self.n = n
		assert n > 1

		self.xmin = x[0]
		self.xmax = x[n-1]

		step = x[1] - x[0]
		self.step = step

		assert step > 0

		self.fill_low = fill_low
		self.fill_high = fill_high
		self.y = y

		dy = np.zeros(n-1, dtype=float)
		# step2 = step * step
		for i in range(n-1):
			dy[i] = y[i+1] - y[i]
		self.dy = dy


	cpdef double[:] evaluate(self, double [:] x):
		""" """
		cdef double f
		cdef long i, j, m, fi
		cdef double [:] dy = self.dy
		cdef double [:] y = self.y
		cdef double step = self.step
		cdef double xmin = self.xmin
		cdef double xmax = self.xmax

		m = x.shape[0]
		cdef double[:] out = np.zeros(m, dtype=float)

		with nogil:
			for i in range(m):

				if x[i] < xmin:
					out[i] = self.fill_low
					continue

				if x[i] > xmax:
					out[i] = self.fill_high
					continue

				f = (x[i] - xmin) / step

				fi = <long> math.floor(f)
				f = f - fi

				out[i] = dy[fi] * f + y[fi]

		return out

	cpdef double[:] product(self, double[:] x, double[:] out):
		""" """
		cdef double f
		cdef long i, j, m, fi
		cdef double [:] dy = self.dy
		cdef double [:] y = self.y
		cdef double step = self.step
		cdef double xmin = self.xmin
		cdef double xmax = self.xmax

		m = x.shape[0]

		with nogil:
			for i in range(m):

				if x[i] < xmin:
					out[i] = out[i] * self.fill_low
					continue

				if x[i] > xmax:
					out[i] = out[i] * self.fill_high
					continue

				f = (x[i] - xmin) / step

				fi = <long> math.floor(f)
				f = f - fi

				out[i] = out[i] * (dy[fi] * f + y[fi])

		return out


	cpdef double scalar(self, double x) nogil:
		""" """
		cdef double f
		cdef long i, j, fi
		cdef double [:] dy = self.dy
		cdef double [:] y = self.y
		cdef double step = self.step
		cdef double xmin = self.xmin
		cdef double xmax = self.xmax

		if x < xmin:
			return self.fill_low

		if x > xmax:
			return self.fill_high

		f = (x - xmin) / step

		fi = <long> math.floor(f)
		f = f - fi

		return dy[fi] * f + y[fi]


cdef class interp1d:

	def __init__(self, double[::1] x, double[::1] y, double fill_low=0, double fill_high=0):
		""" """
		cdef int res, n

		assert x.shape[0] == y.shape[0]

		n = x.shape[0]

		assert n > 1

		res = gsl.gsl_interp_init(self.interp, &x[0], &y[0], n)

		self.n = n
		self.x = x
		self.y = y
		self.fill_low = fill_low
		self.fill_high = fill_high

	def __cinit__(self, double[::1] x, double[::1] y, double fill_low=0, double fill_high=0):
		""" """
		n = x.shape[0]
		self.accel = gsl.gsl_interp_accel_alloc()
		self.interp = gsl.gsl_interp_alloc(gsl.gsl_interp_linear, n)

	def __dealloc__(self):
		""" free allocations """
		gsl.gsl_interp_free (self.interp)
		gsl.gsl_interp_accel_free(self.accel)

	cdef double evaluate(self, double xnew) nogil:
		""" """
		cdef int res
		cdef double ynew
		cdef double [:] x = self.x
		cdef double [:] y = self.y
		cdef gsl.gsl_interp_accel * accel = self.accel 
		cdef gsl.gsl_interp * interp = self.interp

		if xnew < x[0]:
			ynew = self.fill_low
		elif xnew > x[self.n-1]:
			ynew = self.fill_high
		else:
			res = gsl.gsl_interp_eval_e(interp, &x[0], &y[0], xnew, accel, &ynew)
			# if res != gsl.GSL_SUCCESS:
				# raise InterpError(res)

		return ynew

	def __call__(self, double xnew):
		return self.evaluate(xnew)

class InterpError(Exception):
	pass
